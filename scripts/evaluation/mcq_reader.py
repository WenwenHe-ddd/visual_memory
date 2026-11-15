"""Shared pinned Qwen MCQ reader for prepared, signed evaluation inputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
OUT = ROOT / 'results/visual_memory_pipeline'

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))

def run(runtime=None):
    """Run the pinned reader; optionally reuse an identically configured model."""
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    protocol = json.loads((OUT / 'protocol.json').read_text())
    questions = read(OUT / 'qa_manifest.jsonl')
    for name, expected in protocol['input_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('input changed: ' + name)
    done = read(OUT / 'qa_responses.jsonl') if (OUT / 'qa_responses.jsonl').exists() else []
    if len(done) != len({r['id'] for r in done}):
        raise ValueError('duplicate QA response ID')
    qmap = {q['id']: q for q in questions}
    if any(r['experiment_signature'] != protocol['experiment_signature'] or
           r['payload_sha256'] != qmap[r['id']]['payload_sha256'] for r in done):
        raise ValueError('existing QA response signature mismatch')
    pending = [q for q in questions if q['id'] not in {r['id'] for r in done}]
    if not pending:
        return runtime
    torch.manual_seed(protocol['seed'])
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen reader requires CUDA')
    settings = {key: protocol.get(key) for key in
                ('reader_model_path', 'query_image_max_pixels', 'device_map', 'gpu_memory_gib')}
    if runtime is not None:
        if runtime['settings'] != settings:
            raise ValueError('cannot reuse a reader with different model or placement settings')
        processor, model = runtime['processor'], runtime['model']
    else:
        processor = AutoProcessor.from_pretrained(protocol['reader_model_path'], local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        processor.image_processor.size = {'shortest_edge': 65536,
                                          'longest_edge': protocol['query_image_max_pixels']}
        placement = {}
        if protocol.get('device_map'):
            limits = protocol['gpu_memory_gib']
            if len(limits) != torch.cuda.device_count():
                raise ValueError('one GPU memory budget is required per visible device')
            placement = {'device_map': protocol['device_map'],
                         'max_memory': {**{i: int(gib*1024**3) for i, gib in enumerate(limits)}, 'cpu': 0}}
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            protocol['reader_model_path'], dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True, **placement)
        if not placement:
            model = model.to('cuda:0')
        model.eval()
        runtime = {'settings': settings, 'processor': processor, 'model': model}
    input_device = model.get_input_embeddings().weight.device
    context_limit = min(model.config.text_config.max_position_embeddings,
                        protocol.get('context_budget', model.config.text_config.max_position_embeddings))
    dump(OUT / 'generation_environment.json', {'torch': torch.__version__,
         'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(0),
         'context_limit': context_limit, 'processor_size': processor.image_processor.size,
         'device_map': getattr(model, 'hf_device_map', None), 'input_device': str(input_device),
         'pid': os.getpid(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'cuda_allocator_config': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))})
    completed = len(done)
    for has_image in (False, True):
        batch = [q for q in pending if bool(q['query_image_paths']) == has_image]
        size = protocol['batch_size_query_images'] if has_image else protocol['batch_size_text']
        for start in range(0, len(batch), size):
            part = batch[start:start + size]
            messages, paths = [], []
            for q in part:
                paths.extend(q['query_image_paths'])
                content = [{'type': 'image'} for _ in q['query_image_paths']]
                content.append({'type': 'text', 'text': q['reader_text']})
                messages.append([{'role': 'system', 'content': protocol['reader_system_prompt']},
                                 {'role': 'user', 'content': content}])
            rendered = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        for m in messages]
            images = [Image.open(p).convert('RGB') for p in paths] if paths else None
            inputs = processor(text=rendered, images=images, padding=True,
                               return_tensors='pt').to(input_device)
            token_lengths = inputs['attention_mask'].sum(dim=1).tolist()
            if max(token_lengths) + protocol['max_new_tokens'] > context_limit:
                raise ValueError('reader context would truncate')
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False,
                                           max_new_tokens=protocol['max_new_tokens'],
                                           pad_token_id=processor.tokenizer.pad_token_id)
            tails = generated[:, inputs['input_ids'].shape[1]:]
            texts = processor.batch_decode(tails, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            with (OUT / 'qa_responses.jsonl').open('a') as f:
                for q, raw, tokens, length in zip(part, texts, tails.tolist(), token_lengths):
                    raw = raw.strip()
                    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw, flags=re.IGNORECASE)
                    eos = processor.tokenizer.eos_token_id
                    emitted = next((i + 1 for i, t in enumerate(tokens) if t == eos), len(tokens))
                    response = {'id': q['id'], 'raw_response': raw,
                                'predicted_choice': match.group(1).upper() if match else None,
                                'status': 'ok' if match else 'format_error',
                                'input_tokens': int(length), 'generated_tokens': emitted,
                                'hit_token_limit': emitted >= protocol['max_new_tokens'] and eos not in tokens,
                                'payload_sha256': q['payload_sha256'],
                                'experiment_signature': protocol['experiment_signature']}
                    f.write(json.dumps(response, ensure_ascii=False) + '\n')
            completed += len(part)
            print(f'QA {completed}/{len(questions)}', flush=True)
    return runtime
