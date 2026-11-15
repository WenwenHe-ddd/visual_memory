"""Signed paired Qwen MCQ evaluation, one GPU, batch one, fresh payload cache."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from scripts.analysis.evaluate_adaptive_changed_qa import read, lines, dump
from scripts.analysis.evaluate_all_master_answer_reader import verify


def run(output: Path, analyze_fn, resume=False):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    protocol = read(output / 'protocol.json'); verify(protocol)
    settings = protocol['reader_settings']
    manifest = lines(output / 'qa_manifest.jsonl')
    by_id = {r['record_id']: r for r in manifest}
    assert len(by_id) == len(manifest)
    response_path = output / 'responses.jsonl'
    if response_path.exists() and not resume:
        raise FileExistsError('Use --resume for this same signed experiment.')
    completed = lines(response_path) if response_path.exists() else []
    assert len(completed) == len({r['record_id'] for r in completed})
    cache = {}
    for row in completed:
        assert row['experiment_signature'] == protocol['experiment_signature']
        assert row['run_id'] == protocol['experiment_run_id']
        assert row['payload_sha256'] == by_id[row['record_id']]['payload_sha256']
        if row['fresh_inference_executed_for_this_record']:
            assert row['payload_sha256'] not in cache
            cache[row['payload_sha256']] = row
    done = {r['record_id'] for r in completed}
    pending = [r for r in manifest if r['record_id'] not in done]
    if not pending:
        analyze_fn(output)
        return
    gpu = protocol['placement']['gpu_index']
    assert torch.cuda.is_available() and gpu < torch.cuda.device_count()
    free = torch.cuda.mem_get_info(gpu)[0]
    assert free >= protocol['placement']['minimum_free_gib'] * 1024**3, 'Insufficient GPU headroom'
    torch.manual_seed(settings['seed']); torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(settings['reader_model_path'], local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': settings['query_image_max_pixels']}
    started = time.perf_counter()
    print(json.dumps({'stage': 'loading_reader', 'gpu_index': gpu, 'pending_records': len(pending)}), flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(settings['reader_model_path'],
        dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa', device_map={'': gpu})
    model.eval()
    device = model.get_input_embeddings().weight.device
    assert device == torch.device('cuda', gpu)
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    assert parameter_devices == [str(device)], parameter_devices
    execution_id = str(uuid.uuid4())
    environment = {'run_id': protocol['experiment_run_id'], 'execution_id': execution_id,
        'resumed': resume, 'already_completed_records': len(completed),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu_index': gpu, 'gpu': torch.cuda.get_device_name(gpu), 'free_gpu_bytes_before_loading': free,
        'threads': torch.get_num_threads(), 'device_map': getattr(model, 'hf_device_map', {'': gpu}),
        'parameter_devices': parameter_devices, 'input_device': str(device),
        'dtype': str(model.dtype), 'context_limit': model.config.text_config.max_position_embeddings,
        'processor_size': processor.image_processor.size, 'generation_config': model.generation_config.to_dict(),
        'pid': os.getpid(), 'load_seconds': time.perf_counter() - started}
    dump(output / f'generation_environment_{execution_id}.json', environment)
    if not completed: dump(output / 'generation_environment.json', environment)
    for count, row in enumerate(pending, len(completed) + 1):
        previous = cache.get(row['payload_sha256'])
        if previous is not None:
            response = {**previous, 'record_id': row['record_id'], 'condition': row['condition'],
                'question_id': row['id'], 'gold_choice': row['gold_choice'],
                'correct': previous['predicted_choice'] == row['gold_choice'],
                'fresh_inference_executed_for_this_record': False,
                'same_run_shared_inference_record_id': previous['record_id'], 'elapsed_seconds': 0.}
        else:
            content = [{'type': 'image'} for _ in row['query_image_paths']]
            content.append({'type': 'text', 'text': row['reader_text']})
            rendered = processor.apply_chat_template([
                {'role': 'system', 'content': settings['reader_system_prompt']},
                {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
            images = []
            for path in row['query_image_paths']:
                with Image.open(path) as image: images.append(image.convert('RGB'))
            encoded = processor(text=[rendered], images=images or None, padding=True, return_tensors='pt').to(device)
            input_tokens = int(encoded['attention_mask'].sum())
            assert input_tokens + settings['max_new_tokens'] <= model.config.text_config.max_position_embeddings
            torch.manual_seed(settings['seed']); started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=settings['max_new_tokens'],
                                           pad_token_id=processor.tokenizer.pad_token_id)
            tails = generated[:, encoded['input_ids'].shape[1]:]
            raw = processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            tokens = tails[0].tolist(); eos = processor.tokenizer.eos_token_id
            emitted = next((i + 1 for i, token in enumerate(tokens) if token == eos), len(tokens))
            match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw, re.IGNORECASE)
            choice = match.group(1).upper() if match else None
            response = {'record_id': row['record_id'], 'question_id': row['id'], 'condition': row['condition'],
                'raw_response': raw, 'predicted_choice': choice, 'gold_choice': row['gold_choice'],
                'correct': choice == row['gold_choice'], 'status': 'ok' if match else 'format_error',
                'input_tokens': input_tokens, 'generated_tokens': emitted, 'generated_token_ids': tokens,
                'hit_token_limit': emitted >= settings['max_new_tokens'] and eos not in tokens,
                'payload_sha256': row['payload_sha256'], 'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                'experiment_signature': protocol['experiment_signature'], 'run_id': protocol['experiment_run_id'],
                'execution_id': execution_id, 'elapsed_seconds': time.perf_counter() - started,
                'batch_size': 1, 'query_image_count': len(images), 'fresh_inference_executed_for_this_record': True,
                'same_run_shared_inference_record_id': None, 'historical_response_reuse': False}
            cache[row['payload_sha256']] = response
            del encoded, generated, tails
            for image in images: image.close()
        with response_path.open('a') as handle:
            handle.write(json.dumps(response, ensure_ascii=False) + '\n'); handle.flush()
        print(json.dumps({'completed': count, 'total': len(manifest), 'question': row['id'],
            'condition': row['condition'], 'answer': response['predicted_choice'],
            'fresh_execution': response['fresh_inference_executed_for_this_record'],
            'seconds': response['elapsed_seconds']}), flush=True)
    analyze_fn(output)
