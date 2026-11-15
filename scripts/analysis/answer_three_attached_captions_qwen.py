"""Answer one question from all three frozen Qwen captions, without images."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
SOURCE = ROOT / 'results/qwen_three_attached_captions_20261031'
QUESTION = ('In the painting where the reclining statue is surrounded by a circular pool of water, '
            'what appears in the background behind the tower?')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--two-words', action='store_true',
                        help='Require exactly two English words and no explanation')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    source_protocol = json.loads((SOURCE / 'protocol.json').read_text())
    records = [json.loads((SOURCE / f'image_{index}.json').read_text()) for index in (1, 2, 3)]
    for index, record in enumerate(records, 1):
        if record['image_index'] != index or record['image_sha256'] != source_protocol['inputs'][index-1]['sha256']:
            raise ValueError('Caption provenance mismatch')
    context = '\n\n'.join(f'Image {record["image_index"]}: {record["caption"]}' for record in records)
    constraint = '\nAnswer using exactly two English words. No explanation.' if args.two_words else ''
    prompt = f'Context:\n{context}\n\nQuestion: {QUESTION}{constraint}\nAnswer:'
    torch.manual_seed(42)
    torch.set_num_threads(8)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(source_protocol['model'], local_files_only=True)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], return_tensors='pt').to(device)
    print(f'Loading Qwen on {device}; input_tokens={inputs["input_ids"].shape[1]}', flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source_protocol['model'], dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to(device).eval()
    start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                   pad_token_id=processor.tokenizer.pad_token_id)
    tokens = generated[0, inputs['input_ids'].shape[1]:]
    answer = processor.decode(tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False).strip()
    result = {'model': source_protocol['model'], 'device': device,
              'question': QUESTION, 'prompt': prompt, 'answer': answer,
              'source_caption_sha256': {f'image_{i}.json': hashlib.sha256((SOURCE / f'image_{i}.json').read_bytes()).hexdigest()
                                        for i in (1, 2, 3)},
              'do_sample': False, 'seed': 42, 'max_new_tokens': 128,
              'generated_tokens': len(tokens),
              'hit_token_limit': len(tokens) >= 128 and tokens[-1].item() != processor.tokenizer.eos_token_id,
              'elapsed_seconds': round(time.monotonic() - start, 2)}
    output_name = 'qa_replay_two_words.json' if args.two_words else 'qa_replay.json'
    (SOURCE / output_name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('answer', 'generated_tokens', 'hit_token_limit', 'elapsed_seconds')},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
