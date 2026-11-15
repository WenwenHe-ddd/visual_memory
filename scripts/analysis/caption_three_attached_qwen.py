"""Generate three attached-image captions with the frozen local Qwen3-VL-8B.

Uses the original natural-observation prompt and image processor settings.
Falls back to CPU if this runtime has no CUDA device. Writes each result as
soon as it finishes so an interrupted run retains completed captions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
OUT = ROOT / 'results/qwen_three_attached_captions_20261031'
IMAGES = [
    Path('/storage/hewenwen/.codex/attachments/c4754cdb-d3c3-4237-b022-4a39b38dbbd5/codex-clipboard-3c6dbf78-6390-4fca-8b97-0c8f0b183fc0.png'),
    Path('/storage/hewenwen/.codex/attachments/85a48057-b04e-4f37-90fd-4d10aaf23dcf/codex-clipboard-a8dbda8d-e852-4a90-9238-ba11646aa8ca.png'),
    Path('/storage/hewenwen/.codex/attachments/f09e19e8-bf4d-40b7-9d24-b16145b509cb/codex-clipboard-4a089ece-a70a-40b0-8d8e-a6e4bb01fd87.png'),
]


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    OUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / 'results/natural_observation/protocol.json').read_text())
    torch.manual_seed(baseline['seed'])
    torch.set_num_threads(8)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path = baseline['caption_model_path']
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = {'shortest_edge': 65536,
                                      'longest_edge': baseline['image_max_pixels']}
    manifest = {
        'model': model_path, 'prompt': baseline['caption_prompt'],
        'seed': baseline['seed'], 'dtype': 'bfloat16', 'do_sample': False,
        'max_new_tokens': baseline['caption_max_new_tokens'],
        'processor_size': processor.image_processor.size, 'device': device,
        'torch': torch.__version__,
        'inputs': [{'index': i, 'path': str(path),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for i, path in enumerate(IMAGES, 1)],
    }
    (OUT / 'protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Loading {model_path} on {device}', flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to(device).eval()
    print('Model ready', flush=True)

    for index, path in enumerate(IMAGES, 1):
        output_path = OUT / f'image_{index}.json'
        if output_path.exists():
            old = json.loads(output_path.read_text())
            if old['image_sha256'] != manifest['inputs'][index - 1]['sha256']:
                raise ValueError(f'Existing caption source mismatch: {output_path}')
            print(f'Skipping completed image {index}', flush=True)
            continue
        start = time.monotonic()
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': baseline['caption_prompt']}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        with Image.open(path) as image:
            inputs = processor(text=[rendered], images=[image.convert('RGB')],
                               return_tensors='pt').to(device)
        print(f'Image {index}: input_tokens={inputs["input_ids"].shape[1]}', flush=True)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=baseline['caption_max_new_tokens'],
                do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
        tokens = generated[0, inputs['input_ids'].shape[1]:]
        caption = processor.decode(tokens, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False).strip()
        hit_limit = len(tokens) >= baseline['caption_max_new_tokens'] and (
            tokens[-1].item() != processor.tokenizer.eos_token_id)
        if not caption:
            raise ValueError(f'Image {index} returned empty caption')
        result = {'image_index': index, 'image_path': str(path),
                  'image_sha256': manifest['inputs'][index - 1]['sha256'],
                  'caption': caption, 'generated_tokens': len(tokens),
                  'hit_token_limit': hit_limit,
                  'elapsed_seconds': round(time.monotonic() - start, 2)}
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(f'Image {index} done in {result["elapsed_seconds"]}s; '
              f'tokens={len(tokens)}; hit_limit={hit_limit}\n{caption}', flush=True)


if __name__ == '__main__':
    main()
