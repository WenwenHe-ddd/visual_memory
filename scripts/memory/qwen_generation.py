"""Local greedy Qwen image/text generation for object and relation extraction."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))

class QwenGenerator:
    def __init__(self, model_path: str, max_pixels: int, max_new_tokens: int,
                 seed: int, picture_caption_prompt: str, caption_max_tokens: int,
                 *, device_map: str | None = None, max_memory: dict | None = None):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError('Qwen3-VL generation requires an available CUDA GPU')
        torch.manual_seed(seed)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.caption_max_tokens = caption_max_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.image_processor.size = {'shortest_edge': 65536,
                                               'longest_edge': max_pixels}
        placement = {} if device_map is None else {'device_map': device_map, 'max_memory': max_memory}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True, **placement)
        if device_map is None:
            self.model = self.model.to('cuda:0')
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.last_raw = None
        self.last_stage_outputs = {}
        self.picture_caption_prompt = picture_caption_prompt

    def _ask_plain(self, prompt: str, stage: str, max_new_tokens: int,
                   assistant_prefix: str = '', image_path: str | None = None,
                   image_last: bool = False, repetition_penalty: float | None = None) -> str:
        content = [{'type': 'text', 'text': prompt}]
        if image_path is not None:
            image_item = {'type': 'image'}
            if image_last:
                content.append(image_item)
            else:
                content.insert(0, image_item)
        messages = [{'role': 'user', 'content': content}]
        if assistant_prefix:
            messages.append({'role': 'assistant', 'content': [
                {'type': 'text', 'text': assistant_prefix}]})
        rendered = self.processor.apply_chat_template(messages, tokenize=False,
            continue_final_message=bool(assistant_prefix),
            add_generation_prompt=not assistant_prefix)
        if image_path is None:
            inputs = self.processor(text=[rendered], return_tensors='pt').to(self.input_device)
        else:
            from PIL import Image
            with Image.open(image_path) as im:
                inputs = self.processor(
                    text=[rendered], images=[im.convert('RGB')],
                    return_tensors='pt').to(self.input_device)
        generation_options = {} if repetition_penalty is None else {'repetition_penalty': repetition_penalty}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id, **generation_options)
        tokens = output[0, inputs['input_ids'].shape[1]:]
        continuation = self.processor.decode(
            tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        raw = (assistant_prefix + continuation).strip()
        self.last_raw = raw
        self.last_stage_outputs[stage] = raw
        if tokens.numel() >= max_new_tokens and tokens[-1].item() != self.processor.tokenizer.eos_token_id:
            raise ValueError(f'Qwen {stage} reached token limit')
        if not continuation.strip():
            raise ValueError(f'Qwen {stage} returned empty text')
        return raw
