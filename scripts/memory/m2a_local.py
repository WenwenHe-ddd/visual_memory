"""Local Qwen tool-call transport and BGE/SigLIP2 embedding backends for M2A."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def safe_content(value):
    if isinstance(value, str) and value.startswith('data:image'):
        return {'data_uri_sha256': hashlib.sha256(value.encode()).hexdigest(), 'length': len(value)}
    if isinstance(value, list):
        return [safe_content(x) for x in value]
    if isinstance(value, dict):
        return {k: safe_content(v) for k, v in value.items()}
    return value


def parse_calls(raw, schemas):
    """Strict Qwen native tool-call parser; never guess malformed arguments."""
    # Like ChatOpenAI.invoke without tool binding, markup is plain assistant text.
    if not schemas:
        return raw, []
    allowed = {s['function']['name'] for s in schemas}
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', raw, re.S)
    if raw.count('<tool_call>') != len(blocks):
        raise ValueError('Unclosed Qwen tool_call block')
    calls = []
    for block in blocks:
        value = json.loads(block)
        if value.get('name') not in allowed or not isinstance(value.get('arguments'), dict):
            raise ValueError('Invalid tool name or arguments: ' + block)
        calls.append({'name': value['name'], 'args': value['arguments'],
                      'id': 'call_' + digest([raw, len(calls)])[:20], 'type': 'tool_call'})
    return re.sub(r'<tool_call>.*?</tool_call>', '', raw, flags=re.S).strip(), calls


class LocalQwen:
    def __init__(self, model_path, out, device='cuda:0', seed=42,
                 max_new_tokens=1200, image_max_pixels=262144):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError('M2A local Qwen requires host GPU access')
        torch.set_num_threads(8)
        torch.manual_seed(seed)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': image_max_pixels}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.schemas = []
        self.out = Path(out)
        self.stage = 'init'
        self.sample_id = None
        self.counter = [0]

    def bind_tools(self, tools, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        bound = copy.copy(self)
        bound.schemas = [convert_to_openai_tool(t) for t in tools]
        return bound

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        from PIL import Image
        prepared, images, image_audit = [], [], []
        role_map = {'human': 'user', 'ai': 'assistant', 'system': 'system', 'tool': 'tool'}
        for message in messages:
            entry = {'role': role_map[message.type], 'content': message.content}
            if getattr(message, 'tool_calls', None):
                entry['tool_calls'] = [{'type': 'function', 'function': {
                    'name': c['name'], 'arguments': c['args']}} for c in message.tool_calls]
            if isinstance(message.content, list):
                content = []
                for block in message.content:
                    if block.get('type') in ('image', 'image_url'):
                        ref = block.get('url') or block.get('image_url') or block.get('image')
                        if isinstance(ref, dict):
                            ref = ref['url']
                        if ref.startswith('data:'):
                            raw = base64.b64decode(ref.split(',', 1)[1])
                            im = Image.open(io.BytesIO(raw)).convert('RGB')
                            source = {'bytes_sha256': hashlib.sha256(raw).hexdigest()}
                        else:
                            raw = Path(ref).read_bytes()
                            im = Image.open(io.BytesIO(raw)).convert('RGB')
                            source = {'path': ref, 'bytes_sha256': hashlib.sha256(raw).hexdigest()}
                        images.append(im)
                        image_audit.append(source)
                        content.append({'type': 'image'})
                    elif block.get('type') == 'text':
                        content.append({'type': 'text', 'text': block['text']})
                    else:
                        raise ValueError('Unsupported content block: ' + repr(block))
                entry['content'] = content
            prepared.append(entry)
        rendered = self.processor.apply_chat_template(
            prepared, tools=self.schemas or None, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], images=images or None,
                                return_tensors='pt').to(self.device)
        length = inputs['input_ids'].shape[1]
        if length + self.max_new_tokens > 32768:
            raise ValueError(f'M2A input exceeds audited 32768-token context: {length}')
        self.counter[0] += 1
        started = time.time()
        print(f'LLM {self.counter[0]} {self.stage} {self.sample_id} input={length} images={len(images)}', flush=True)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id)
        tail = outputs[0, length:]
        raw = self.processor.decode(tail, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()
        record = {'call': self.counter[0], 'stage': self.stage, 'sample_id': self.sample_id,
                  'messages': safe_content([m.model_dump() for m in messages]),
                  'tools': self.schemas, 'rendered_prompt': rendered,
                  'prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                  'images': image_audit, 'input_tokens': length, 'output_tokens': len(tail),
                  'elapsed_seconds': time.time() - started, 'raw_response': raw,
                  'hit_token_limit': len(tail) >= self.max_new_tokens and
                      tail[-1].item() != self.processor.tokenizer.eos_token_id}
        try:
            if record['hit_token_limit']:
                raise ValueError('Qwen generation reached token limit')
            content, calls = parse_calls(raw, self.schemas)
            record['tool_calls'] = calls
            record['status'] = 'ok'
        except Exception as exc:
            record['status'] = 'error'
            record['error'] = str(exc)
            append_json(self.out / 'llm_calls.jsonl', record)
            raise
        append_json(self.out / 'llm_calls.jsonl', record)
        print(f'LLM done {record["elapsed_seconds"]:.1f}s tokens={len(tail)} tools={[c["name"] for c in calls]}', flush=True)
        return AIMessage(content=content, tool_calls=calls)


class LocalEmbeddings:
    def __init__(self, bge_path, siglip_path, out, device='cuda:0'):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.bge = SentenceTransformer(str(bge_path), device=device, local_files_only=True)
        self.siglip = AutoModel.from_pretrained(str(siglip_path), local_files_only=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(str(siglip_path), local_files_only=True)
        self.device, self.out = device, Path(out)
        self.cache = {}
        self.prefix = 'Represent this sentence for searching relevant passages: '
        assert self.bge.get_sentence_embedding_dimension() == 384

    def text(self, value, query=False):
        text = (self.prefix if query else '') + value
        key = ('bge', text)
        if key not in self.cache:
            length = len(self.bge.tokenizer(text, truncation=False)['input_ids'])
            self.cache[key] = self.bge.encode(text, normalize_embeddings=True).tolist()
            append_json(self.out / 'embedding_calls.jsonl', {
                'kind': 'bge_query' if query else 'bge_document', 'text': text,
                'tokens': length, 'limit': self.bge.max_seq_length,
                'truncated': length > self.bge.max_seq_length,
                'dimension': len(self.cache[key])})
        return self.cache[key]

    def visual(self, value, image=True):
        from PIL import Image
        key = ('siglip_image' if image else 'siglip_text', value)
        if key not in self.cache:
            if image:
                with Image.open(value) as im:
                    inputs = self.processor(images=[im.convert('RGB')], return_tensors='pt').to(self.device)
                method = self.siglip.get_image_features
            else:
                inputs = self.processor(text=[value], padding='max_length', max_length=64,
                                        truncation=True, return_tensors='pt').to(self.device)
                method = self.siglip.get_text_features
            with self.torch.inference_mode():
                vec = method(**inputs)
                if not isinstance(vec, self.torch.Tensor):
                    vec = vec.pooler_output
                vec = self.torch.nn.functional.normalize(vec.float(), dim=-1)
            self.cache[key] = vec[0].cpu().tolist()
            assert len(self.cache[key]) == 768
            append_json(self.out / 'embedding_calls.jsonl', {
                'kind': key[0], 'input': value, 'dimension': 768, 'normalized': True,
                'image_sha256': hashlib.sha256(Path(value).read_bytes()).hexdigest() if image else None})
        return self.cache[key]

    def install(self):
        from agent.stores import semantic
        owner = self

        class Text:
            def __init__(self, **kwargs):
                pass

            def embed_query(self, text):
                return owner.text(text, query=True)

            def embed_documents(self, texts):
                return [owner.text(t, query=False) for t in texts]

        class Visual:
            def __init__(self, **kwargs):
                pass

            def encode_image(self, path):
                return owner.visual(path, image=True)

            def embed_text(self, text):
                return owner.visual(text, image=False)

        semantic.TextEmbedding = Text
        semantic.MultimodalEmbedder = Visual
