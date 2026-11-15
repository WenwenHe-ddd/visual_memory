"""Pinned MemEye RFMemory core with local Qwen transport and explicit gold feedback.

The upstream store, recall, truncation, reflection prompt and optimize code are
unchanged. One fixed MCQ permutation is used for compatibility with local runs.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'scripts/vendor_memeye_reflexion'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_rows(path, rows):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    temp.replace(path)


def upstream_helpers():
    """Load unchanged pure adapter helpers without its unrelated router imports."""
    metadata = json.loads((VENDOR / 'UPSTREAM.json').read_text())
    for name, digest in metadata['files'].items():
        if sha(VENDOR / name) != digest:
            raise ValueError('pinned upstream source changed: ' + name)
    source = VENDOR / 'reflexion_method.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, os=os)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class LocalReflectionResult:
    """Deliver a batched local result through the core's original fast_run interface."""
    def __init__(self):
        self.prompt = None
        self.result = None

    def fast_run(self, prompt):
        if self.prompt != prompt or self.result is None:
            raise ValueError('reflection transport prompt/result mismatch')
        result = self.result
        self.prompt, self.result = None, None
        return result


def make_memory(observations, helpers=None):
    from scripts.vendor_memeye_reflexion.memengine.config.Config import ArgConfig
    from scripts.vendor_memeye_reflexion.memengine.memory.RFMemory import RFMemory
    helpers = helpers or upstream_helpers()
    config = helpers['_build_rfmemory_config'](
        {'context_word_limit': 6000}, llm_model='local-qwen',
        api_key='unused-local-transport', base_url='http://127.0.0.1:1/v1')
    # ArgConfig preserves the caller's GPU visibility; MemoryConfig would set it to ''.
    memory = RFMemory(SimpleNamespace(args=ArgConfig(config)))
    memory.optimize_op.reflector.llm = LocalReflectionResult()
    for observation in observations:
        memory.store(dict(observation))
    return memory


def trial_text(question, context, raw_answer, gold):
    # Exact upstream trial formatting and correctness policy, after answer generation.
    correct = raw_answer.strip().upper() == gold.upper()
    return (f'Question:\n{question}\n\nRetrieved context:\n{context}\n\n'
            f'Model answer:\n{raw_answer}\n\nGround truth:\n{gold}\n\n'
            f'Correct: {"yes" if correct else "no"}')


def reflection_prompt(memory, trial):
    from scripts.vendor_memeye_reflexion.memengine.function.Reflector import _format_prompt
    config = memory.optimize_op.reflector.config
    return _format_prompt(config.prompt, {
        'previous_insight': memory.insight['global_insight'],
        'new_trial': trial, 'example': config.example})


def apply_reflection(memory, trial, prompt, response):
    transport = memory.optimize_op.reflector.llm
    transport.prompt, transport.result = prompt, response
    memory.optimize(new_trial=trial)


def prepare(args):
    from scripts.memory.full_context_memory import captioned_histories
    helpers = upstream_helpers()
    natural = json.loads((args.natural_root / 'protocol.json').read_text())
    baseline = json.loads((args.benchmark_manifest.parent / 'protocol.json').read_text())
    questions = read_rows(args.benchmark_manifest)
    if args.task:
        questions = [q for q in questions if q['task'] == args.task]
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    if not questions or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('empty or duplicate question set')
    tasks = {q['task'] for q in questions}
    # Reuse the established source hash, caption-signature and image mapping validation.
    _, sources = captioned_histories(args.natural_root, natural, tasks)
    captions = {r['id']: r['caption'] for r in read_rows(args.natural_root / 'captions.jsonl')}
    image_ids = {p: r['id'] for r in read_rows(args.natural_root / 'images.jsonl')
                 for p in r['source_paths']}
    observations = {}
    for task in sorted(tasks):
        source = next(ROOT / s['path'] for s in natural['source_files']
                      if Path(s['path']).name == task + '_Open.json')
        data = json.loads(source.read_text())
        name = str((data.get('character_profile') or {}).get('name', '')).strip()
        speaker = f'user ({name})' if name else 'user'
        observations[task] = []
        for session in data['multi_session_dialogues']:
            for turn in session['dialogues']:
                refs = turn.get('input_image') or []
                if isinstance(refs, str):
                    refs = [refs]
                payload = {'user': turn.get('user', ''), 'assistant': turn.get('assistant', ''),
                           'raw': {'image_caption': [captions[image_ids[p]] for p in refs]}}
                text = helpers['build_reflexion_note_text'](payload, speaker, 'assistant')
                if text:
                    observations[task].append({'text': text,
                        'timestamp': str(session.get('date') or '').strip(),
                        'dialogue_id': turn['round'], 'session_id': session['session_id']})
    sanitized = []
    for q in questions:
        # Fixed presented choices; never copy a canonical answer from another rotation.
        row = {k: q[k] for k in ('id', 'task', 'x', 'y', 'question', 'options',
                                 'gold_choice', 'gold_ids', 'query_image_paths')}
        query_captions = []
        for p in q['query_image_paths']:
            image_id = sha(Path(p))
            query_captions.append(captions[image_id])
            sources.append(Path(p))
        row['query_image_captions'] = query_captions
        sanitized.append(row)
    keys = ('reader_model', 'reader_model_path', 'reader_system_prompt', 'seed',
            'do_sample', 'dtype', 'max_new_tokens')
    sources += [args.benchmark_manifest, args.benchmark_manifest.parent / 'protocol.json',
                Path(__file__), ROOT / 'scripts/memory/full_context_memory.py',
                ROOT / 'scripts/data/generate.py', VENDOR / 'UPSTREAM.json']
    sources += [p for p in VENDOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    model_root = Path(baseline['reader_model_path'])
    sources += [p for p in model_root.iterdir() if p.suffix in ('.json', '.txt', '.jinja')]
    protocol = {**{k: baseline[k] for k in keys},
        'study': 'MemEye Reflexion core with local Qwen; sequential gold-feedback evaluation',
        'mode': 'reflexion', 'upstream': json.loads((VENDOR / 'UPSTREAM.json').read_text()),
        'questions': len(questions), 'tasks': sorted(tasks),
        'question_order': [q['id'] for q in sanitized],
        'memory_rounds': sum(len(v) for v in observations.values()),
        'caption_model_path': natural['caption_model_path'],
        'context_word_limit': 6000,
        'recall': 'unchanged RFMemoryRecall; all observations plus insight; query-independent; keep last 6000 space-delimited pieces',
        'feedback_policy': 'after each answer, reveal presented gold_choice and correctness to reflection; carry insight within task; reset between tasks',
        'query_image_policy': 'existing Qwen captions, no native query images (upstream text-only policy)',
        'answer_prompt_policy': 'existing local system prompt; upstream assistant-role recalled context; question/options as final user message',
        'reflection_max_new_tokens': args.reflection_max_new_tokens,
        'task_batch_size': args.reflexion_batch_size,
        'model_placement': 'accelerate auto; weights capped at min(10 GiB, 60% free memory) per visible GPU; CPU offload allowed',
        'context_budget': min(128000, json.loads((model_root / 'config.json').read_text())['text_config']['max_position_embeddings']),
        'differences_from_paper': ['local Qwen caption/answer/reflection', 'one fixed MCQ permutation per question instead of four rotations',
            'local reader system prompt and 16-token answer budget', 'explicit reflection generation cap',
            'independent tasks batched; within-task order preserved'],
        'input_hashes': {str(p.resolve()): sha(p) for p in sources}}
    protocol['prepared_questions_sha256'] = hash_object(sanitized)
    protocol['observations_sha256'] = hash_object(observations)
    protocol['experiment_signature'] = hash_object(protocol)
    target = args.output_root / 'reflexion'
    old_path = target / 'protocol.json'
    if old_path.exists() and json.loads(old_path.read_text()) != protocol:
        raise ValueError('different existing Reflexion protocol; choose a new output root')
    target.mkdir(parents=True, exist_ok=True)
    write_json(old_path, protocol)
    write_json(target / 'observations.json', observations)
    write_rows(target / 'questions.jsonl', sanitized)
    print(f'Reflexion prepared: {len(questions)} questions, {protocol["memory_rounds"]} observations; gold-feedback enabled', flush=True)


class LocalQwen:
    def __init__(self, protocol):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        import transformers
        self.torch = torch
        self.protocol = protocol
        if not torch.cuda.is_available():
            raise RuntimeError('local Reflexion requires CUDA')
        torch.manual_seed(protocol['seed'])
        torch.set_num_threads(8)
        self.processor = transformers.AutoProcessor.from_pretrained(protocol['reader_model_path'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        weight_budget = {i: min(10 * 1024**3, int(torch.cuda.mem_get_info(i)[0] * 0.6))
                         for i in range(torch.cuda.device_count())}
        explicit_map = protocol.get('reader_device_map')
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            protocol['reader_model_path'], dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True, device_map='cpu' if explicit_map else 'auto',
            max_memory={**weight_budget, 'cpu': '64GiB'}).eval()
        if explicit_map:
            from accelerate import dispatch_model
            # Keep CPU modules resident there. Ordinary automatic offloading
            # would copy an entire CPU embedding matrix to GPU for every token.
            self.model = dispatch_model(self.model, explicit_map, main_device='cpu',
                                        skip_keys=self.model._skip_keys_device_placement)
        # Text-only inputs belong on the language embedding device.
        embedding = self.model.get_input_embeddings()
        self.input_device = (getattr(getattr(embedding, '_hf_hook', None), 'execution_device', embedding.weight.device)
                             if 'reader_device_map' in protocol else self.model.device)
        self.environment = {'torch': str(torch.__version__), 'transformers': transformers.__version__,
                            'gpu': torch.cuda.get_device_name(0),
                            'device_map': self.model.hf_device_map, 'weight_budget_bytes': weight_budget,
                            'device_map_policy': 'explicit_resident_cpu' if explicit_map else 'auto',
                            'input_device': str(self.input_device)}

    def generate(self, messages, max_new_tokens):
        rendered = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        inputs = self.processor(text=rendered, padding=True, return_tensors='pt').to(
            getattr(self, 'input_device', self.model.device))
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        if max(lengths) + max_new_tokens > self.protocol['context_budget']:
            raise ValueError('Reflexion input exceeds reader context budget')
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id)
        tails = output[:, inputs['input_ids'].shape[1]:]
        texts = self.processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.processor.tokenizer.eos_token_id
        result = []
        for raw, tokens, length in zip(texts, tails.tolist(), lengths):
            emitted = next((i + 1 for i, t in enumerate(tokens) if t == eos), len(tokens))
            result.append({'raw_response': raw.strip(), 'input_tokens': int(length),
                'generated_tokens': emitted, 'hit_token_limit': eos not in tokens and emitted >= max_new_tokens})
        return result


def answer_messages(q, memory, protocol, helpers):
    question = q['question'] + '\n\nOptions:\n' + '\n'.join(k + '. ' + q['options'][k] for k in 'ABCD')
    query = helpers['_question_with_image_caption'](
        {'question_image': q['query_image_paths'], 'image_caption': q['query_image_captions']}, question)
    context = str(memory.recall({'text': query})).strip()
    messages = [{'role': 'system', 'content': protocol['reader_system_prompt']},
                {'role': 'assistant', 'content': 'Retrieved memory context:\n' + context},
                {'role': 'user', 'content': query + '\n\nAnswer with one letter only.'}]
    return question, query, context, messages


def run(args):
    target = args.output_root / 'reflexion'
    protocol = json.loads((target / 'protocol.json').read_text())
    for p, digest in protocol['input_hashes'].items():
        if sha(Path(p)) != digest:
            raise ValueError('Reflexion source changed: ' + p)
    questions = read_rows(target / 'questions.jsonl')
    observations = json.loads((target / 'observations.json').read_text())
    if (hash_object(questions) != protocol['prepared_questions_sha256'] or
            hash_object(observations) != protocol['observations_sha256']):
        raise ValueError('prepared Reflexion data changed')
    helpers = upstream_helpers()
    memories = {t: make_memory(v, helpers) for t, v in observations.items()}
    signature = protocol['experiment_signature']
    states, pending = {}, {t: [] for t in memories}
    # Restore only committed, contiguous per-task state transitions.
    gaps = set()
    for q in questions:
        state_path = target / 'checkpoints' / (hash_object(q['id']) + '.json')
        if state_path.exists():
            if q['task'] in gaps:
                raise ValueError('non-contiguous task checkpoints')
            state = json.loads(state_path.read_text())
            memory = memories[q['task']]
            _, _, context, messages = answer_messages(q, memory, protocol, helpers)
            if (state['experiment_signature'] != signature or state['id'] != q['id'] or
                state['insight_before'] != memory.insight['global_insight'] or
                state['answer_payload_sha256'] != hash_object(messages)):
                raise ValueError('Reflexion checkpoint provenance mismatch')
            trial = trial_text(state['question_with_options'], context, state['answer']['raw_response'], q['gold_choice'])
            if state['reflection_prompt'] != reflection_prompt(memory, trial):
                raise ValueError('Reflexion checkpoint reflection mismatch')
            apply_reflection(memory, trial, state['reflection_prompt'], state['reflection']['raw_response'])
            states[q['id']] = state
        else:
            gaps.add(q['task'])
            pending[q['task']].append(q)
    if len(states) == len(questions):
        materialize(target, questions, states, protocol)
        return
    engine = LocalQwen(protocol)
    write_json(target / 'generation_environment.json', engine.environment)
    while any(pending.values()):
        active = [t for t in sorted(pending) if pending[t]][:protocol['task_batch_size']]
        batch = []
        for task in active:
            q = pending[task][0]
            memory = memories[task]
            question, query, context, messages = answer_messages(q, memory, protocol, helpers)
            batch.append({'q': q, 'memory': memory, 'question': question, 'query': query,
                          'context': context, 'messages': messages,
                          'pending_path': target / 'pending_answers' / (hash_object(q['id']) + '.json')})
        need = [b for b in batch if not b['pending_path'].exists()]
        if need:
            answers = engine.generate([b['messages'] for b in need], protocol['max_new_tokens'])
            for b, answer in zip(need, answers):
                write_json(b['pending_path'], {'id': b['q']['id'], 'experiment_signature': signature,
                    'payload_sha256': hash_object(b['messages']), 'answer': answer})
        for b in batch:
            saved = json.loads(b['pending_path'].read_text())
            if (saved['experiment_signature'] != signature or saved['id'] != b['q']['id'] or
                    saved['payload_sha256'] != hash_object(b['messages'])):
                raise ValueError('pending answer provenance mismatch')
            b['answer'] = saved['answer']
            b['trial'] = trial_text(b['question'], b['context'], b['answer']['raw_response'], b['q']['gold_choice'])
            b['prompt'] = reflection_prompt(b['memory'], b['trial'])
        reflections = engine.generate([[{'role': 'user', 'content': b['prompt']}] for b in batch],
                                      protocol['reflection_max_new_tokens'])
        for b, reflection in zip(batch, reflections):
            q, memory = b['q'], b['memory']
            prior = memory.insight['global_insight']
            apply_reflection(memory, b['trial'], b['prompt'], reflection['raw_response'])
            state = {'id': q['id'], 'task': q['task'], 'experiment_signature': signature,
                'question_with_options': b['question'], 'recall_query': b['query'],
                'recalled_context': b['context'], 'answer_messages': b['messages'],
                'answer_payload_sha256': hash_object(b['messages']), 'answer': b['answer'],
                'insight_before': prior, 'insight_after': memory.insight['global_insight'],
                'prior_insight_fully_visible': bool(prior) and prior in b['context'],
                'reflection_prompt': b['prompt'], 'reflection_payload_sha256': hash_object([{'role': 'user', 'content': b['prompt']}]),
                'reflection': reflection}
            write_json(target / 'checkpoints' / (hash_object(q['id']) + '.json'), state)
            states[q['id']] = state
            pending[q['task']].pop(0)
        print(f'Reflexion {len(states)}/{len(questions)}; completed answer + gold-feedback reflection', flush=True)
    materialize(target, questions, states, protocol)


def materialize(target, questions, states, protocol):
    manifests, responses, details = [], [], []
    for q in questions:
        state = states[q['id']]
        answer = state['answer']
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', answer['raw_response'], re.I)
        choice = match.group(1).upper() if match else None
        manifests.append({**q, 'answer_messages': state['answer_messages'],
                          'payload_sha256': state['answer_payload_sha256']})
        responses.append({'id': q['id'], **answer, 'predicted_choice': choice,
            'status': 'ok' if match else 'format_error',
            'experiment_signature': protocol['experiment_signature'],
            'payload_sha256': state['answer_payload_sha256']})
        details.append({'id': q['id'], 'task': q['task'], 'x': q['x'], 'y': q['y'],
            'correct': choice == q['gold_choice'], 'gold_choice': q['gold_choice'],
            'predicted_choice': choice, 'status': 'ok' if match else 'format_error',
            'answer_token_limit_hit': answer['hit_token_limit'],
            'reflection_token_limit_hit': state['reflection']['hit_token_limit'],
            'has_prior_insight': bool(state['insight_before']),
            'prior_insight_fully_visible': state['prior_insight_fully_visible'],
            'input_tokens': answer['input_tokens']})
    write_rows(target / 'qa_manifest.jsonl', manifests)
    write_rows(target / 'qa_responses.jsonl', responses)
    write_rows(target / 'per_question.jsonl', details)


def analyze(args):
    target = args.output_root / 'reflexion'
    protocol = json.loads((target / 'protocol.json').read_text())
    qs = read_rows(target / 'questions.jsonl')
    rows = read_rows(target / 'per_question.jsonl')
    if len(rows) != len(qs) or {r['id'] for r in rows} != {q['id'] for q in qs}:
        raise ValueError('incomplete Reflexion results')
    def count(group):
        return {'n': len(group), 'correct': sum(r['correct'] for r in group),
            'accuracy': sum(r['correct'] for r in group) / len(group),
            'format_errors': sum(r['status'] != 'ok' for r in group),
            'answer_token_limit_hits': sum(r['answer_token_limit_hit'] for r in group),
            'reflection_token_limit_hits': sum(r['reflection_token_limit_hit'] for r in group),
            'prior_insight_present_questions': sum(r['has_prior_insight'] for r in group),
            'prior_insight_fully_visible_questions': sum(r['prior_insight_fully_visible'] for r in group)}
    summary = {**count(rows), 'feedback_policy': protocol['feedback_policy'],
        'experiment_signature': protocol['experiment_signature'],
        'by_task': {t: count([r for r in rows if r['task'] == t]) for t in protocol['tasks']}}
    write_json(target / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
