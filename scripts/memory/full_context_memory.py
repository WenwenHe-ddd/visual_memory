"""FC(T): task-local dialogue history with the existing Qwen image captions."""
import hashlib
import json
from pathlib import Path

from scripts.data.generate import signature as caption_signature

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def captioned_histories(natural_root, config, tasks):
    """Whitelist dialogue fields; never include supplied captions or QA gold."""
    captions = read_jsonl(natural_root / 'captions.jsonl')
    expected = caption_signature(config)
    if any(r['protocol_signature'] != expected or not r['caption'].strip() for r in captions):
        raise ValueError('missing captions or incompatible natural caption generation configuration')
    by_id = {r['id']: r['caption'] for r in captions}
    if len(by_id) != len(captions):
        raise ValueError('duplicate caption IDs')
    memories = read_jsonl(natural_root / 'memories.jsonl')
    by_round = {r['id']: r for r in memories}
    if len(by_round) != len(memories):
        raise ValueError('duplicate memory IDs')
    images = read_jsonl(natural_root / 'images.jsonl')
    image_ids = {rel: r['id'] for r in images for rel in r['source_paths']}
    sources = [natural_root / name for name in
               ('protocol.json', 'captions.jsonl', 'memories.jsonl', 'images.jsonl')]
    histories = {}
    for task in sorted(tasks):
        candidates = [s for s in config['source_files']
                      if Path(s['path']).name == task + '_Open.json']
        if len(candidates) != 1:
            raise ValueError('expected one pinned dialogue source for ' + task)
        source = candidates[0]
        path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError('dialogue source changed: ' + str(path))
        sources.append(path)
        rows, seen = [], set()
        for session in json.loads(path.read_text())['multi_session_dialogues']:
            for turn in session['dialogues']:
                mid = task + '/' + turn['round']
                if mid in seen:
                    raise ValueError('duplicate dialogue round: ' + mid)
                seen.add(mid)
                lines = [f'Session: {session["session_id"]}; Date: {session.get("date", "")}',
                         f'Round: {turn["round"]}', 'User: ' + turn.get('user', '')]
                refs = turn.get('input_image') or []
                if isinstance(refs, str):
                    refs = [refs]
                if refs:
                    ids = [image_ids[rel] for rel in refs]
                    if mid not in by_round or ids != by_round[mid]['images']:
                        raise ValueError('natural image manifest disagrees with dialogue: ' + mid)
                    for i, image_id in enumerate(ids, 1):
                        if image_id not in by_id:
                            raise ValueError('missing Qwen caption: ' + image_id)
                        lines.append(f'Image {i} description: ' + by_id[image_id])
                elif mid in by_round:
                    raise ValueError('image-bearing memory has no dialogue image: ' + mid)
                lines.append('Assistant: ' + turn.get('assistant', ''))
                rows.append({'id': mid, 'text': '\n'.join(lines), 'has_image': bool(refs)})
        expected_rounds = {r['id'] for r in memories if r['task'] == task}
        if not rows or {r['id'] for r in rows if r['has_image']} != expected_rounds:
            raise ValueError('dialogue image round coverage mismatch: ' + task)
        histories[task] = rows
    return histories, sources


def format_history(question, rounds):
    return ('Full dialogue history (chronological; historical images replaced by captions):\n\n'
            + '\n\n'.join(r['text'] for r in rounds)
            + '\n\nQuestion:\n' + question['question'] + '\n\nOptions:\n'
            + '\n'.join(label + '. ' + question['options'][label] for label in 'ABCD')
            + '\n\nAnswer with one letter only.')


def fit_history(question, rounds, count_tokens, budget, max_new_tokens):
    """Remove complete oldest rounds; protect question/options and generation space."""
    for start in range(len(rounds) + 1):
        text = format_history(question, rounds[start:])
        tokens = count_tokens(text)
        if tokens + max_new_tokens <= budget:
            return text, rounds[start:], [r['id'] for r in rounds[:start]], tokens
    raise ValueError('question and query images exceed FC(T) context budget')


def prepare_full_context(args, questions, natural_config, baseline):
    from PIL import Image
    from transformers import AutoProcessor
    from scripts.evaluation import mcq_reader as reader

    histories, sources = captioned_histories(args.natural_root, natural_config,
                                            {q['task'] for q in questions})
    model_root = Path(baseline['reader_model_path'])
    model_config = json.loads((model_root / 'config.json').read_text())
    model_limit = model_config['text_config']['max_position_embeddings']
    budget = min(args.context_budget, model_limit)
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    processor.image_processor.size = {'shortest_edge': 65536,
                                      'longest_edge': baseline['query_image_max_pixels']}
    rows = []
    for q in questions:
        images = []
        for path in q['query_image_paths']:
            with Image.open(path) as img:
                images.append(img.convert('RGB'))

        def count_tokens(text):
            content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': text}]
            messages = [{'role': 'system', 'content': baseline['reader_system_prompt']},
                        {'role': 'user', 'content': content}]
            rendered = processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
            encoded = processor(text=[rendered], images=images or None, return_tensors='np')
            return int(encoded['attention_mask'].sum())

        history = histories[q['task']]
        text, kept, dropped, tokens = fit_history(q, history, count_tokens, budget,
                                                  baseline['max_new_tokens'])
        for img in images:
            img.close()
        image_rounds = [r['id'] for r in kept if r['has_image']]
        payload = {'system': baseline['reader_system_prompt'], 'user_text': text,
                   'query_images': [{'path': p, 'sha256': sha(Path(p))}
                                    for p in q['query_image_paths']]}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
        gold = set(q['gold_ids'])
        rows.append({**{key: q[key] for key in
                        ('id', 'task', 'x', 'y', 'question', 'options', 'gold_choice',
                         'gold_ids', 'query_image_paths')},
                     'top_memory_ids': image_rounds, 'top_query_scores': [],
                     'context_round_ids': [r['id'] for r in kept],
                     'dropped_round_ids': dropped, 'history_round_count': len(history),
                     'input_tokens': tokens, 'reader_text': text, 'payload_sha256': digest,
                     'any_gold_topk': bool(gold & set(image_rounds)),
                     'all_gold_topk': gold <= set(image_rounds)})
    sources += [Path(__file__), ROOT / 'scripts/visual_memory_pipeline.py',
                ROOT / 'scripts/evaluation/mcq_reader.py',
                ROOT / 'scripts/data/generate.py', args.benchmark_manifest,
                args.benchmark_manifest.parent / 'protocol.json']
    sources += [p for p in model_root.iterdir()
                if p.suffix in ('.json', '.txt', '.jinja') and p.is_file()]
    sources += [Path(p) for q in questions for p in q['query_image_paths']]
    keys = ('reader_model', 'reader_model_path', 'reader_system_prompt', 'seed', 'dtype',
            'max_new_tokens', 'query_image_max_pixels')
    protocol = {**{key: baseline[key] for key in keys},
                'study': 'MemEye FC(T) with existing local Qwen captions', 'mode': 'fc_t',
                'questions': len(rows), 'retrieval_top_k': None, 'retriever': None,
                'retrieval_scope': 'complete history of the original question task; no retrieval',
                'reader_memory_text': 'user/assistant dialogue, session/date/round, natural Qwen captions',
                'caption_model_path': natural_config['caption_model_path'],
                'caption_protocol_signature': caption_signature(natural_config),
                'memory_nodes': sum(r['has_image'] for h in histories.values() for r in h),
                'history_rounds': sum(len(h) for h in histories.values()),
                'context_budget_requested': args.context_budget, 'context_budget': budget,
                'context_policy': 'FIFO whole dialogue rounds; count actual reader processor tokens including query images; reserve max_new_tokens',
                'context_truncated_questions': sum(bool(r['dropped_round_ids']) for r in rows),
                'do_sample': False, 'batch_size_text': 1, 'batch_size_query_images': 1,
                'batch_size_reason': 'full histories evaluated individually to limit GPU memory',
                'coverage_fields': 'legacy any_gold_topk/all_gold_topk measure retained gold image rounds after FIFO, not retrieval',
                'input_hashes': {str(p.resolve()): sha(p) for p in sources}}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()
    protocol['experiment_signature'] = signature
    target = args.output_root / 'fc_t'
    old = read_jsonl(target / 'qa_responses.jsonl') if (target / 'qa_responses.jsonl').exists() else []
    by_id = {q['id']: q for q in rows}
    if any(r['experiment_signature'] != signature or r['id'] not in by_id or
           r['payload_sha256'] != by_id[r['id']]['payload_sha256'] for r in old):
        raise ValueError('existing responses have different inputs: ' + str(target))
    target.mkdir(parents=True, exist_ok=True)
    reader.dump(target / 'protocol.json', protocol)
    reader.jsonl(target / 'qa_manifest.jsonl', rows)
    print(f'fc_t: {len(rows)} questions; {protocol["history_rounds"]} history rounds; '
          f'{protocol["context_truncated_questions"]} truncated; '
          f'max input tokens {max(r["input_tokens"] for r in rows)}', flush=True)
