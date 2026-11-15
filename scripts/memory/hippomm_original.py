"""Paired HippoMM reader ablation: frozen retrieval, retained representative pixels.

The completed no-original-image run supplies the memory, routes and candidates.
Only detailed final answers with selected images are regenerated. Fast answers
and empty-retrieval answers are reused exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

from scripts.memory import hippomm as base, hippomm_core as core, hippomm_inputs as inputs

ROOT = inputs.ROOT
METHOD = 'hippomm_representative_originals_frozen_retrieval'
MEMORY_FILES = ('events.jsonl', 'dialogues.jsonl', 'image_mapping.jsonl', 'visual_embeddings.npz')
FROZEN_FIELDS = ('task', 'route', 'fast', 'fast_parse_error', 'fast_context_event_ids',
                 'fast_context_round_ids', 'selected_event_ids', 'detail_candidates',
                 'ranked_candidates', 'context_round_ids', 'visual_query', 'query_encoding')


def code_manifest(directory):
    code = base.implementation_hashes() | {str(Path(__file__)): inputs.sha(__file__)}
    base.contract_at(Path(directory).parent / 'experiment_manifest.json',
                     {'method': METHOD, 'implementation': code,
                      'model': str(base.MODEL), 'clip': str(base.CLIP)})


def baseline(args):
    root = args.hippomm_baseline_root.resolve() / args.dataset
    base.code_manifest(root / 'offline')
    base.code_manifest(root / 'online')
    memory, events, vectors, dialogues = base.verify_memory(root / 'offline')
    manifest = json.loads((root / 'online/completion_manifest.json').read_text())
    questions = inputs.read_lines(root / 'online/questions.jsonl')
    responses = inputs.read_lines(root / 'online/qa_responses.jsonl')
    qmap, rmap = inputs.unique(questions), inputs.unique(responses)
    if (manifest['status'] != 'complete' or manifest['questions'] != len(questions)
            or set(qmap) != set(rmap) or memory['dataset'] != args.dataset):
        raise ValueError('A complete matching no-original-image baseline is required')
    return root, memory, events, vectors, dialogues, questions, responses


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError('Existing copied artifact differs: ' + str(destination))
    else:
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def build(args):
    root, memory, events, _, _, _, _ = baseline(args)
    _, images, _ = inputs.memory_inputs(args)
    by_ref = inputs.unique(images, 'image_ref')
    args.output.mkdir(parents=True, exist_ok=True)
    code_manifest(args.output)
    records = []
    for event in events:
        image = by_ref[event['representative_image']]
        representative_rounds = {ref['round_id'] for ref in event['source_images']
                                 if ref['relation'] == 'representative'}
        if (image['task'] != event['task'] or image['caption'] != event['caption']
                or set(image['source_round_ids']) != representative_rounds):
            raise ValueError('Representative source identity differs from baseline')
        source = Path(image['path'])
        if inputs.sha(source) != image['image_sha256']:
            raise ValueError('Source image changed before copying')
        relative = Path('images') / (event['event_id'] + source.suffix.lower())
        records.append({'event_id': event['event_id'], 'task': event['task'],
                        'image_ref': image['image_ref'], 'file': str(relative),
                        'bytes': source.stat().st_size})
        if args.hippomm_stage != 'prepare':
            copy_exact(source, args.output / relative)
    if args.hippomm_stage == 'prepare':
        base.dump(args.output / 'readiness.json', {'representative_images': len(records),
                  'source_image_bytes': sum(r['bytes'] for r in records)})
        return
    for name in MEMORY_FILES:
        copy_exact(root / 'offline' / name, args.output / name)
    base.write_lines(args.output / 'representative_images.jsonl', records)
    base.dump(args.output / 'memory.json', {'offline_method': 'hippomm_original',
              'method': METHOD, 'dataset': args.dataset, 'baseline_dir': str(root),
              'counts': memory['counts'], 'representative_images': len(records),
              'original_image_bytes': sum(r['bytes'] for r in records),
              'memory_artifact_bytes': sum((args.output / n).stat().st_size for n in MEMORY_FILES)
                                      + (args.output / 'representative_images.jsonl').stat().st_size,
              'image_policy': 'exact representative source files only; no rejected-image copies'})
    verify_memory(args.output)
    print(json.dumps({'stage': 'memory_complete', 'representative_images': len(records),
                      'original_image_bytes': sum(r['bytes'] for r in records)}), flush=True)


def verify_memory(directory):
    directory = Path(directory)
    code_manifest(directory)
    memory = json.loads((directory / 'memory.json').read_text())
    if memory['method'] != METHOD:
        raise ValueError('Not a representative-originals memory')
    events = inputs.read_lines(directory / 'events.jsonl')
    records = inputs.unique(inputs.read_lines(directory / 'representative_images.jsonl'), 'event_id')
    if set(records) != {e['event_id'] for e in events} or len(records) != memory['counts']['events']:
        raise ValueError('Every event must have exactly one stored representative image')
    for event in events:
        record = records[event['event_id']]
        path = (directory / record['file']).resolve()
        if (not path.is_relative_to((directory / 'images').resolve()) or not path.is_file()
                or path.stat().st_size != record['bytes'] or record['task'] != event['task']
                or record['image_ref'] != event['representative_image']):
            raise ValueError('Invalid representative image record')
    actual = {p.resolve() for p in (directory / 'images').iterdir() if p.is_file()}
    if actual != {(directory / r['file']).resolve() for r in records.values()}:
        raise ValueError('Unexpected image files in retained store')
    dialogues = inputs.unique(inputs.read_lines(directory / 'dialogues.jsonl'))
    return memory, inputs.unique(events, 'event_id'), records, dialogues


def prepare(args):
    root, _, _, _, _, questions, responses = baseline(args)
    memory, _, _, _ = verify_memory(args.memory_dir)
    if memory['baseline_dir'] != str(root) or memory['dataset'] != args.dataset:
        raise ValueError('Offline memory belongs to a different baseline')
    for name in MEMORY_FILES:
        if (args.memory_dir / name).read_bytes() != (root / 'offline' / name).read_bytes():
            raise ValueError('Frozen memory differs from baseline: ' + name)
    if args.question_ids:
        questions = [q for q in questions if q['id'] in args.question_ids
                     or q['id'].split('/')[-1] in args.question_ids]
    if args.task:
        questions = [q for q in questions if q['task'] == args.task]
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    if not questions:
        raise ValueError('No selected questions')
    qids = {q['id'] for q in questions}
    responses = [r for r in responses if r['id'] in qids]
    original = json.loads((root / 'online/protocol.json').read_text())['contract']
    if (args.hippomm_max_new_tokens != original['max_new_tokens']
            or args.hippomm_context_budget != original['context_budget']
            or args.top_k != original['top_k']):
        raise ValueError('Reader budget and Top-K must match the paired baseline')
    args.output.mkdir(parents=True, exist_ok=True)
    code_manifest(args.output)
    base.contract_at(args.output / 'protocol.json', {
        'method': METHOD, 'dataset': args.dataset, 'baseline_dir': str(root),
        'memory_dir': str(args.memory_dir.resolve()), 'questions': len(questions),
        'max_new_tokens': args.hippomm_max_new_tokens, 'context_budget': args.hippomm_context_budget,
        'evaluation_source': original['evaluation_source'], 'top_k': args.top_k,
        'retrieval_policy': 'reuse exact baseline routes, queries, candidates and contexts',
        'fast_policy': 'reuse exact baseline fast answers; no historical images',
        'detail_policy': 'same context plus selected representative originals; query images first; reuse empty-retrieval answers',
        'image_preprocessing': 'same local Qwen reader preprocessing as baseline'})
    for name, rows in [('questions.jsonl', questions), ('baseline_responses.jsonl', responses)]:
        path = args.output / name
        if path.exists() and inputs.read_lines(path) != rows:
            raise ValueError('Prepared baseline snapshot differs: ' + name)
        base.write_lines(path, rows)
    print(json.dumps({'stage': 'prepared', 'questions': len(questions),
                      'detailed_to_regenerate': sum(r['route'] == 'detailed' and bool(r['selected_event_ids'])
                                                   for r in responses)}), flush=True)


def load_run(args):
    code_manifest(args.output)
    memory, events, images, dialogues = verify_memory(args.memory_dir)
    protocol = json.loads((args.output / 'protocol.json').read_text())['contract']
    if (protocol['memory_dir'] != str(args.memory_dir.resolve()) or protocol['dataset'] != args.dataset
            or protocol['max_new_tokens'] != args.hippomm_max_new_tokens
            or protocol['context_budget'] != args.hippomm_context_budget or protocol['top_k'] != args.top_k):
        raise ValueError('Run arguments differ from prepared protocol')
    questions = inputs.read_lines(args.output / 'questions.jsonl')
    originals = inputs.unique(inputs.read_lines(args.output / 'baseline_responses.jsonl'))
    if len(questions) != protocol['questions'] or set(originals) != set(inputs.unique(questions)):
        raise ValueError('Frozen question/response coverage differs')
    return protocol, memory, questions, originals, events, images, dialogues


def reader_payload(question, trace, events, images, dialogues, memory_dir, instruction=''):
    ids = trace['selected_event_ids']
    if trace['route'] != 'detailed' or len(ids) > 5 or len(ids) != len(set(ids)):
        raise ValueError('Detailed reader requires at most five unique selected events')
    if any(events[e]['task'] != question['task'] for e in ids):
        raise ValueError('Cross-task representative image')
    paths = list(question['query_image_paths'])
    order = [{'image_number': i + 1, 'kind': 'question_image'} for i in range(len(paths))]
    for eid in ids:
        paths.append(str((Path(memory_dir) / images[eid]['file']).resolve()))
        order.append({'image_number': len(paths), 'kind': 'historical_representative',
                      'event_id': eid, 'image_ref': events[eid]['representative_image'],
                      'source_image_ids': list(dict.fromkeys(
                          r['image_id'] for r in events[eid]['source_images']
                          if r['relation'] == 'representative'))})
    evidence = {'question': question['question'], 'image_order': order,
                'memories': [core.event_context(events[e], dialogues) for e in ids]}
    if question.get('options'):
        evidence['options'] = question['options']
    prompt = ('Answer using only the retrieved memory evidence and attached images. '
              'The image_order list identifies each attached image in its order of appearance. '
              'Historical images show only their listed representative, not visually redundant aliases. '
              'Use the original representative pixels to inspect visual details. Respect dates and '
              'source image IDs; do not transfer those details to other images. '
              'If evidence is absent, acknowledge it. ' + instruction + '\n'
              + ('Return only A, B, C or D.\n' if question.get('options') else 'Return a concise answer.\n')
              + json.dumps(evidence, ensure_ascii=False))
    return prompt, paths, order


def check_trace(row, original):
    if any(row.get(key) != original.get(key) for key in FROZEN_FIELDS):
        raise ValueError('Frozen retrieval/context changed: ' + row['id'])
    if ((row['route'] == 'fast' or not row['selected_event_ids'])
            and row['answer'] != original['answer']):
        raise ValueError('Answer without historical image access changed')


def run(args):
    protocol, _, questions, originals, events, images, dialogues = load_run(args)
    path = args.output / 'qa_responses.jsonl'
    done = inputs.unique(inputs.read_lines(path)) if path.exists() else {}
    if not set(done) <= set(originals):
        raise ValueError('Unexpected response ID')
    for key, row in done.items():
        check_trace(row, originals[key])
    from scripts.evaluation.mem_gallery_master_reader import TASK_INSTRUCTIONS
    chat = base.LocalChat(args.output, protocol, args.device,
                          args.hippomm_max_new_tokens, args.hippomm_context_budget)
    try:
        for q in questions:
            if q['id'] in done:
                continue
            original = originals[q['id']]
            row = {**original, 'baseline_llm_calls': original['llm_calls'], 'llm_calls': [],
                   'answer_reused': original['route'] == 'fast' or not original['selected_event_ids'],
                   'historical_image_reads': 0,
                   'reader_image_order': [], 'elapsed_seconds': 0.0}
            if original['route'] == 'detailed' and original['selected_event_ids']:
                prompt, paths, order = reader_payload(q, original, events, images, dialogues,
                    args.memory_dir, TASK_INSTRUCTIONS.get(q['point'], ''))
                started = time.perf_counter()
                try:
                    row['answer'] = chat('answer_with_representatives', prompt, paths).strip()
                except ValueError as exc:
                    base.append(args.output / 'question_errors.jsonl', {'id': q['id'],
                                'error': str(exc), 'last_call_id': chat.last_call_id})
                    continue
                row.update(llm_calls=[chat.last_call_id], reader_image_order=order,
                           elapsed_seconds=time.perf_counter() - started,
                           historical_image_reads=len(original['selected_event_ids']))
            check_trace(row, original)
            base.append(path, row)
            print(json.dumps({'id': q['id'], 'route': row['route'], 'answer': row['answer'],
                              'representative_images': row['historical_image_reads']}), flush=True)
    finally:
        chat.close()


def analyze(args):
    protocol, memory, questions, originals, events, images, dialogues = load_run(args)
    responses = inputs.read_lines(args.output / 'qa_responses.jsonl')
    rmap = inputs.unique(responses)
    if set(rmap) != {q['id'] for q in questions}:
        raise ValueError('Incomplete answers; no final metrics written')
    source = args.hippomm_gallery_inputs if args.dataset == 'mem_gallery' else args.benchmark_manifest
    if str(source.resolve()) != protocol['evaluation_source']:
        raise ValueError('Evaluation source differs from baseline')
    gold = inputs.evaluation_gold(args)
    from scripts.evaluation.mem_gallery_master_reader import score_answer, TASK_INSTRUCTIONS
    rows = []
    qmap = inputs.unique(questions)
    for row in responses:
        old, q = originals[row['id']], qmap[row['id']]
        check_trace(row, old)
        if row['route'] == 'detailed' and old['selected_event_ids']:
            prompt, paths, order = reader_payload(q, old, events, images, dialogues,
                                  args.memory_dir, TASK_INSTRUCTIONS.get(q['point'], ''))
            if len(row['llm_calls']) != 1 or row['reader_image_order'] != order:
                raise ValueError('Detailed image/call trace differs')
            cache = json.loads((args.output / 'llm_cache' / (row['llm_calls'][0] + '.json')).read_text())
            if (cache['request']['prompt'] != prompt or cache['request']['query_image_paths'] != paths
                    or cache['response']['hit_token_limit']
                    or cache['response']['raw_response'].strip() != row['answer']
                    or row['historical_image_reads'] != len(old['selected_event_ids'])):
                raise ValueError('Reader did not use the specified representative images')
        elif row['llm_calls'] or row['historical_image_reads']:
            raise ValueError('Reused answers must not read historical images')
        g = gold[row['id']]
        if args.dataset == 'memeye':
            metrics = {'accuracy': float(row['answer'] == g['gold_choice']),
                       'format_error': row['answer'] not in ('A', 'B', 'C', 'D')}
            old_metrics = {'accuracy': float(old['answer'] == g['gold_choice'])}
        else:
            metrics = score_answer(row['answer'], g['answer'])
            old_metrics = score_answer(old['answer'], g['answer'])
        labels = set(g['gold_ids'])
        hit = len(labels & set(row['context_round_ids']))
        rows.append({'id': row['id'], 'task': row['task'], 'route': row['route'], **metrics,
                     'baseline_metrics': old_metrics, 'context_gold_hits': hit, 'gold_count': len(labels),
                     'historical_images': row['historical_image_reads']})
    names = ['accuracy'] if args.dataset == 'memeye' else ['em', 'f1', 'bleu']
    detail = [r for r in rows if r['route'] == 'detailed']
    summary = {'method': METHOD, 'questions': len(rows), 'model': str(base.MODEL),
               'fast_answers_reused': len(rows) - len(detail),
               'detailed_answers_regenerated': sum(r['historical_images'] > 0 for r in detail),
               'empty_retrieval_answers_reused': sum(r['historical_images'] == 0 for r in detail),
               'format_errors': sum(r.get('format_error', False) for r in rows),
               'context_source_micro_recall': sum(r['context_gold_hits'] for r in rows)
                                            / sum(r['gold_count'] for r in rows),
               'recall_unchanged': True, 'retrieval_context_and_routes_unchanged': True,
               'representative_images_stored': memory['representative_images'],
               'original_image_bytes': memory['original_image_bytes'],
               'memory_artifact_bytes': memory['memory_artifact_bytes'],
               'total_memory_bytes': memory['original_image_bytes'] + memory['memory_artifact_bytes'],
               'historical_image_attachments': sum(r['historical_images'] for r in rows),
               'baseline': {}, 'delta_percentage_points': {}, 'detailed_subset': {}}
    for name in names:
        summary[name] = sum(r[name] for r in rows) / len(rows)
        summary['baseline'][name] = sum(r['baseline_metrics'][name] for r in rows) / len(rows)
        summary['delta_percentage_points'][name] = 100 * (summary[name] - summary['baseline'][name])
        if detail:
            summary['detailed_subset'][name] = sum(r[name] for r in detail) / len(detail)
            summary['detailed_subset']['baseline_' + name] = sum(r['baseline_metrics'][name] for r in detail) / len(detail)
    base.write_lines(args.output / 'per_question.jsonl', rows)
    base.dump(args.output / 'summary.json', summary)
    base.dump(args.output / 'completion_manifest.json', {'status': 'complete', 'questions': len(rows),
              'paired_retrieval_verified': True, 'reader_image_payloads_verified': True,
              'artifacts': ['questions.jsonl', 'baseline_responses.jsonl', 'qa_responses.jsonl',
                            'per_question.jsonl', 'summary.json']})
    print(json.dumps(summary, indent=2), flush=True)


def add_arguments(parser):
    parser.add_argument('--hippomm-baseline-root', type=Path,
                        default=ROOT / 'results/hippomm_short_ids_20261104',
                        help='completed no-original HippoMM run used for frozen retrieval comparison')


def pipeline(args, parser):
    default = ROOT / 'results/debug_caption_master_answer'
    target = ROOT / 'results/visual_memory_pipeline/hippomm_original' / args.dataset
    if args.output == default / args.phase:
        args.output = target / args.phase
    if args.phase == 'offline':
        if args.task or args.limit or args.image_files or args.memory_source or args.from_start:
            parser.error('hippomm_original builds all representatives from the completed baseline')
        return build(args)
    if args.memory_path is not None:
        args.memory_dir = args.memory_path
    elif args.memory_dir == default / 'offline/memory':
        args.memory_dir = target / 'offline'
    args.top_k = 5 if args.top_k is None else args.top_k
    if (args.reader_backend != 'local' or args.online_weighting is not None or args.retrieval is not None
            or args.retrieval_scoring != 'master_answer_query' or args.query_image_captions
            or args.encode_queries or args.whitening_lambda is not None or args.reweight_alpha is not None
            or args.top_k != 5 or args.limit_questions < 0 or args.simplemem_backbone != 'local-qwen'
            or args.hippomm_confidence_threshold != .75 or args.hippomm_fallback_threshold != .4
            or args.hippomm_fallback_score != .6 or args.output.resolve() == args.memory_dir.resolve()):
        parser.error('hippomm_original requires the unchanged baseline retrieval and local Qwen reader')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('run', 'all'):
        run(args)
    if args.stage in ('analyze', 'all'):
        analyze(args)
