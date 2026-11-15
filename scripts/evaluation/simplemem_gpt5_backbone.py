"""Official SimpleMem with GPT-5 captions, construction, planning and reader.

The Qwen control is read-only. Reuse verified GPT-5 seed42 image captions, but
rebuild every memory and run every text operation in a separate experiment.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluation import mcq_reader  # Project dependency bootstrap.
from scripts.evaluation import gpt5_text_memory_transport as api
from scripts.evaluation.simplemem_reflection_ablation import operation
from scripts.memory import text_memory_baselines as base
from scripts.memory import simplemem_official as official

ROOT = base.ROOT
CONTROL = ROOT / 'results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028/top5/simplemem_official_text_context5'
CAPTIONS = ROOT / 'results/memeye_gpt5_seed42_text_methods_20261025/caption_source'
OUTPUT = ROOT / 'results/simplemem_official_gpt5_memeye_k10_rrf5_reflection_20261029'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')


def read(path):
    return json.loads(Path(path).read_text())


def signature(protocol):
    return base.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if base.sha(Path(name)) != expected:
            raise ValueError('input changed: ' + name)


def verify_captions(source):
    manifest = read(source / 'completion_manifest.json')
    verify_hashes(manifest['artifacts_sha256'])
    protocol = read(source / 'protocol.json')
    verify_hashes(protocol['source_sha256'])
    verify_hashes(read(source / 'inputs/protocol.json')['source_sha256'])
    captions = base.read_rows(source / 'captions.jsonl')
    if not manifest['complete'] or manifest['caption_count'] != 437 or len(captions) != 437:
        raise ValueError('complete 437-image GPT-5 caption source required')
    if (protocol['model'], protocol['seed'], protocol['reasoning_effort']) != ('gpt-5', 42, 'minimal'):
        raise ValueError('caption generation policy differs')
    for row in captions:
        raw = read(source / 'responses' / (row['id'] + '.json'))
        if (row['model'] != manifest['returned_model'] or row['requested_model'] != 'gpt-5'
                or row['seed'] != 42 or not row['caption'].strip()
                or row['caption'] != raw['raw_response'].strip()
                or raw['finish_reason'] != 'stop' or raw.get('refusal')):
            raise ValueError('invalid caption or provenance: ' + row['id'])
    if len({r['id'] for r in captions}) != 437:
        raise ValueError('duplicate caption identities')
    return manifest, {r['id']: r['caption'] for r in captions}


def replace_captions(observations, questions, captions):
    """Change caption text only, using actual image bytes as the join key."""
    obs, qs = copy.deepcopy(observations), copy.deepcopy(questions)
    uses = []

    def caption(path, owner):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / 'data/MemEye/data/image' / path
        identity = base.sha(path)
        uses.append({'owner': owner, 'path': str(path.resolve()), 'image_sha256': identity})
        return captions[identity]

    for task, rows in obs.items():
        for row in rows:
            row['raw']['image_caption'] = [caption(p, task + '/' + row['id'])
                                            for p in row['raw']['input_image']]
    for q in qs:
        q['query_image_captions'] = [caption(p, q['id']) for p in q['query_image_paths']]
    return obs, qs, uses


def prepare(args):
    root = base.output_dir(args)
    source, caption_source = args.simplemem_source_run.resolve(), args.simplemem_caption_source.resolve()
    parent = read(source / 'protocol.json')
    if signature(parent) != parent['experiment_signature'] or read(source / 'summary.json')['status'] != 'complete':
        raise ValueError('control must be complete with a valid protocol')
    verify_hashes(parent['input_hashes'])
    if official.verify_installation() != parent['upstream']:
        raise ValueError('official SimpleMem installation changed')
    if (args.simplemem_semantic_top_k, args.simplemem_context_limit, args.simplemem_context_fusion,
            args.simplemem_reflection) != (10, 5, 'channel_rrf', True):
        raise ValueError('GPT-5 comparison requires semantic K10, context5, channel_rrf and reflection enabled')
    manifest, captions = verify_captions(caption_source)
    original_obs, original_qs = read(source / 'observations.json'), base.read_rows(source / 'questions.jsonl')
    if (base.hash_object(original_obs) != parent['observations_sha256']
            or base.hash_object(original_qs) != parent['prepared_questions_sha256']):
        raise ValueError('control prepared inputs changed')
    obs, qs, image_uses = replace_captions(original_obs, original_qs, captions)
    if args.task:
        qs = [q for q in qs if q['task'] == args.task]
    if args.limit_questions:
        qs = qs[:args.limit_questions]
    if not qs:
        raise ValueError('no selected questions')
    tasks = sorted({q['task'] for q in qs})
    obs = {t: obs[t] for t in tasks}
    p = copy.deepcopy(parent)
    for key in ('ablation', 'recovery', 'reader_device_map', 'retrieval_generation_batch_size',
                'retrieval_max_padded_batch_tokens'):
        p.pop(key, None)
    p.update(study='Official SimpleMem K10/channel-RRF context5, complete GPT-5 backbone',
        model='gpt-5', provider='OpenAI official', endpoint='https://api.openai.com/v1/chat/completions',
        expected_returned_model=manifest['returned_model'], caption_model=manifest['returned_model'],
        caption_model_path=None, reader_model={'repo': 'gpt-5', 'revision': manifest['returned_model']},
        reader_model_path=None, dtype=None, do_sample=None,
        tasks=tasks, questions=len(qs), memory_rounds=sum(map(len, obs.values())),
        task_workers=min(8, args.memory_workers), generation_batch_size=args.api_workers,
        # The API has no shared padded KV allocation. Let all API workers run;
        # preserve the per-request token cap and 128k context-budget audit.
        max_padded_batch_tokens=max(parent['max_padded_batch_tokens'],
                                    args.api_workers * parent['memory_max_new_tokens']),
        api_workers=args.api_workers, reasoning_effort='minimal', temperature_wire_policy='omitted',
        memory_sampling='native requested temperatures recorded by broker; GPT-5 wire omits temperature; seed42; minimal reasoning',
        answer_prompt_policy='unchanged common MCQ messages and options; GPT-5 completion cap at least 2048 including reasoning',
        query_policy='question plus GPT-5 query-image captions; options and gold excluded from retrieval',
        query_image_policy='verified GPT-5 seed42 captions, no native images in text operations',
        completion_budget_policy='max(2048,native requested cap); native logical caps remain recorded',
        prior_memory_reused=False, prior_planning_reused=False, prior_answers_reused=False,
        caption_reuse={'root': str(caption_source), 'count': 437, 'model': manifest['returned_model'],
                      'seed': 42, 'completion_manifest_sha256': base.sha(caption_source / 'completion_manifest.json')},
        control={'root': str(source), 'experiment_signature': parent['experiment_signature'],
                 'responses_sha256': base.sha(source / 'qa_responses.jsonl'),
                 'changed_roles': ['history_caption', 'query_image_caption', 'memory_construction',
                                   'retrieval_planning', 'retrieval_reflection', 'answer_reader'],
                 'fixed': ['raw_dialogues', 'images', 'questions', 'options', 'reader_prompt', 'MiniLM',
                           'window40_overlap2', 'semantic10_keyword5_structured5', 'reflection_max2', 'channel_rrf_context5']},
        observations_sha256=base.hash_object(obs), prepared_questions_sha256=base.hash_object(qs))
    p['differences_from_paper'] = [
        'Official pinned text core; GPT-5 for captions and all text LLM roles',
        'MiniLM-L6-v2 CPU embeddings; unchanged from Qwen control',
        'Common MCQ reader with fixed option permutation; options/gold excluded from retrieval',
        'Semantic K10, keyword K5, structured K5; reflection max2; two-stage channel RRF then five complete entries',
        'Sequential task-local construction with window40/overlap2; FTS rebuilt after finalize',
        'GPT-5 requires temperature omitted; minimal reasoning, seed42, completion cap includes reasoning',
        'Reuse verified GPT-5 seed42 captions; rebuild memory and all text operations from scratch']
    files = [Path(__file__), ROOT / 'scripts/visual_memory_pipeline.py',
             source / 'protocol.json', source / 'questions.jsonl', source / 'observations.json',
             source / 'qa_responses.jsonl', caption_source / 'completion_manifest.json',
             caption_source / 'captions.jsonl', caption_source / 'protocol.json',
             Path(api.__file__), ROOT / 'scripts/evaluation/openai_gpt5_base_transport.py',
             ROOT / 'scripts/evaluation/openai_backbone_transport.py',
             ROOT / 'scripts/evaluation/openai_chat_transport.py',
             ROOT / 'scripts/evaluation/openai_gpt54_seeded_transport.py',
             ROOT / 'scripts/evaluation/simplemem_reflection_ablation.py']
    files += [Path(u['path']) for u in image_uses]
    p['input_hashes'].update({str(f.resolve()): base.sha(f) for f in files})
    p['experiment_signature'] = signature(p)
    if (root / 'protocol.json').exists() and read(root / 'protocol.json') != p:
        raise ValueError('existing protocol differs; use a new output root')
    root.mkdir(parents=True, exist_ok=True)
    base.write_json(root / 'protocol.json', p)
    base.write_json(root / 'observations.json', obs)
    base.write_rows(root / 'questions.jsonl', qs)
    base.write_rows(root / 'caption_image_uses.jsonl', image_uses)
    snapshots = {}
    for name, digest in p['input_hashes'].items():
        if Path(name).suffix == '.py':
            target = root / 'source_snapshot' / (digest[:12] + '_' + Path(name).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(name, target)
            snapshots[name] = {'snapshot': str(target), 'sha256': digest}
    base.write_json(root / 'source_snapshot/manifest.json', snapshots)
    base.write_json(root / 'run_status.json', {'status': 'prepared', 'questions': len(qs),
        'memory_rounds': p['memory_rounds'], 'caption_model': p['caption_model']})
    print(f'GPT-5 prepared: {len(qs)} questions / {p["memory_rounds"]} rounds / verified GPT-5 captions', flush=True)


class GPT5Model(api.APIModel):
    def one(self, messages, cap, temperature):
        if self.fatal_errors:
            raise RuntimeError('GPT-5 batch stopped after a terminal API failure; inspect api_attempts.jsonl')
        return super().one(messages, cap, temperature)


class GPT5OfficialMemory(official.OfficialSimpleMemMemory):
    def __init__(self, protocol, workdir):
        super().__init__(protocol, workdir)
        # Initialization uses the pinned adapter; no generation happens before
        # this point. The broker routes all actual calls to the GPT-5 transport.
        self.core.llm_client.model = 'gpt-5'
        config = read(workdir / 'resolved_config.json')
        config.update(llm_model='gpt-5', actual_model=protocol['expected_returned_model'],
                      transport='official OpenAI via audited broker', reasoning_effort='minimal')
        base.write_json(workdir / 'resolved_config.json', config)


def run(args):
    root = base.output_dir(args)
    p = read(root / 'protocol.json')
    if signature(p) != p['experiment_signature']:
        raise ValueError('protocol signature differs')
    verify_hashes(p['input_hashes'])
    status = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'model': 'gpt-5', 'questions': p['questions']}
    base.write_json(root / 'run_status.json', status)
    try:
        with patch.object(base, 'LocalMemoryQwen', lambda protocol: GPT5Model(root, args.api_env_file, p['api_workers'])), \
                patch.object(official, 'OfficialSimpleMemMemory', GPT5OfficialMemory):
            base.run(args)
        analyze(args)
    except BaseException as exc:
        base.write_json(root / 'run_status.json', dict(status, status='stopped', error_type=type(exc).__name__,
            details='Inspect run_errors.json and api_attempts.jsonl; no model fallback.'))
        raise


def analyze(args):
    root = base.output_dir(args)
    base.analyze(args)
    p, summary = read(root / 'protocol.json'), read(root / 'summary.json')
    verify_hashes(p['input_hashes'])
    if signature(p) != p['experiment_signature']:
        raise ValueError('protocol signature differs')
    checks, failures = Counter(), []

    def check(ok, name, detail=''):
        checks[name] += 1
        if not ok:
            failures.append({'check': name, 'detail': detail})

    obs, qs = read(root / 'observations.json'), base.read_rows(root / 'questions.jsonl')
    control = Path(p['control']['root'])
    _, captions = verify_captions(Path(p['caption_reuse']['root']))
    expected_obs, expected_qs, _ = replace_captions(read(control / 'observations.json'),
                                                   base.read_rows(control / 'questions.jsonl'), captions)
    check(obs == {t: expected_obs[t] for t in p['tasks']}, 'only_history_captions_changed')
    check(qs == [q for q in expected_qs if q['id'] in {x['id'] for x in qs}], 'only_query_captions_changed')
    check(base.hash_object(qs) == p['prepared_questions_sha256'], 'question_hash')
    check(base.hash_object(obs) == p['observations_sha256'], 'observation_hash')
    api_rows = base.read_rows(root / 'api_responses.jsonl') if (root / 'api_responses.jsonl').exists() else []
    wire_outputs = {r['output']['response_id']: r for r in api_rows}
    calls, operations, usage = {}, Counter(), Counter()
    for path in (root / 'calls').glob('*.json'):
        call = read(path)
        req, out = call['request'], call['output']
        calls[path.stem] = call
        check(base.hash_object(req) == path.stem, 'call_hash', path.name)
        check(req['experiment_signature'] == p['experiment_signature'], 'call_signature', path.name)
        check('inherited_from' not in call, 'no_inherited_qwen_call', path.name)
        wire = api.build_request(req['messages'], req['max_new_tokens'], req['temperature'])
        saved = wire_outputs.get(out['response_id'])
        check(saved is not None and saved['request'] == wire and saved['output'] == out,
              'actual_gpt5_wire_payload_and_response', path.name)
        check(out['returned_model'] == p['expected_returned_model'] and out['requested_model'] == 'gpt-5',
              'actual_model_identity', path.name)
        check(out.get('refusal') is None, 'no_refusal', path.name)
        check(out['input_tokens'] + wire['max_completion_tokens'] <= p['context_budget'], 'context_budget', path.name)
        operations[operation(req)] += 1
        usage['input_tokens'] += out['input_tokens']
        usage['output_tokens'] += out['generated_tokens']
        usage['reasoning_tokens'] += out.get('reasoning_tokens') or 0
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    formatter = object.__new__(AnswerGenerator)
    selected_ids, task_entries = set(), {}
    rows = base.read_rows(root / 'qa_responses.jsonl')
    for task in p['tasks']:
        path = root / task
        if not (path / 'final_memory.json').exists():
            continue
        entries = {e['entry_id']: e for e in read(path / 'final_memory.json')}
        task_entries[task] = entries
        windows = read(path / 'runtime/construction_windows.json')
        check(all(w['parsed_successfully'] for w in windows), 'construction_parse_success', task)
        check({i for w in windows for i in w['entry_ids']} == set(entries), 'constructed_entries', task)
        check({i for w in windows for i in w['source_round_ids']} == {r['id'] for r in obs[task]},
              'all_source_rounds_constructed', task)
        cfg = read(path / 'runtime/resolved_config.json')
        check(cfg['llm_model'] == 'gpt-5' and cfg['settings'] == p['method_defaults'], 'resolved_task_config', task)
    for row in rows:
        detail = row['retrieval_detail']
        ids = detail['context_memory_ids']
        check(len(ids) == len(set(ids)) == 5 or (not ids and detail['source_empty_preserved']), 'context_limit', row['id'])
        selected_ids.update(ids)
        entries = task_entries[row['task']]
        check(all(e['id'] in entries and {k: v for k, v in e.items() if k != 'id'} == entries[e['id']]
                  for e in detail['items']), 'selected_entries_from_new_memory', row['id'])
        expected_context = formatter._format_contexts([MemoryEntry(**entries[i]) for i in ids])
        check(row['retrieved_context'] == expected_context, 'whole_final_context', row['id'])
        check(row['retrieval_query'] == base.retrieval_query(row), 'answer_free_query', row['id'])
        answer_call = calls[row['call_key']]
        check(answer_call['request']['messages'] == base.reader_messages(row, expected_context, p),
              'actual_reader_payload', row['id'])
        for batch in detail['search_batches']:
            check(batch['top_k'] == (10 if batch['channel'] == 'semantic' else 5), 'channel_limits', row['id'])
        base.write_json(root / 'retrieval_checkpoints' / (base.hash_object(row['id']) + '.json'), row)
    from scripts.analysis.audit_simplemem_fusion_ranks import audit as audit_ranks
    ranks = audit_ranks(root)
    base.write_json(root / 'independent_rank_audit.json', ranks)
    check(ranks['audit_passed'], 'independent_rrf_recalculation')
    check(not summary['memory_token_limit_hits'], 'no_truncated_memory_generations')
    audit = {'passed': not failures, 'complete': summary['status'] == 'complete',
        'checks': dict(checks), 'failures': failures, 'operations': dict(operations), 'text_api_usage': dict(usage),
        'memory_entries': sum(map(len, task_entries.values())), 'unique_selected_entries': len(selected_ids),
        'actual_model': p['expected_returned_model'], 'caption_model': p['caption_model'],
        'caption_count': p['caption_reuse']['count'], 'experiment_signature': p['experiment_signature']}
    base.write_json(root / 'backbone_audit.json', audit)
    control_rows = {r['id']: r for r in base.read_rows(control / 'qa_responses.jsonl')}
    paired = [{'id': r['id'], 'task': r['task'], 'qwen_correct': control_rows[r['id']]['predicted_choice'] == r['gold_choice'],
               'gpt5_correct': r['predicted_choice'] == r['gold_choice']} for r in rows]
    base.write_rows(root / 'paired_vs_qwen.jsonl', paired)
    base.write_json(root / 'comparison.json', {'questions': len(rows), 'complete': summary['status'] == 'complete',
        'qwen_correct': sum(x['qwen_correct'] for x in paired), 'gpt5_correct': summary['correct'],
        'gpt5_accuracy': summary['accuracy'],
        'wrong_to_correct': sum(not x['qwen_correct'] and x['gpt5_correct'] for x in paired),
        'correct_to_wrong': sum(x['qwen_correct'] and not x['gpt5_correct'] for x in paired),
        'scope': 'full generative backbone including GPT-5 captions; same embeddings and retrieval settings'})
    base.write_json(root / 'run_status.json', {'status': summary['status'] if not failures else 'audit_failed',
        'questions': len(rows), 'expected_questions': p['questions'], 'correct': summary['correct'],
        'accuracy': summary['accuracy'], 'audit_passed': not failures})
    if failures:
        raise RuntimeError('GPT-5 backbone audit failed; inspect backbone_audit.json')
    print(json.dumps({'status': summary['status'], 'questions': len(rows), 'accuracy': summary['accuracy'],
                      'audit_passed': True, 'text_api_usage': dict(usage)}), flush=True)


def pipeline(args):
    if args.mode != 'simplemem' or args.simplemem_backend != 'official_text' or args.dataset != 'memeye':
        raise ValueError('GPT-5 memory backbone requires official SimpleMem on MemEye')
    if args.api_model != 'gpt-5' or args.api_workers < 1:
        raise ValueError('use literal --api-model gpt-5 and positive API workers')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('run', 'all'):
        run(args)
    elif args.stage == 'analyze':
        analyze(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'analyze', 'all'], default='all')
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--source-run', type=Path, default=CONTROL)
    parser.add_argument('--caption-source', type=Path, default=CAPTIONS)
    parser.add_argument('--api-env-file', type=Path, default=ENV)
    parser.add_argument('--api-workers', type=int, default=8)
    parser.add_argument('--task')
    parser.add_argument('--limit-questions', type=int, default=0)
    args = parser.parse_args()
    args = SimpleNamespace(**vars(args), mode='simplemem', dataset='memeye', simplemem_backend='official_text',
        simplemem_source_run=args.source_run, simplemem_caption_source=args.caption_source,
        simplemem_context_limit=5, simplemem_semantic_top_k=10, simplemem_context_fusion='channel_rrf',
        simplemem_reflection=True, simplemem_preserve_empty=True, memory_workers=8, api_model='gpt-5')
    pipeline(args)
