"""Resume a validated official-SimpleMem database without rebuilding memories.

Fork results into a new protocol, retain original request/output provenance,
and generate only answers that are missing. Source results remain untouched.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.evaluation import mcq_reader  # Project dependency bootstrap.
from scripts.memory import simplemem_official as official
from scripts.memory import text_memory_baselines as baseline


def read(path):
    return json.loads(path.read_text())


def fork(source, target):
    if target.exists():
        raise ValueError('recovery target already exists')
    parent = read(source / 'protocol.json')
    if baseline.hash_object({k: v for k, v in parent.items() if k != 'experiment_signature'}) != parent['experiment_signature']:
        raise ValueError('source protocol signature differs')
    changed = []
    adapter = Path(official.__file__).resolve()
    for name, digest in parent['input_hashes'].items():
        actual = baseline.sha(Path(name))
        if actual != digest:
            if Path(name) != adapter:
                raise ValueError('unexpected source change: ' + name)
            changed.append({'path': name, 'old_sha256': digest, 'new_sha256': actual})
    if official.verify_installation() != parent['upstream']:
        raise ValueError('upstream installation changed')
    protocol = copy.deepcopy(parent)
    inherited_rows = sum(len(baseline.read_rows(source / task / 'responses.jsonl'))
                         for task in parent['tasks'] if (source / task / 'responses.jsonl').exists())
    protocol['recovery'] = {'source_root': str(source.resolve()),
        'source_signature': parent['experiment_signature'],
        'source_protocol_sha256': baseline.sha(source / 'protocol.json'),
        'inherited_answers': inherited_rows, 'source_changes': changed,
        'policy': 'preserve all built memories, retrieved contexts and completed answers; '
                  'repair non-object query analyses with at most three schema-only LLM retries; '
                  'generate only missing answers; provenance retained for every inherited call'}
    protocol['differences_from_paper'].append(
        'Non-object query-analysis outputs receive bounded format-only retries before retrieval continues')
    for name in protocol['input_hashes']:
        protocol['input_hashes'][name] = baseline.sha(Path(name))
    protocol['input_hashes'][str(Path(__file__).resolve())] = baseline.sha(Path(__file__))
    audit_source = Path(__file__).resolve().parents[1] / 'analysis/audit_simplemem_official_run.py'
    protocol['input_hashes'][str(audit_source)] = baseline.sha(audit_source)
    protocol['experiment_signature'] = baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})
    target.mkdir(parents=True)
    baseline.write_json(target / 'protocol.json', protocol)
    for name in ('questions.jsonl', 'observations.json'):
        shutil.copyfile(source / name, target / name)
    snapshots = {}
    for name, digest in protocol['input_hashes'].items():
        if Path(name).suffix == '.py':
            snapshot = target / 'source_snapshot' / (digest[:12] + '_' + Path(name).name)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, snapshot)
            snapshots[name] = {'snapshot': str(snapshot.resolve()), 'sha256': digest}
    baseline.write_json(target / 'source_snapshot/manifest.json', snapshots)
    mapping = {}
    for path in sorted((source / 'calls').glob('*.json')):
        call = read(path)
        if baseline.hash_object(call['request']) != path.stem or call['request']['experiment_signature'] != parent['experiment_signature']:
            raise ValueError('source call hash or signature differs: ' + str(path))
        call['request']['experiment_signature'] = protocol['experiment_signature']
        key = baseline.hash_object(call['request'])
        mapping[path.stem] = key
        call['inherited_from'] = {'path': str(path.resolve()), 'sha256': baseline.sha(path)}
        baseline.write_json(target / 'calls' / (key + '.json'), call)
    events = baseline.read_rows(source / 'call_events.jsonl')
    baseline.write_rows(target / 'call_events.jsonl', [dict(e, key=mapping[e['key']], reused=True, inherited=True) for e in events])
    provenance = {}
    for task in parent['tasks']:
        shutil.copytree(source / task, target / task)
        response_path = target / task / 'responses.jsonl'
        if response_path.exists():
            rows = baseline.read_rows(response_path)
            for row in rows:
                row['experiment_signature'] = protocol['experiment_signature']
                row['call_key'] = mapping[row['call_key']]
            baseline.write_rows(response_path, rows)
            provenance[task] = {'source_responses_sha256': baseline.sha(source / task / 'responses.jsonl'),
                                'inherited_ids': [r['id'] for r in rows],
                                'source_memory_sha256': baseline.sha(source / task / 'final_memory.json')}
        completed = target / task / 'complete.json'
        if completed.exists():
            checkpoint = read(source / task / 'complete.json')
            if checkpoint['responses_sha256'] != baseline.sha(source / task / 'responses.jsonl'):
                raise ValueError('source completion checkpoint differs: ' + task)
            baseline.write_json(completed, dict(checkpoint, experiment_signature=protocol['experiment_signature'],
                                                responses_sha256=baseline.sha(response_path)))
    baseline.write_json(target / 'recovery_provenance.json', provenance)
    shutil.copyfile(source / 'generation_environment.json', target / 'inherited_generation_environment.json')
    print(f'Recovery prepared: {inherited_rows} answers and all task memories preserved', flush=True)


def run(root):
    protocol = read(root / 'protocol.json')
    for name, digest in protocol['input_hashes'].items():
        if baseline.sha(Path(name)) != digest:
            raise ValueError('recovery input changed: ' + name)
    if official.verify_installation() != protocol['upstream']:
        raise ValueError('upstream installation changed')
    questions = baseline.read_rows(root / 'questions.jsonl')
    observations = read(root / 'observations.json')
    if baseline.hash_object(questions) != protocol['prepared_questions_sha256'] or baseline.hash_object(observations) != protocol['observations_sha256']:
        raise ValueError('prepared inputs changed')
    model = baseline.LocalMemoryQwen(protocol)
    baseline.write_json(root / 'generation_environment.json', model.environment)
    broker = baseline.Broker(model, root, protocol, protocol['generation_batch_size'])
    baseline.install_runtime(broker, protocol)
    try:
        for task in protocol['tasks']:
            target = root / task
            if (target / 'complete.json').exists():
                continue
            broker.local.owner = task
            memory = official.OfficialSimpleMemMemory(protocol, target / 'runtime')
            memory.stats = read(target / 'memory_stats.json')
            expected_entries = {e['entry_id']: e for e in read(target / 'final_memory.json')}
            actual_entries = {e.entry_id: e.model_dump() for e in memory.core.get_all_memories()}
            if actual_entries != expected_entries or memory.stats['stored_rounds'] != len(observations[task]):
                raise ValueError('restored memory differs: ' + task)
            memory.finalized = True
            rows = baseline.read_rows(target / 'responses.jsonl')
            answered = {r['id'] for r in rows}
            task_questions = [q for q in questions if q['task'] == task]
            for q in task_questions:
                if q['id'] in answered:
                    continue
                query = baseline.retrieval_query(q)
                context, detail = memory.recall(query)
                messages = baseline.reader_messages(q, context, protocol)
                output = broker.request(messages, protocol['max_new_tokens'], kind='answer', owner=q['id'])
                raw = output['raw_response'].strip()
                rows.append({**q, **output, 'predicted_choice': raw if raw in 'ABCD' and len(raw) == 1 else None,
                    'retrieval_query': query, 'retrieved_context': context, 'retrieval_detail': detail,
                    'reader_messages_sha256': baseline.hash_object(messages), 'experiment_signature': protocol['experiment_signature']})
                baseline.write_rows(target / 'responses.jsonl', rows)
                baseline.write_json(target / 'progress.json', {'phase': 'answer', 'done': len(rows), 'total': len(task_questions)})
                print(f'{task}: answered {len(rows)}/{len(task_questions)}', flush=True)
            baseline.write_json(target / 'complete.json', {'experiment_signature': protocol['experiment_signature'],
                'responses_sha256': baseline.sha(target / 'responses.jsonl'), 'questions': len(rows)})
    finally:
        broker.close()
    args = SimpleNamespace(mode='simplemem', simplemem_backend='official_text', simplemem_context_limit=5,
                           output_root=root.parent.parent)
    baseline.analyze(args)
    from scripts.analysis.audit_simplemem_official_run import audit
    report = audit(root)
    if report['status'] != 'complete' or not report['audit_passed_so_far']:
        raise RuntimeError('recovery result audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'run', 'all'), default='all')
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        if args.source is None:
            parser.error('--source is required for preparation')
        fork(args.source, args.target)
    if args.stage in ('run', 'all'):
        run(args.target)
