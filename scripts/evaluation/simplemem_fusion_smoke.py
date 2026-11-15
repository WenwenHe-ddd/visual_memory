"""Run one real Qwen retrieval/answer on a frozen official SimpleMem database."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from scripts.evaluation import mcq_reader
from scripts.evaluation.simplemem_reflection_ablation import operation
from scripts.memory import simplemem_official as official
from scripts.memory import text_memory_baselines as baseline


def read(path):
    return json.loads(path.read_text())


def run(source, root, question_id):
    parent = read(source / 'protocol.json')
    if official.verify_installation() != parent['upstream']:
        raise ValueError('official runtime changed')
    if baseline.hash_object({k: v for k, v in parent.items() if k != 'experiment_signature'}) != parent['experiment_signature']:
        raise ValueError('source protocol signature differs')
    if not read(source / 'audit.json')['audit_passed_so_far']:
        raise ValueError('source run was not audited')
    question = next(q for q in baseline.read_rows(source / 'questions.jsonl') if q['id'] == question_id)
    task = question['task']
    protocol = copy.deepcopy(parent)
    protocol.pop('recovery', None)
    protocol.update(tasks=[task], questions=1, task_workers=1, generation_batch_size=1,
        context_fusion='channel_rrf', fusion_rank_constant=60, context_memory_limit=5,
        reader_device_map={'': 0}, study='Single-question real-Qwen validation of K10 three-channel RRF')
    protocol['method_defaults']['SEMANTIC_TOP_K'] = 10
    protocol['method_defaults']['ENABLE_REFLECTION'] = True
    protocol['prepared_questions_sha256'] = baseline.hash_object([question])
    observations = {task: read(source / 'observations.json')[task]}
    protocol['memory_rounds'] = len(observations[task])
    protocol['observations_sha256'] = baseline.hash_object(observations)
    protocol['differences_from_paper'] = [s.replace('semantic top-k=5', 'semantic top-k=10')
        .replace('reflection=False', 'reflection=True') for s in protocol['differences_from_paper']]
    protocol['differences_from_paper'].append('Adapter applies two-stage equal-channel RRF before the final five-entry cap')
    protocol['retrieval_count_policy'] = 'semantic K10 per query, keyword K5, structured K5; reflection enabled; '
    protocol['retrieval_count_policy'] += 'two-stage equal-channel RRF then five complete entries'
    protocol['smoke_source'] = {'root': str(source.resolve()), 'protocol_sha256': baseline.sha(source / 'protocol.json'),
        'responses_sha256': baseline.sha(source / 'qa_responses.jsonl'),
        'memory_sha256': baseline.sha(source / task / 'final_memory.json'),
        'scope': 'One diagnostic question, not a benchmark accuracy estimate; frozen construction and primary planning calls.'}
    changed = []
    allowed = {Path(official.__file__).resolve(), Path(baseline.__file__).resolve()}
    for name, digest in protocol['input_hashes'].items():
        current = baseline.sha(Path(name))
        if current != digest:
            if Path(name) not in allowed:
                raise ValueError('unexpected source input change: ' + name)
            changed.append({'path': name, 'old_sha256': digest, 'new_sha256': current})
        protocol['input_hashes'][name] = current
    for path in (Path(__file__), Path(official.__file__).with_name('simplemem_fusion.py')):
        protocol['input_hashes'][str(path.resolve())] = baseline.sha(path)
    protocol['smoke_source']['source_changes'] = changed
    protocol['experiment_signature'] = baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})
    root.mkdir(parents=True, exist_ok=False)
    baseline.write_json(root / 'protocol.json', protocol)
    baseline.write_rows(root / 'questions.jsonl', [question])
    baseline.write_json(root / 'observations.json', observations)
    target = root / task
    target.mkdir()
    shutil.copytree(source / task / 'runtime', target / 'runtime')
    for name in ('final_memory.json', 'memory_stats.json'):
        shutil.copyfile(source / task / name, target / name)
    assets = {str(p.relative_to(target)): baseline.sha(p) for p in (target / 'runtime/lancedb').rglob('*') if p.is_file()}
    baseline.write_json(root / 'frozen_database_hashes.json', assets)
    for path in (source / 'calls').glob('*.json'):
        call = read(path)
        if operation(call['request']) not in ('information_plan', 'query_plan', 'query_analysis'):
            continue
        if baseline.hash_object(call['request']) != path.stem:
            raise ValueError('cached source request hash differs')
        call['request']['experiment_signature'] = protocol['experiment_signature']
        call['inherited_from'] = {'path': str(path.resolve()), 'sha256': baseline.sha(path)}
        baseline.write_json(root / 'calls' / (baseline.hash_object(call['request']) + '.json'), call)
    model = baseline.LocalMemoryQwen(protocol)
    baseline.write_json(root / 'generation_environment.json', model.environment)
    broker = baseline.Broker(model, root, protocol, batch_size=1)
    try:
        baseline.install_runtime(broker, protocol)
        memory = official.OfficialSimpleMemMemory(protocol, target / 'runtime')
        actual = {e.entry_id: e.model_dump() for e in memory.core.get_all_memories()}
        if actual != {e['entry_id']: e for e in read(target / 'final_memory.json')}:
            raise ValueError('frozen memory contents differ')
        memory.finalized = True
        context, detail = memory.recall(baseline.retrieval_query(question))
        messages = baseline.reader_messages(question, context, protocol)
        answer = broker.request(messages, protocol['max_new_tokens'], kind='answer', owner=question_id)
    finally:
        broker.close()
    result = {**question, **answer, 'predicted_choice': answer['raw_response'].strip(),
        'retrieval_query': baseline.retrieval_query(question), 'retrieved_context': context,
        'retrieval_detail': detail, 'reader_messages_sha256': baseline.hash_object(messages),
        'experiment_signature': protocol['experiment_signature']}
    baseline.write_rows(root / 'qa_responses.jsonl', [result])
    assert len(detail['context_memory_ids']) == len(set(detail['context_memory_ids'])) == 5
    assert detail['context_memory_ids'] == [r['id'] for r in detail['fusion']['ranked_candidates'][:5]]
    assert all(b['top_k'] == (10 if b['channel'] == 'semantic' else 5) for b in detail['search_batches'])
    assert all(baseline.sha(target / name) == digest for name, digest in assets.items())
    assert all(baseline.sha(Path(name)) == digest for name, digest in protocol['input_hashes'].items())
    call = read(root / 'calls' / (result['call_key'] + '.json'))
    assert 'inherited_from' not in call and call['request']['messages'] == messages
    assert result['predicted_choice'] in ('A', 'B', 'C', 'D') and not result['hit_token_limit']
    selected = detail['fusion']['ranked_candidates'][:5]
    report = {'status': 'complete', 'scope': 'single-question integration smoke test; not overall accuracy',
        'question_id': question_id, 'reflection_enabled': True, 'semantic_top_k': 10,
        'keyword_top_k': 5, 'structured_top_k': 5, 'context_count': 5,
        'candidate_count': len(detail['fusion']['ranked_candidates']),
        'selected_channel_hits': {ch: sum(ch in r['channels'] for r in selected)
                                  for ch in ('semantic', 'keyword', 'structured')},
        'same_as_old_ordered_first_five': detail['context_memory_ids'] == detail['original_item_ids'][:5],
        'new_entries_outside_first_semantic_five': len(set(detail['context_memory_ids']) - set(detail['search_batches'][0]['result_ids'][:5])),
        'reflection_searches': sum(b['phase'] == 'reflection' for b in detail['search_batches']),
        'prediction': result['predicted_choice'], 'gold': question['gold_choice'], 'audit_passed': True}
    baseline.write_json(root / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--question-id', default='Brand_Memory_Test/Q9')
    args = parser.parse_args()
    run(args.source, args.root, args.question_id)
