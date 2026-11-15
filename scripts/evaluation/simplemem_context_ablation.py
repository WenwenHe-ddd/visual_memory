"""Answer again with first N distinct items from frozen, audited SimpleMem retrieval.

This changes final context count only. It does not rebuild memories, change the
query strategy's adaptive K, or pretend that the configured retrieve_k was the
actual vector-search K. No source answer calls are reused.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys

from scripts.evaluation import mcq_reader
from scripts.memory import text_memory_baselines as baseline
from scripts.memory.reflexion_memory import ROOT, LocalQwen, hash_object, sha, read_rows, write_json, write_rows
from scripts.memory.text_memory_transport import Broker


def formatter():
    sys.path.insert(0, str(baseline.VENDOR))
    from omni_memory.retrieval.pyramid_retriever import PyramidRetriever, RetrievalResult, RetrievalLevel
    def render(detail, limit=None, allow_empty=False):
        data = {key: detail[key] for key in ('query', 'level', 'items', 'total_candidates',
                'tokens_used_estimate', 'can_expand', 'expansion_candidates')}
        data['level'] = RetrievalLevel(data['level'])
        result = RetrievalResult(**data)
        metadata = detail
        if limit is not None:
            result, metadata = baseline.bounded_simplemem_result(result, limit, allow_empty=allow_empty)
        return PyramidRetriever.format_for_llm(None, result, include_instructions=False), metadata
    return render


def prepare(source, target, limit, preserve_empty=False):
    if limit < 1 or target.resolve() == source.resolve():
        raise ValueError('use a positive context limit and a distinct output directory')
    if (target / 'protocol.json').exists():
        raise ValueError('output already prepared; use --stage run or a new output directory')
    baseline.verify_vendor()
    old = json.loads((source / 'protocol.json').read_text())
    if old['mode'] != 'simplemem':
        raise ValueError('expected a SimpleMem source experiment')
    questions = read_rows(source / 'questions.jsonl')
    original = read_rows(source / 'qa_responses.jsonl')
    by_id = {r['id']: r for r in original}
    if len(by_id) != len(original) or set(by_id) != {q['id'] for q in questions}:
        raise ValueError('source answers must cover all questions exactly once')
    if hash_object(questions) != old['prepared_questions_sha256']:
        raise ValueError('source question manifest changed')
    render = formatter()
    rows, inputs, counts = [], [], []
    for q in questions:
        row = by_id[q['id']]
        if any(row[key] != value for key, value in q.items()):
            raise ValueError('source question/answer mismatch: ' + q['id'])
        messages = baseline.reader_messages(q, row['retrieved_context'], old)
        call_path = source / 'calls' / (row['call_key'] + '.json')
        call = json.loads(call_path.read_text())
        if (row['experiment_signature'] != old['experiment_signature'] or
                hash_object(messages) != row['reader_messages_sha256'] or
                call['request']['messages'] != messages or
                hash_object(call['request']) != row['call_key'] or
                call['request']['experiment_signature'] != old['experiment_signature'] or
                call['request']['kind'] != 'answer' or
                call['output']['raw_response'] != row['raw_response']):
            raise ValueError('source payload provenance mismatch: ' + q['id'])
        if row['retrieval_query'] != baseline.retrieval_query(q):
            raise ValueError('source query mismatch')
        full, _ = render(row['retrieval_detail'])
        if full != row['retrieved_context']:
            raise ValueError('upstream formatting no longer reproduces source context: ' + q['id'])
        context, detail = render(row['retrieval_detail'], limit, allow_empty=preserve_empty)
        counts.append(len(row['retrieval_detail']['items']))
        rows.append({**q, 'retrieval_query': row['retrieval_query'],
                     'retrieved_context': context, 'retrieval_detail': detail})
        inputs.append(call_path)
    inputs += [source / n for n in ('protocol.json', 'questions.jsonl', 'observations.json',
                                   'qa_responses.jsonl', 'summary.json', 'audit.json')]
    inputs += [Path(__file__), Path(baseline.__file__), ROOT / 'scripts/memory/reflexion_memory.py',
               ROOT / 'scripts/memory/text_memory_transport.py', baseline.VENDOR / 'UPSTREAM.json']
    protocol = deepcopy(old)
    protocol.pop('experiment_signature')
    protocol.update(study='SimpleMem frozen-retrieval final-context-count ablation',
        context_memory_limit=limit, preserve_empty_context=preserve_empty, input_hashes={str(p.resolve()): sha(p) for p in inputs},
        retrieval_count_policy='retain original adaptive retrieval and BM25 ordering; first N distinct items in final answer context',
        frozen_retrieval_source=str(source.resolve()), source_experiment_signature=old['experiment_signature'],
        retrieval_manifest_sha256=hash_object(rows),
        original_context_item_counts=dict(Counter(counts)),
        inference_device_policy='single GPU; same Qwen weights, bfloat16, SDPA, greedy reader and prompt',
        context_ablation_only=True)
    protocol['differences_from_paper'].append(f'Frozen original retrieval, answer context capped to {limit} distinct MAUs')
    protocol['experiment_signature'] = hash_object(protocol)
    write_json(target / 'protocol.json', protocol)
    write_rows(target / 'questions.jsonl', questions)
    write_rows(target / 'retrieval_manifest.jsonl', rows)
    write_json(target / 'preparation_audit.json', {'status': 'passed', 'questions': len(rows),
        'source_reader_contexts_exactly_reconstructed': True, 'context_limit': limit,
        'source_context_item_counts': dict(Counter(counts)), 'source_answer_calls_copied': 0,
        'preserved_empty_context_ids': [r['id'] for r in rows if r['retrieval_detail']['source_empty_preserved']]})
    print(f'Prepared {len(rows)} questions; original item counts {dict(Counter(counts))}; final count {limit}', flush=True)


def verify(target):
    protocol = json.loads((target / 'protocol.json').read_text())
    baseline.verify_vendor()
    for name, digest in protocol['input_hashes'].items():
        if sha(Path(name)) != digest:
            raise ValueError('input changed: ' + name)
    rows = read_rows(target / 'retrieval_manifest.jsonl')
    if hash_object(rows) != protocol['retrieval_manifest_sha256']:
        raise ValueError('retrieval manifest changed')
    return protocol, rows


class SingleGpuReader(baseline.LocalMemoryQwen):
    """Use the standard text-memory generation method on the available GPU."""
    def __init__(self, protocol):
        import torch
        import transformers
        self.torch, self.protocol = torch, protocol
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        torch.manual_seed(protocol['seed'])
        torch.set_num_threads(8)
        self.processor = transformers.AutoProcessor.from_pretrained(protocol['reader_model_path'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            protocol['reader_model_path'], dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True).to('cuda:0').eval()
        self.environment = {'torch': str(torch.__version__), 'transformers': transformers.__version__,
                            'gpu': torch.cuda.get_device_name(0), 'device': 'cuda:0',
                            'dtype': 'bfloat16', 'attention': 'sdpa'}


def run(target):
    protocol, rows = verify(target)
    model = SingleGpuReader(protocol)
    write_json(target / 'generation_environment.json', model.environment)
    broker = Broker(model, target, protocol, protocol['generation_batch_size'])
    try:
        def task_run(task):
            (target / task).mkdir(parents=True, exist_ok=True)
            completed = []
            for row in (r for r in rows if r['task'] == task):
                messages = baseline.reader_messages(row, row['retrieved_context'], protocol)
                output = broker.request(messages, protocol['max_new_tokens'], kind='answer', owner=row['id'])
                raw = output['raw_response'].strip()
                predicted = raw if raw in ('A', 'B', 'C', 'D') else None
                completed.append({**row, **output, 'predicted_choice': predicted,
                    'correct': predicted == row['gold_choice'],
                    'reader_messages_sha256': hash_object(messages),
                    'experiment_signature': protocol['experiment_signature']})
                write_rows(target / task / 'responses.jsonl', completed)
                write_json(target / task / 'progress.json', {'done': len(completed)})
            print(f'{task}: {len(completed)} answers', flush=True)
            return completed
        with ThreadPoolExecutor(max_workers=protocol['task_workers']) as pool:
            chunks = list(pool.map(task_run, protocol['tasks']))
        by_id = {r['id']: r for chunk in chunks for r in chunk}
        write_rows(target / 'qa_responses.jsonl', [by_id[r['id']] for r in rows])
    finally:
        broker.close()


def analyze(target):
    protocol, manifests = verify(target)
    rows = read_rows(target / 'qa_responses.jsonl')
    if [r['id'] for r in rows] != [r['id'] for r in manifests]:
        raise ValueError('incomplete or reordered answers')
    source = Path(protocol['frozen_retrieval_source'])
    before = {r['id']: r for r in read_rows(source / 'qa_responses.jsonl')}
    limit = protocol['context_memory_limit']
    for row, expected in zip(rows, manifests):
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError('answer inputs changed')
        ids = [r['id'] for r in row['retrieval_detail']['items']]
        empty_ok = (protocol['preserve_empty_context'] and not ids and
                    not before[row['id']]['retrieval_detail']['items'])
        if len(ids) != len(set(ids)) or (len(ids) != limit and not empty_ok):
            raise ValueError('context count mismatch')
        call = json.loads((target / 'calls' / (row['call_key'] + '.json')).read_text())
        assert hash_object(call['request']) == row['call_key']
        assert call['request']['experiment_signature'] == row['experiment_signature'] == protocol['experiment_signature']
        assert call['request']['messages'] == baseline.reader_messages(row, row['retrieved_context'], protocol)
        assert call['output']['raw_response'] == row['raw_response']
        assert row['correct'] == (row['predicted_choice'] == row['gold_choice'])
    n = len(rows)
    summary = {'mode': 'simplemem', 'variant': f'simplemem_context{limit}', 'status': 'complete',
        'questions': n, 'correct': sum(r['correct'] for r in rows),
        'accuracy': sum(r['correct'] for r in rows) / n, 'context_memory_limit': limit,
        'preserved_empty_contexts': sum(not r['retrieval_detail']['items'] for r in rows),
        'format_errors': sum(r['predicted_choice'] is None for r in rows),
        'answer_token_limit_hits': sum(r['hit_token_limit'] for r in rows),
        'experiment_signature': protocol['experiment_signature']}
    old_correct = sum(before[r['id']]['predicted_choice'] == r['gold_choice'] for r in rows)
    comparison = {'source_correct': old_correct, 'source_accuracy': old_correct/n,
        'strict_context_correct': summary['correct'], 'strict_context_accuracy': summary['accuracy'],
        'questions': n, 'context_limit': limit,
        'accuracy_change_percentage_points': 100*(summary['correct']-old_correct)/n,
        'wrong_to_correct': sum(r['correct'] and not before[r['id']]['correct'] for r in rows),
        'correct_to_wrong': sum(not r['correct'] and before[r['id']]['correct'] for r in rows),
        'source_description': 'upstream adaptive K=5/10/20 with BM25 union; configured retrieve_k=20 was overridden',
        'source_response_sha256': sha(source / 'qa_responses.jsonl'), 'response_sha256': sha(target / 'qa_responses.jsonl')}
    events = read_rows(target / 'call_events.jsonl')
    assert all(e['kind'] == 'answer' for e in events)
    write_json(target / 'summary.json', summary)
    write_json(target / 'context_limit_comparison.json', comparison)
    write_json(target / 'audit.json', {'status': 'passed', 'verified_answer_payloads': n,
        'nonempty_contexts_contain_exactly_n_distinct_items': limit,
        'preserved_empty_contexts': summary['preserved_empty_contexts'],
        'frozen_retrieval_only_no_memory_rebuild': True,
        'answer_requests': len(events), 'fresh_answer_requests': sum(not e['reused'] for e in events)})
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/visual_memory_pipeline/simplemem')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--context-limit', type=int, default=5)
    parser.add_argument('--preserve-empty', action='store_true')
    parser.add_argument('--stage', choices=('prepare', 'run', 'analyze', 'all'), default='all')
    args = parser.parse_args()
    target = args.output or ROOT / f'results/visual_memory_pipeline/top{args.context_limit}/simplemem_context{args.context_limit}'
    if args.stage in ('prepare', 'all'):
        prepare(args.source, target, args.context_limit, args.preserve_empty)
    if args.stage in ('run', 'all'):
        run(target)
    if args.stage in ('analyze', 'all'):
        analyze(target)


if __name__ == '__main__':
    main()
