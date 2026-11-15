"""Freshly score frozen, audited reflection contexts with the control reader batch size."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.evaluation import mcq_reader  # Project dependency bootstrap.
from scripts.memory import text_memory_baselines as baseline


def read(path):
    return json.loads(path.read_text())


def run(source, root):
    reflection = read(source / 'protocol.json')
    reflection_audit = read(source / 'reflection_audit.json')
    if not reflection_audit['audit_passed'] or reflection_audit['questions'] != 371:
        raise ValueError('reflection run must first finish and pass its audit')
    control_root = Path(reflection['ablation']['control_root'])
    control_protocol = read(control_root / 'protocol.json')
    control = {r['id']: r for r in baseline.read_rows(control_root / 'qa_responses.jsonl')}
    rows = baseline.read_rows(source / 'qa_responses.jsonl')
    if len(rows) != 371 or len({r['id'] for r in rows}) != 371 or {r['id'] for r in rows} != set(control):
        raise ValueError('question coverage differs')
    # This diagnostic measures repeated answer generation only. The actual
    # reflection retrieval, entries, and context are frozen in the source run.
    protocol = copy.deepcopy(reflection)
    protocol['generation_batch_size'] = control_protocol['generation_batch_size']
    protocol['max_padded_batch_tokens'] = control_protocol['max_padded_batch_tokens']
    protocol['reader_check'] = {
        'source_root': str(source.resolve()),
        'source_protocol_sha256': baseline.sha(source / 'protocol.json'),
        'source_responses_sha256': baseline.sha(source / 'qa_responses.jsonl'),
        'source_audit_sha256': baseline.sha(source / 'reflection_audit.json'),
        'scope': 'fresh final answers only; reuse actual audited reflection retrieval and context',
        'batch_policy': 'consecutive source rows, control maximum batch size and padded-token budget; no answer cache',
        'note': 'same maximum batch size as control; historical asynchronous batch composition is unavailable',
    }
    protocol['ablation']['execution_changes'] = {
        'reflection_retrieval_generation_batch_size': [control_protocol['generation_batch_size'], reflection['generation_batch_size']],
        'final_reader_generation_batch_size': [control_protocol['generation_batch_size'], protocol['generation_batch_size']],
        'final_reader_max_padded_batch_tokens': [control_protocol['max_padded_batch_tokens'], protocol['max_padded_batch_tokens']],
    }
    protocol['input_hashes'][str(Path(__file__).resolve())] = baseline.sha(Path(__file__))
    protocol['experiment_signature'] = baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})
    root.mkdir(parents=True, exist_ok=False)
    baseline.write_json(root / 'protocol.json', protocol)
    model = baseline.LocalMemoryQwen(protocol)
    baseline.write_json(root / 'generation_environment.json', model.environment)
    results, batches = [], []
    for start in range(0, len(rows), protocol['generation_batch_size']):
        group = rows[start:start + protocol['generation_batch_size']]
        messages = [baseline.reader_messages(r, r['retrieved_context'], protocol) for r in group]
        for row, message in zip(group, messages):
            source_call = read(source / 'calls' / (row['call_key'] + '.json'))
            if source_call['request']['messages'] != message or baseline.hash_object(message) != row['reader_messages_sha256']:
                raise ValueError('reader input changed: ' + row['id'])
        lengths = [len(model.processor.tokenizer.encode(model.processor.apply_chat_template(m, tokenize=False,
                    add_generation_prompt=True), add_special_tokens=False)) for m in messages]
        cap = protocol['max_new_tokens']
        if (max(lengths) + cap) * len(group) > protocol['max_padded_batch_tokens']:
            raise ValueError('reader batch exceeds control padded token budget')
        outputs = model.generate(messages, cap, sampling_temperature=0.0)
        if len(outputs) != len(group):
            raise ValueError('incomplete generation batch')
        batches.append({'ids': [r['id'] for r in group], 'input_tokens': lengths})
        for row, message, output in zip(group, messages, outputs):
            prediction = output['raw_response'].strip()
            if prediction not in ('A', 'B', 'C', 'D') or output['hit_token_limit']:
                raise ValueError('invalid or truncated reader answer: ' + row['id'])
            request = {'messages': message, 'max_new_tokens': cap, 'kind': 'answer', 'do_sample': False,
                       'temperature': 0.0, 'seed': protocol['seed'], 'response_format': None,
                       'experiment_signature': protocol['experiment_signature']}
            key = baseline.hash_object(request)
            baseline.write_json(root / 'calls' / (key + '.json'), {'request': request, 'output': output})
            results.append({**row, **output, 'call_key': key, 'predicted_choice': prediction,
                'experiment_signature': protocol['experiment_signature'],
                'retrieval_source': {'root': str(source.resolve()), 'call_key': row['call_key'],
                    'response_hash': baseline.hash_object(row)}})
        baseline.write_rows(root / 'qa_responses.jsonl', results)
        baseline.write_json(root / 'progress.json', {'status': 'running', 'answered': len(results),
            'expected': 371, 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
        print(f'Reader batch=2: {len(results)}/371', flush=True)
    baseline.write_json(root / 'batches.json', batches)
    comparisons = []
    raw_reflection = {r['id']: r for r in rows}
    by_task = {}
    for row in results:
        before = control[row['id']]
        raw = raw_reflection[row['id']]
        comparisons.append({'id': row['id'], 'task': row['task'], 'gold': row['gold_choice'],
            'control_prediction': before['predicted_choice'], 'reflection_batch8_prediction': raw['predicted_choice'],
            'reflection_batch2_prediction': row['predicted_choice'],
            'same_context_as_control': row['retrieved_context'] == before['retrieved_context'],
            'same_reader_messages_as_control': row['reader_messages_sha256'] == before['reader_messages_sha256']})
        counts = by_task.setdefault(row['task'], {'questions': 0, 'correct': 0})
        counts['questions'] += 1
        counts['correct'] += int(row['predicted_choice'] == row['gold_choice'])
    for counts in by_task.values():
        counts['accuracy'] = counts['correct'] / counts['questions']
    for name, digest in protocol['input_hashes'].items():
        if baseline.sha(Path(name)) != digest:
            raise ValueError('source or input changed: ' + name)
    if baseline.sha(source / 'qa_responses.jsonl') != protocol['reader_check']['source_responses_sha256']:
        raise ValueError('reflection source changed')
    correct = sum(r['predicted_choice'] == r['gold_choice'] for r in results)
    report = {'status': 'complete', 'questions': len(results), 'correct': correct, 'accuracy': correct / len(results),
        'control_correct': sum(r['predicted_choice'] == r['gold_choice'] for r in control.values()),
        'reflection_batch8_correct': reflection_audit['correct'],
        'same_contexts_as_control': sum(x['same_context_as_control'] for x in comparisons),
        'same_reader_messages_as_control': sum(x['same_reader_messages_as_control'] for x in comparisons),
        'changed_predictions_from_control': sum(x['control_prediction'] != x['reflection_batch2_prediction'] for x in comparisons),
        'changed_predictions_from_batch8': sum(x['reflection_batch8_prediction'] != x['reflection_batch2_prediction'] for x in comparisons),
        'wrong_to_right_vs_control': sum(x['control_prediction'] != x['gold'] and x['reflection_batch2_prediction'] == x['gold'] for x in comparisons),
        'right_to_wrong_vs_control': sum(x['control_prediction'] == x['gold'] and x['reflection_batch2_prediction'] != x['gold'] for x in comparisons),
        'context_counts': dict(Counter(len(r['retrieval_detail']['context_memory_ids']) for r in results)),
        'fresh_answer_calls': len(list((root / 'calls').glob('*.json'))), 'by_task': by_task,
        'experiment_signature': protocol['experiment_signature'],
        'interpretation': 'Identical reader inputs cannot demonstrate a retrieval-reflection effect. '
            'Differences from historical answers are repeat-generation variation. '
            'Maximum batch size is matched, but asynchronous historical batch composition is unavailable.'}
    if report['fresh_answer_calls'] != 371 or report['same_reader_messages_as_control'] != 371:
        raise ValueError('fresh answer count or expected input equivalence differs')
    baseline.write_rows(root / 'paired_comparison.jsonl', comparisons)
    baseline.write_json(root / 'summary.json', report)
    baseline.write_json(root / 'progress.json', {'status': 'complete', 'answered': 371, 'expected': 371})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.root)
