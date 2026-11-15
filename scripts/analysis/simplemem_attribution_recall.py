"""Review, freeze and score reconstructed SimpleMem source-round attribution.

Review/freeze are gold-blind. Score reads benchmark labels only after the
annotation file has been sealed. Ambiguous sources are never all credited.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from scripts.analysis.simplemem_source_attribution import normalize, parse_and_validate, verify_inputs
from scripts.memory.reflexion_memory import hash_object, read_rows, sha, write_json, write_rows


def read(path):
    return json.loads(path.read_text())


def duplicate_evidence_sources(annotation, job):
    """Find alternative round identities supporting all quoted text of an edge."""
    grouped = defaultdict(list)
    for e in annotation.get('evidence', []):
        grouped[e['source']].append(e['quote'])
    duplicates = {}
    for alias, quotes in grouped.items():
        alternatives = []
        for source in job['sources']:
            fields = [normalize(source.get(f, '')) for f in ('user', 'assistant', 'caption', 'memory')]
            if all(any(normalize(q) in field for field in fields) for q in quotes):
                alternatives.append(source['alias'])
        if len(alternatives) > 1:
            duplicates[alias] = alternatives
    return duplicates


def review(root):
    p = read(root / 'protocol.json')
    verify_inputs(p, root)
    jobs = {j['id']: j for j in read_rows(root / 'jobs.jsonl')}
    records = read_rows(root / 'annotations.jsonl')
    queue = []
    for record in records:
        job = jobs[record['job_id']]
        duplicates = [duplicate_evidence_sources(record[key] or {}, job) for key in ('pass1', 'pass2')]
        reasons = []
        if record['status'] != 'two_pass_agreement':
            reasons.append('pass_disagreement_or_quote_validation')
        if any(duplicates):
            reasons.append('quoted_content_has_multiple_possible_sources')
        if any(any(e['source'].startswith('P') for e in (record[k] or {}).get('evidence', [])) for k in ('pass1', 'pass2')):
            reasons.append('uses_previous_window_memory')
        if reasons:
            queue.append({**record, 'review_reasons': reasons, 'duplicate_evidence_sources': duplicates,
                          'sources': job['sources']})
    write_rows(root / 'review_queue.jsonl', queue)
    report = {'entries': len(records), 'review_entries': len(queue), 'selected_review_entries': sum(r['selected'] for r in queue),
        'reasons': dict(Counter(x for r in queue for x in r['review_reasons'])),
        'source_annotations_sha256': sha(root / 'annotations.jsonl'), 'review_queue_sha256': sha(root / 'review_queue.jsonl')}
    write_json(root / 'review_status.json', report)
    print(json.dumps(report), flush=True)


def validate_manual(item, target, job):
    if item['decision'] not in ('accept', 'partial', 'unresolved', 'unmatched'):
        raise ValueError('invalid manual decision')
    status = {'accept': 'resolved', 'partial': 'partial', 'unresolved': 'ambiguous', 'unmatched': 'unmatched'}[item['decision']]
    annotation = {'id': target['alias'], 'status': status, 'evidence': item.get('evidence', []),
                  'ambiguous_sources': item.get('ambiguous_sources', []), 'note': item['note']}
    mini_job = dict(job, targets=[target])
    _, issues = parse_and_validate(json.dumps([annotation]), mini_job)
    if issues:
        raise ValueError('manual evidence validation failed: ' + item['entry_id'] + ': ' + str(issues))
    if item['decision'] in ('unresolved', 'unmatched') and annotation['evidence']:
        raise ValueError('unresolved/unmatched overrides must not assert definite evidence')
    if not item.get('reviewer') or not item['note'].strip():
        raise ValueError('manual review requires reviewer identity and a reason')
    return annotation


def freeze(root):
    p = read(root / 'protocol.json')
    verify_inputs(p, root)
    if (root / 'annotation_lock.json').exists():
        raise ValueError('annotations already sealed; do not change labels after scoring')
    jobs = {j['id']: j for j in read_rows(root / 'jobs.jsonl')}
    records = read_rows(root / 'annotations.jsonl')
    queue = {r['entry_id']: r for r in read_rows(root / 'review_queue.jsonl')}
    manual_path = root / 'manual_review.jsonl'
    manual_rows = read_rows(manual_path) if manual_path.exists() else []
    manual = {r['entry_id']: r for r in manual_rows}
    if len(manual) != len(manual_rows):
        raise ValueError('duplicate manual reviews')
    if set(queue) - set(manual):
        raise ValueError(f'{len(set(queue) - set(manual))} flagged entries still need explicit review')
    if set(manual) - {r['entry_id'] for r in records}:
        raise ValueError('review refers to unknown memory')
    annotations = {}
    checks = Counter()
    for record in records:
        identity = record['entry_id']
        job = jobs[record['job_id']]
        target = next(t for t in job['targets'] if t['entry_id'] == identity)
        # Revalidate every raw model request, output, and literal source quote.
        for index in (1, 2):
            result = read(root / f'pass{index}' / (job['id'] + '.json'))
            call = read(root / 'calls' / (result['call_key'] + '.json'))
            if hash_object(call['request']) != result['call_key'] or call['output'] != result['output']:
                raise ValueError('model call changed')
            payload = json.loads(call['request']['messages'][1]['content'])
            if set(payload) != {'original_sources', 'target_memories'}:
                raise ValueError('unexpected annotation payload')
            if any(set(t) != {'id', 'statement'} for t in payload['target_memories']):
                raise ValueError('benchmark metadata entered target prompt')
            checks['gold_blind_request'] += 1
        if identity in manual:
            accepted = validate_manual(manual[identity], target, job)
            decision, method = manual[identity]['decision'], 'codex_text_review'
        else:
            if record['status'] != 'two_pass_agreement':
                raise ValueError('unreviewed disagreement')
            proposals = []
            for index in (1, 2):
                result = read(root / f'pass{index}' / (job['id'] + '.json'))
                values, issues = parse_and_validate(result['output']['raw_response'], job)
                if target['alias'] in issues or values[target['alias']] != record[f'pass{index}']:
                    raise ValueError('unvalidated automatic evidence')
                proposals.append(values[target['alias']])
            if {e['source'] for e in proposals[0]['evidence']} != {e['source'] for e in proposals[1]['evidence']}:
                raise ValueError('automatic source sets disagree')
            accepted, decision, method = proposals[0], 'accept', 'two_pass_model_agreement'
        annotations[identity] = {**record, 'accepted_annotation': accepted, 'decision': decision,
                                 'annotation_method': method}
        checks['entry_annotation_validated'] += 1
    resolved, active = {}, set()

    def resolve(identity):
        if identity in resolved:
            return resolved[identity]
        if identity in active:
            raise ValueError('cyclic previous-window provenance')
        active.add(identity)
        record = annotations[identity]
        job = jobs[record['job_id']]
        sources = {s['alias']: s for s in job['sources']}
        ids, possible_ids, edges, uncertainty = set(), set(), [], []
        if record['decision'] != 'accept':
            uncertainty.append('annotation_' + record['decision'])
        for evidence in record['accepted_annotation']['evidence']:
            origin = sources[evidence['source']]
            if origin['kind'] == 'round':
                ids.add(origin['round_id'])
                possible_ids.add(origin['round_id'])
                edges.append({**evidence, 'source_round_id': origin['round_id']})
            else:
                parent = resolve(origin['entry_id'])
                possible_ids.update(parent['possible_source_round_ids'])
                # A quote may refer to only part of a multi-source prior fact.
                # Do not silently assign all of that fact's parent rounds.
                if len(parent['source_round_ids']) == 1 and not parent['uncertainty']:
                    ids.update(parent['source_round_ids'])
                    edges.append({**evidence, 'via_entry_id': origin['entry_id'],
                                  'source_round_id': parent['source_round_ids'][0]})
                else:
                    uncertainty.append('previous_entry_not_uniquely_resolved:' + origin['entry_id'])
        possible_aliases = record['accepted_annotation']['ambiguous_sources']
        if uncertainty and not possible_aliases and not edges:
            # An unmatched statement has no narrow candidate set. This broad
            # envelope is for sensitivity reporting only, never scored as hits.
            possible_aliases = list(sources)
        for alias in possible_aliases:
            origin = sources[alias]
            if origin['kind'] == 'round':
                possible_ids.add(origin['round_id'])
            else:
                possible_ids.update(resolve(origin['entry_id'])['possible_source_round_ids'])
        possible_ids.update(ids)
        result = {'entry_id': identity, 'task': record['task'], 'selected': record['selected'],
            'memory': record['memory'], 'source_round_ids': sorted(ids), 'evidence': edges,
            'possible_source_round_ids': sorted(possible_ids),
            'uncertainty': uncertainty, 'annotation_method': record['annotation_method'],
            'decision': record['decision'], 'note': record['accepted_annotation']['note']}
        resolved[identity] = result
        active.remove(identity)
        return result

    for identity in annotations:
        resolve(identity)
    output = [resolved[r['entry_id']] for r in records]
    write_rows(root / 'frozen_source_annotations.jsonl', output)
    files = [root / 'protocol.json', root / 'jobs.jsonl', root / 'annotations.jsonl',
             root / 'review_queue.jsonl', root / 'frozen_source_annotations.jsonl', Path(__file__)]
    if manual_path.exists():
        files.append(manual_path)
    for folder in ('pass1', 'pass2', 'calls'):
        files.extend(sorted((root / folder).glob('*.json')))
    hashes = {str(f.resolve()): sha(f) for f in files}
    report = {'status': 'sealed_before_scoring', 'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
        'entries': len(output), 'selected_entries': sum(r['selected'] for r in output),
        'manual_review_entries': len(manual), 'annotation_methods': dict(Counter(r['annotation_method'] for r in output)),
        'selected_unresolved_or_partial_entries': sum(r['selected'] and bool(r['uncertainty']) for r in output),
        'selected_entries_without_confirmed_rounds': sum(r['selected'] and not r['source_round_ids'] for r in output),
        'checks': dict(checks), 'input_hashes': hashes,
        'interpretation': 'Reconstructed model-assisted annotations with exact supporting quotations and '
                          'explicit Codex text adjudication of flagged cases. Not native provenance or human gold labels.'}
    write_json(root / 'annotation_lock.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'input_hashes'}), flush=True)


def score(root):
    p = read(root / 'protocol.json')
    verify_inputs(p, root)
    lock = read(root / 'annotation_lock.json')
    for name, digest in lock['input_hashes'].items():
        if sha(Path(name)) != digest:
            raise ValueError('sealed annotation input changed: ' + name)
    source = Path(p['source_root'])
    annotations = {r['entry_id']: r for r in read_rows(root / 'frozen_source_annotations.jsonl')}
    # Benchmark questions and gold labels enter only here, after label sealing.
    rows = read_rows(source / 'qa_responses.jsonl')
    canonical_path = Path(__file__).resolve().parents[2] / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
    canonical = {q['id']: q for q in read_rows(canonical_path)}
    if len(rows) != 371 or {r['id'] for r in rows} != set(canonical):
        raise ValueError('question coverage differs')
    evaluated = []
    for row in rows:
        q = canonical[row['id']]
        if any(row[k] != q[k] for k in ('task', 'question', 'options', 'gold_choice', 'gold_ids')):
            raise ValueError('benchmark question changed')
        mids = row['retrieval_detail']['context_memory_ids']
        if not (len(mids) == len(set(mids)) == 5):
            raise ValueError('context count differs')
        selected = [annotations[i] for i in mids]
        if any(a['task'] != row['task'] or not a['selected'] for a in selected):
            raise ValueError('selected memory annotation mismatch')
        recalled = set().union(*(set(a['source_round_ids']) for a in selected))
        possible = set().union(*(set(a['possible_source_round_ids']) for a in selected))
        gold = set(q['gold_ids'])
        hit = recalled & gold
        evaluated.append({'id': row['id'], 'task': row['task'], 'context_memory_ids': mids,
            'attributed_source_round_ids': sorted(recalled), 'gold_ids': sorted(gold), 'hit_ids': sorted(hit),
            'gold_count': len(gold), 'hit_count': len(hit), 'recall': len(hit) / len(gold),
            'possible_hit_ids': sorted(possible & gold), 'possible_hit_count': len(possible & gold),
            'any_hit': bool(hit), 'all_hit': hit == gold,
            'uncertain_entry_ids': [a['entry_id'] for a in selected if a['uncertainty']],
            'qa_correct': row['predicted_choice'] == row['gold_choice']})

    def metric(group):
        total = sum(x['gold_count'] for x in group)
        hits = sum(x['hit_count'] for x in group)
        return {'questions': len(group), 'gold_round_references': total, 'gold_round_hits': hits,
                'recall_at5_micro': hits / total, 'recall_at5_macro': mean(x['recall'] for x in group),
                'ambiguous_mapping_sensitivity_micro_ceiling': sum(x['possible_hit_count'] for x in group) / total,
                'questions_any_hit': sum(x['any_hit'] for x in group),
                'questions_all_hit': sum(x['all_hit'] for x in group),
                'questions_with_uncertain_entries': sum(bool(x['uncertain_entry_ids']) for x in group),
                'qa_correct': sum(x['qa_correct'] for x in group)}

    report = {'status': 'complete', **metric(evaluated), 'context_limit': 5,
        'metric_unit': 'gold source dialogue round; deduplicated within each question',
        'estimate_type': 'conservative, post-hoc reconstructed source attribution',
        'uncertainty_policy': 'Only explicitly accepted source edges count. Ambiguous alternative sources '
                              'and unresolved previous-memory dependencies are not credited.',
        'limitations': 'These source links are model-assisted reconstructions with quotation checks and '
                      'Codex text review, not original source IDs or independent human gold annotations. '
                      'Five compressed facts can represent fewer or more than five original dialogue rounds. '
                      'Source-round recall does not establish that the answer-relevant fact survived compression. '
                      'The sensitivity ceiling gives credit to every ambiguous candidate and is not a measured '
                      'recall or statistical confidence bound.',
        'annotations': {k: lock[k] for k in ('entries', 'selected_entries', 'manual_review_entries',
                      'annotation_methods', 'selected_unresolved_or_partial_entries', 'selected_entries_without_confirmed_rounds')},
        'by_task': {t: metric([r for r in evaluated if r['task'] == t]) for t in sorted({r['task'] for r in evaluated})},
        'source_experiment_signature': p['source_experiment_signature'],
        'annotation_lock_sha256': sha(root / 'annotation_lock.json'), 'canonical_sha256': sha(canonical_path),
        'responses_sha256': sha(source / 'qa_responses.jsonl')}
    write_rows(root / 'recall_per_question.jsonl', evaluated)
    with (root / 'recall_per_question.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(evaluated[0]))
        writer.writeheader()
        for row in evaluated:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    write_json(root / 'recall_summary.json', report)
    with (root / 'source_annotations.csv').open('w', newline='') as f:
        fields = ['entry_id', 'task', 'selected', 'memory', 'source_round_ids', 'possible_source_round_ids',
                  'annotation_method', 'decision', 'uncertainty', 'evidence', 'note']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in annotations.values():
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list)
                             else row[k] for k in fields})
    print(json.dumps({k: v for k, v in report.items() if k != 'by_task'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('review', 'freeze', 'score'), required=True)
    args = parser.parse_args()
    {'review': review, 'freeze': freeze, 'score': score}[args.stage](args.root.resolve())
