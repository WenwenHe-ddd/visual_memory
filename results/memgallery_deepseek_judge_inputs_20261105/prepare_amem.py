"""Freeze the already completed 678/1711 DeepSeek A-Mem run for LLM judging.

No answer selection, retrieval, generation, or API access occurs here.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.mem_gallery_master_reader import score_answer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keyed(records, key='id'):
    result = {r[key]: r for r in records}
    assert len(result) == len(records)
    return result


def main():
    base = ROOT / 'results/mem_gallery_amem_backbones_20261104'
    source_roots = [base / n for n in (
        'deepseek', 'deepseek_retry3_20261104', 'deepseek_astronomy_purpose_20261105')]
    final = source_roots[-1]
    canonical = ROOT / 'results/mem_gallery_master_answer_20261021/inputs'
    questions = keyed(rows(canonical / 'questions.jsonl'))
    gold = keyed(rows(canonical / 'gold.jsonl'))
    assert len(questions) == len(gold) == 1711 and questions.keys() == gold.keys()
    hashes = {str(canonical / n): sha(canonical / n) for n in ['questions.jsonl', 'gold.jsonl']}
    verified_artifacts = {}
    response_map, scored_map, payload_map, response_sources = {}, {}, {}, {}
    context_counts = Counter()
    source_counts = {}
    for source in source_roots:
        audit = read(source / 'partial_audit.json')
        assert audit['status'] == 'passed'
        for filename, digest in audit['artifact_sha256'].items():
            assert sha(filename) == digest, filename
            verified_artifacts[filename] = digest
        protocol = read(source / 'protocol.json')
        assert protocol['expected_model'] == protocol['requested_model'] == 'deepseek-v4.1-flash'
        assert protocol['context_memory_limit'] == protocol['method_defaults']['retrieve_k'] == 5
        qmap = keyed(rows(source / 'inputs/questions.jsonl'))
        gmap = keyed(rows(source / 'inputs/gold.jsonl'))
        assert qmap.keys() == questions.keys() and gmap.keys() == gold.keys()
        for qid, q in questions.items():
            assert all(qmap[qid][k] == q[k] for k in ['task', 'point', 'question'])
            assert gmap[qid]['answer'] == gold[qid]['answer']
        responses = keyed(rows(source / 'partial_reader/responses.jsonl'))
        scored = keyed(rows(source / 'partial_scored_questions.jsonl'))
        payloads = keyed(rows(source / 'partial_reader/manifest.jsonl'), 'record_id')
        assert responses.keys() == scored.keys() == payloads.keys()
        assert not response_map.keys() & responses.keys(), 'Overlapping answers would require explicit provenance'
        source_counts[str(source)] = len(responses)
        response_map.update(responses)
        scored_map.update(scored)
        payload_map.update(payloads)
        response_sources.update({qid: str(source / 'partial_reader/responses.jsonl') for qid in responses})
        for task in {questions[qid]['task'] for qid in responses}:
            retrieval_file = source / 'memory' / task / 'retrieval.jsonl'
            retrieval = keyed(rows(retrieval_file))
            for qid, r in retrieval.items():
                assert qid in responses
                ids = r['retrieval_detail']['context_memory_ids']
                assert len(ids) == len(set(ids)) == 5
                context_counts[len(ids)] += 1
            hashes[str(retrieval_file)] = sha(retrieval_file)
        for leaf in ['inputs/questions.jsonl', 'inputs/gold.jsonl', 'protocol.json',
                     'partial_reader/responses.jsonl', 'partial_reader/manifest.jsonl',
                     'partial_scored_questions.jsonl', 'partial_results.json', 'partial_audit.json']:
            path = source / leaf
            hashes[str(path)] = sha(path)
    assert list(source_counts.values()) == [1429, 228, 54]
    assert response_map.keys() == scored_map.keys() == questions.keys()
    assert context_counts == {5: 1711}
    prior = source_roots[1]
    prior_combined = keyed(rows(prior / 'combined_scored_questions.jsonl'))
    assert len(prior_combined) == 1657
    for qid, scored in prior_combined.items():
        assert scored == scored_map[qid]
    combined = keyed(rows(final / 'combined_scored_questions.jsonl'))
    assert combined == scored_map
    final_audit = read(final / 'combined_audit.json')
    assert final_audit['status'] == 'passed' and final_audit['full_dataset']
    assert final_audit['result_sha256'] == sha(final / 'combined_results.json')
    assert final_audit['scored_sha256'] == sha(final / 'combined_scored_questions.jsonl')
    assert final_audit['parent_audit_sha256'] == sha(prior / 'combined_audit.json')
    assert final_audit['subset_audit_sha256'] == sha(final / 'partial_audit.json')
    assert final_audit['purpose_variant_audit_sha256'] == sha(final / 'purpose_variant_audit.json')
    summary = read(final / 'combined_results.json')
    amendment = read(final / 'purpose_variant_amendment.json')
    assert summary['model'] == 'deepseek-v4.1-flash' and summary['original_prompt_baseline'] is False
    assert summary['prior_results_reused'] == 1657 and summary['retry_questions'] == 54
    variant_audit = read(final / 'purpose_variant_audit.json')
    assert variant_audit['status'] == 'passed' and variant_audit['modified_calls'] == 1
    for filename, digest in amendment['artifact_sha256'].items():
        assert sha(filename) == digest, filename
        verified_artifacts[filename] = digest
    for source, leaves in [(prior, ['combined_results.json', 'combined_scored_questions.jsonl', 'combined_audit.json']),
                           (final, ['combined_results.json', 'combined_scored_questions.jsonl', 'combined_audit.json',
                                    'purpose_variant_amendment.json', 'purpose_variant_audit.json'])]:
        for leaf in leaves:
            path = source / leaf
            hashes[str(path)] = sha(path)
    frozen, metrics, models, finish = [], [], Counter(), Counter()
    for qid, q in questions.items():
        r = response_map[qid]
        assert r['requested_model'] == r['returned_model'] == 'deepseek-v4.1-flash'
        assert r['seed'] == 42 and r['temperature'] == 0 and r['enable_thinking'] is False
        assert r['finish_reason'] == 'stop' and not r.get('refusal')
        assert r['hit_token_limit'] is False and r['raw_response'].strip()
        measured = score_answer(r['raw_response'], gold[qid]['answer'])
        assert all(abs(measured[k] - scored_map[qid][k]) < 1e-12 for k in ['em', 'f1', 'bleu_1'])
        metrics.append(measured)
        models[r['returned_model']] += 1
        finish[r['finish_reason']] += 1
        frozen.append(dict(id=qid, task=q['task'], point=q['point'], question=q['question'],
            ground_truth=gold[qid]['answer'], model_output=r['raw_response'], em=int(measured['em']),
            source_record_id=qid, answer_model=r['returned_model'], source_returned_model=r['returned_model'],
            source_finish_reason=r['finish_reason'], source_response_id=r['response_id'],
            source_request_sha256=r['request_sha256'], source_hit_token_limit=r['hit_token_limit'],
            source_response_file=response_sources[qid]))
    correct = sum(int(m['em']) for m in metrics)
    means = {key: sum(m[key] for m in metrics) / 1711 for key in ['em', 'f1', 'bleu_1']}
    assert correct == summary['overall']['correct'] == 678
    assert all(abs(means[k] - summary['overall'][k]) < 1e-12 for k in means)
    records_path = HERE / 'a_mem.jsonl'
    records_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in frozen))
    validation = dict(question_and_gold_exact_canonical_match=True,
        responses_exact_match_scored_questions=True, all_em_f1_bleu1_recomputed_per_question=True,
        source_answer_model_counts=dict(models), all_original_answers_preserved=True,
        source_split_counts=source_counts, prior_1657_answers_unchanged=True,
        original_prompt_baseline=False, one_memory_evolution_call_has_authorized_purpose_statement=True)
    method = dict(id='a_mem', label='A-Mem (final 5)', display_label='A-Mem (final 5)',
        code_method='a_mem', answer_model='deepseek-v4.1-flash', expected_em_correct=678,
        records_path=str(records_path), records_sha256=sha(records_path), source_sha256=hashes,
        source_root=str(final), source_condition='a_mem', **means,
        context='A-Mem primary K5, final context5. Frozen 1429 + 228 + 54 composite; one Astronomy memory-evolution call has a documented authorized evaluation-purpose statement; reader prompts unchanged.',
        context_counts=dict(context_counts), source_token_limit_records=0, source_token_limit_known_records=1711,
        source_empty_answer_records=0, validation=validation,
        source_variant=dict(original_prompt_baseline=False, modified_memory_calls=1,
            amendment=str(final / 'purpose_variant_amendment.json'), statement=amendment['statement'],
            user_authorization=amendment['user_authorization']))
    manifest = dict(dataset='MemGallery', answer_model='deepseek-v4.1-flash', questions_per_method=1711,
        selection_policy='Freeze the complete 678/1711 existing A-Mem run matching the supplied screenshot; no per-question selection.', methods=[method])
    (HERE / 'amem_methods_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    audit = dict(status='passed', api_calls=0, questions=1711, correct=correct, **means,
        source_split_counts=source_counts, original_answers_preserved=True, canonical_ids_verified=True,
        context_counts=dict(context_counts), source_model_counts=dict(models), finish_reason_counts=dict(finish),
        original_prompt_baseline=False, one_memory_evolution_call_has_authorized_purpose_statement=True,
        verified_original_artifact_files=len(verified_artifacts), original_artifact_sha256=verified_artifacts,
        source_sha256=hashes, records_sha256=sha(records_path), validator_sha256=sha(Path(__file__)))
    (HERE / 'amem_input_validation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:audit[k] for k in ['status','api_calls','questions','correct','em','f1','bleu_1','source_split_counts','verified_original_artifact_files']}, indent=2))


if __name__ == '__main__':
    main()
