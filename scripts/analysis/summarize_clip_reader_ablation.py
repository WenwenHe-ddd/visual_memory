"""Audit and summarize completed CLIP image/dialogue reader-only experiments."""
import argparse
import csv
from pathlib import Path

from scripts.evaluation import clip_image_dialogue_api as api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=api.ROOT / 'reports/clip_image_dialogue_readers_20261101')
    args = parser.parse_args()
    questions = api.rows(api.DEFAULT_SOURCE / 'qa_manifest.jsonl')
    source_protocol = api.read(api.DEFAULT_SOURCE / 'protocol.json')
    qwen = {r['id']: r for r in api.rows(api.DEFAULT_SOURCE / 'qa_responses.jsonl')}
    assert len(qwen) == len(questions) == 371
    baseline = {q['id']: qwen[q['id']]['predicted_choice'] == q['gold_choice'] for q in questions}
    gold_hits = sum(len(set(q['gold_ids']) & set(q['top_memory_ids'])) for q in questions)
    gold_total = sum(len(set(q['gold_ids'])) for q in questions)
    comparison = [dict(model='Qwen3-VL-8B-Instruct', correct=sum(baseline.values()), questions=371,
        accuracy=sum(baseline.values()) / 371, top_k=5, micro_recall=gold_hits/gold_total,
        wrong_to_correct_vs_qwen=0, correct_to_wrong_vs_qwen=0,
        provider_rejections=0, format_errors=0, token_limit_hits=0,
        source=str(api.DEFAULT_SOURCE), returned_model='Qwen3-VL-8B-Instruct')]
    detail = {q['id']: dict(id=q['id'], task=q['task'], gold_choice=q['gold_choice'],
        qwen_predicted=qwen[q['id']]['predicted_choice'], qwen_correct=baseline[q['id']]) for q in questions}
    for model in api.MODELS:
        target = api.DEFAULT_OUTPUT / model
        p, qs, request_manifest, responses = api.verify(target)
        assert qs == questions and p['source_signature'] == source_protocol['experiment_signature']
        assert len(responses) == 371
        answers = {r['id']:r for r in responses}
        assert set(answers) == set(baseline)
        result = {}
        for q in qs:
            a = answers[q['id']]
            assert a['source_payload_sha256'] == q['payload_sha256']
            assert a['reader_image_count'] == len(q['reader_image_paths'])
            assert a['request_sha256'] == request_manifest[q['id']]['request_sha256']
            assert a['predicted_choice'] == api.parse_choice(a['raw_response'])
            result[q['id']] = a['status'] == 'ok' and a['predicted_choice'] == q['gold_choice']
            detail[q['id']].update({model + '_predicted': a['predicted_choice'],
                                  model + '_correct': result[q['id']], model + '_status': a['status']})
        summary = api.read(target / 'summary.json')
        assert sum(result.values()) == summary['correct']
        row = dict(model=model, correct=sum(result.values()), questions=371,
            accuracy=sum(result.values()) / 371, top_k=5, micro_recall=gold_hits/gold_total,
            wrong_to_correct_vs_qwen=sum(v and not baseline[k] for k,v in result.items()),
            correct_to_wrong_vs_qwen=sum(not v and baseline[k] for k,v in result.items()),
            provider_rejections=summary['provider_rejections'], format_errors=summary['format_errors'],
            token_limit_hits=summary['token_limit_hits'], source=str(target),
            returned_model=';'.join(summary['returned_models']))
        comparison.append(row)
        api.dump(target / 'independent_output_audit.json', dict(status='passed',
            questions=371, exact_source_questions=True, response_ids_unique_and_complete=True,
            payload_and_request_hashes_verified=True, image_counts_verified=True,
            answer_parsing_verified=True, correct=row['correct'],
            experiment_signature=p['experiment_signature']))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in (('comparison.csv', comparison), ('per_question.csv', list(detail.values()))):
        with (args.output / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    api.dump(args.output / 'comparison.json', dict(method='CLIP Top5 -> native images + source dialogue',
        retrieval_unchanged=True, gold_hits=gold_hits, gold_total=gold_total, rows=comparison,
        caveat='Reader-only comparison. GPT-5 uses minimal reasoning/default temperature; DeepSeek thinking=false/temperature=0; provider-native image preprocessing differs from local Qwen.'))
    print(comparison)


if __name__ == '__main__':
    main()
