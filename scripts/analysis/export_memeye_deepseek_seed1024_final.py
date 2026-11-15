"""Export an audited DeepSeek seed42/1024 workbook without running inference.

The default requires both final independent audits and a fresh SimpleMem
attribution seal. --allow-partial writes an explicitly named DRAFT workbook;
it never labels pending values as zero or marks the experiment complete.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scripts.analysis import audit_memeye_deepseek_seed1024 as audit
from scripts.evaluation import memeye_deepseek_seed1024_metrics as mapping

OUT = mapping.OUT
JUDGE42 = ROOT / 'reports/memeye_deepseek_open_llm_judge_20261106'
LABELS = {**{m: s[0] for m, s in mapping.OURS.items()}, **mapping.TEXT,
          'clip_image_dialogue': 'VisRAG (CLIP → original images + dialogue)', 'hippomm': 'HippoMM'}


def reference_judges():
    complete = audit.read(JUDGE42 / 'completion.json')
    audit.require(complete['status'] == 'complete' and complete['methods'] == 10, 'Seed42 judge reference incomplete')
    audit.hashes({str(JUDGE42 / name): value for name, value in complete['artifact_sha256'].items()})
    report = audit.read(JUDGE42 / 'audit.json')
    result = {r['id']: r for r in report['methods']}
    audit.require(set(result) == set(mapping.METHODS), 'Seed42 judge method coverage differs')
    for item in result.values():
        audit.require(item['questions'] == 371 and item['score_sum'] / 371 == item['llm_judge']
                      and sum(item['score_distribution'].values()) == 371, 'Seed42 judge aggregate differs')
    return result


def audit_reports(output, allow_partial=False):
    paths = [output / 'audit/final_audit_mcq_metrics.json', output / 'audit/final_audit_open_judge.json']
    reports = [audit.read(p) if p.exists() else {} for p in paths]
    for report in reports:
        audit.require(report.get('status') != 'failed', 'Independent audit reports an actual failure; resolve it before exporting')
    ready = all(r.get('passed') is True and r.get('status') == 'passed' for r in reports)
    ready = ready and reports[0].get('methods_complete') == reports[0].get('metrics_complete') == 10
    ready = ready and reports[1].get('methods_passed') == 10
    if not allow_partial:
        audit.require(ready, 'Require both complete10-method audits, all metrics, and fresh sealed SimpleMem attribution; use --allow-partial only for an explicitly named draft')
    if ready:
        audit.require(reports[0]['metric_index_unique_methods'] == 10, 'Metric index does not cover all10 methods')
        audit.require(reports[0]['metric_index_complete'] and reports[0]['metric_index_sha256'] ==
                      audit.sha(output / 'metrics/metric_summary.json'), 'Final metric index changed after independent audit')
        audit.require(reports[0]['source_plan_sha256'] == audit.sha(output / 'plan.json') and
                      reports[0]['implementation_sha256'] == audit.sha(audit.__file__), 'MCQ audit source or plan changed')
        audit.hashes(audit.read(output / 'plan.json')['source_sha256'])
        for item in reports[0]['methods']:
            audit.hashes(item['citations_sha256'])
        audit.hashes(reports[0]['artifact_sha256'])
        audit.hashes(reports[1]['artifact_sha256'])
        seal = output / 'metrics/simplemem_source_attribution/annotation_lock.json'
        audit.require(seal.exists(), 'Fresh SimpleMem annotation lock missing')
        audit.hashes(audit.read(seal)['input_hashes'])
    return reports, paths, ready


def comparison(output=OUT, allow_partial=False):
    output = Path(output).resolve()
    reports, report_paths, ready = audit_reports(output, allow_partial)
    reference = mapping.reference_mapping(); judges42 = reference_judges()
    mcq = {r['method']: r for r in reports[0].get('methods', [])}
    opened = {r['method']: r for r in reports[1].get('methods', [])}
    paired_path = output / 'metrics/paired_mcq_audit.json'
    independent_pairs = audit.read(paired_path).get('methods', {}) if paired_path.exists() else {}
    result = []
    for method in mapping.METHODS:
        old, new, op = reference[method], mcq.get(method, {}), opened.get(method, {})
        full_mcq = new.get('mcq_complete') is True
        metrics_ready = new.get('status') == 'passed'
        judge_ready = op.get('status') == 'passed'
        a42, a1024 = old['accuracy'], new.get('accuracy') if full_mcq else None
        mean = (a42 + a1024) / 2 if a1024 is not None else None
        sd = abs(a42 - a1024) / math.sqrt(2) if a1024 is not None else None
        pair = new.get('paired_outcomes', {})
        row = dict(method=method, label=LABELS[method], status='complete' if metrics_ready and judge_ready else
                   'pending_mcq' if not full_mcq else 'pending_metrics' if not metrics_ready else 'pending_open_or_judge',
            seed42_correct=old['correct'], seed1024_correct=new.get('correct') if full_mcq else None,
            questions=371, seed42_accuracy=a42, seed1024_accuracy=a1024,
            accuracy_delta_pp=(a1024 - a42) * 100 if a1024 is not None else None,
            two_seed_accuracy_mean=mean, two_seed_accuracy_sample_sd=sd,
            seed42_recall_at5_micro=old['recall_at5_micro'],
            seed1024_recall_at5_micro=new.get('recall_at5_micro') if metrics_ready else None,
            seed42_context_source_micro_recall=old.get('context_source_micro_recall'),
            seed1024_context_source_micro_recall=new.get('context_source_micro_recall') if metrics_ready else None,
            recall_is_literal_top5=method != 'hippomm',
            seed42_llm_judge=judges42[method]['llm_judge'],
            seed1024_llm_judge=op.get('judge', {}).get('llm_judge') if judge_ready else None,
            seed42_storage_bytes=old['storage_bytes'], seed42_storage_mib=old['storage_mib'],
            seed1024_storage_bytes=new.get('storage_bytes'), seed1024_storage_mib=new.get('storage_mib'),
            both_correct=pair.get('correct42_to_correct1024'),
            seed42_only_correct=pair.get('correct42_to_wrong1024'),
            seed1024_only_correct=pair.get('wrong42_to_correct1024'),
            both_wrong=pair.get('wrong42_to_wrong1024'),
            changed_predicted_answers=new.get('changed_predicted_answers_vs_seed42'),
            changed_native_predictions=new.get('changed_native_predictions_vs_seed42'),
            format_errors=new.get('format_errors'), answer_token_limit_hits=new.get('answer_token_limit_hits'),
            provider_input_rejections=new.get('provider_input_rejections'), invalid_choice_records=new.get('invalid_choice_records'),
            historical_api_error_events=new.get('historical_api_error_events'),
            memory_token_limit_hits=new.get('memory_token_limit_hits'),
            native_outcomes=new.get('native_outcomes'), empty_open_answers=op.get('empty_answers'),
            open_token_limit_hits=op.get('token_limit_hits'))
        independent = independent_pairs.get(method, {})
        if independent.get('status') == 'complete' and full_mcq:
            for field, paired_field in (('seed42_correct', 'seed42_correct'), ('seed1024_correct', 'seed1024_correct'),
                ('both_correct', 'both_correct'), ('seed42_only_correct', 'seed42_only_correct'),
                ('seed1024_only_correct', 'seed1024_only_correct'), ('both_wrong', 'both_wrong'), ('changed_predicted_answers', 'choice_changed')):
                audit.require(row[field] == independent[paired_field], 'Independent paired MCQ audit differs: ' + method + '/' + field)
            audit.require(row['changed_native_predictions'] == independent['native_prediction_changed'], 'Independent native MCQ prediction changes differ: ' + method)
        result.append(row)
    return result, reports, report_paths, ready


def export(output=OUT, destination=None, allow_partial=False):
    output = Path(output).resolve()
    result, reports, report_paths, ready = comparison(output, allow_partial)
    destination = Path(destination or output / ('MemEye_DeepSeek_seed42_vs_seed1024.xlsx' if ready else
                                               'MemEye_DeepSeek_seed42_vs_seed1024_DRAFT.xlsx')).resolve()
    audit.require(ready or 'DRAFT' in destination.stem.upper(), 'Partial workbook destination must contain DRAFT')
    workbook = Workbook(); main = workbook.active; main.title = 'Seed1024'
    main.append(['Method', 'Status', 'Accuracy', 'Recall@5 micro', 'LLM Judge', 'Storage MiB',
                 'Correct', 'Questions', 'Hippo context source coverage', 'Metric note'])
    paired = workbook.create_sheet('Seed42_vs_1024')
    paired.append(['Method', 'Accuracy42', 'Accuracy1024', 'Delta pp', 'Correct42', 'Correct1024',
        'Recall@5 42', 'Recall@5 1024', 'Judge42', 'Judge1024', 'Storage42 MiB', 'Storage1024 MiB',
        'Two-seed mean accuracy', 'Two-seed sample SD', 'Hippo context coverage42', 'Hippo context coverage1024'])
    transitions = workbook.create_sheet('Paired_MCQ')
    transitions.append(['Method', 'Both correct', '42 only correct', '1024 only correct', 'Both wrong',
                        'Changed A/B/C/D/None choices', 'Accuracy change in correct answers', 'All questions retained',
                        'Changed native predictions including invalid text'])
    coverage = workbook.create_sheet('Coverage_and_native_outputs')
    coverage.append(['Method', 'MCQ complete', 'MCQ records', 'Unique IDs', 'Open verified', 'Judge verified',
        'Gold source hits', 'Gold source references', 'Format errors', 'Answer caps', 'Memory caps',
        'Invalid structured memory outputs', 'Native outcomes', 'Empty Open answers', 'Open caps', 'Judge grade distribution',
        'Permanent provider input rejections', 'Invalid choice records including native failures',
        'Historical API error attempts', 'Errors with later identical-wire success', 'Historical failure snapshots retained'])
    storage = workbook.create_sheet('Storage_files')
    storage.append(['Method', 'Absolute file path', 'Logical bytes', 'SHA256'])
    provenance = workbook.create_sheet('Artifacts')
    provenance.append(['Method', 'Artifact', 'Absolute path', 'SHA256'])
    mcq = {r['method']: r for r in reports[0].get('methods', [])}
    opened = {r['method']: r for r in reports[1].get('methods', [])}
    for row in result:
        method = row['method']; record, op = mcq.get(method, {}), opened.get(method, {})
        note = ('Context source coverage includes fast-summary provenance; literal Recall@5 is blank.' if method == 'hippomm' else
                'Native Top10/dynamic contexts; Recall@5 not reported.' if method in ('gen_agents', 'memoryos') else
                'Conservative post-hoc source attribution; fresh evidence sealed before scoring.' if method == 'simplemem' else '')
        main.append([row['label'], row['status'], row['seed1024_accuracy'], row['seed1024_recall_at5_micro'],
            row['seed1024_llm_judge'], row['seed1024_storage_mib'], row['seed1024_correct'], 371,
            row['seed1024_context_source_micro_recall'], note])
        paired.append([row['label'], row['seed42_accuracy'], row['seed1024_accuracy'], row['accuracy_delta_pp'],
            row['seed42_correct'], row['seed1024_correct'], row['seed42_recall_at5_micro'], row['seed1024_recall_at5_micro'],
            row['seed42_llm_judge'], row['seed1024_llm_judge'], row['seed42_storage_mib'], row['seed1024_storage_mib'],
            row['two_seed_accuracy_mean'], row['two_seed_accuracy_sample_sd'],
            row['seed42_context_source_micro_recall'], row['seed1024_context_source_micro_recall']])
        transitions.append([row['label'], row['both_correct'], row['seed42_only_correct'], row['seed1024_only_correct'],
            row['both_wrong'], row['changed_predicted_answers'],
            row['seed1024_correct'] - row['seed42_correct'] if row['seed1024_correct'] is not None else None, 371,
            row['changed_native_predictions']])
        if record.get('mcq_complete'):
            actual = audit.read_mcq(method, output); ids = [r['id'] for r in actual]
            audit.require(len(ids) == len(set(ids)) == 371, 'Workbook MCQ coverage differs')
        else:
            actual, ids = [], []
        coverage.append([row['label'], bool(record.get('mcq_complete')), len(actual), len(set(ids)),
            op.get('open_answers_verified', 0), op.get('judge_scores_verified', 0),
            record.get('recall_numerator'), record.get('recall_denominator'), row['format_errors'],
            row['answer_token_limit_hits'], row['memory_token_limit_hits'], record.get('invalid_structured_outputs'),
            json.dumps(row['native_outcomes'], ensure_ascii=False), row['empty_open_answers'], row['open_token_limit_hits'],
            json.dumps(op.get('judge', {}).get('score_distribution')), row['provider_input_rejections'], row['invalid_choice_records'],
            record.get('historical_api_error_events'), record.get('historical_api_errors_with_later_exact_wire_success'),
            len(record.get('failure_snapshots_verified', [])) if 'failure_snapshots_verified' in record else None])
        metric = mapping.metric_path(method, output)
        if metric.exists():
            for item in audit.read(metric).get('storage_components', []):
                storage.append([row['label'], item['path'], item['bytes'], item['sha256']])
        paths = audit.method_paths(method, output)
        open_root = Path(op.get('source_open_root', paths['opened']))
        files = [('MCQ responses', paths['mcq']), ('MCQ summary', paths['summary']), ('Metric summary', paths['metric']),
                 ('Open source protocol', open_root / 'protocol.json'),
                 ('Judge summary', open_root / 'judge/llm_judge_summary.json'),
                 ('Judge per question', open_root / 'judge/per_question.jsonl')]
        old_source = Path(mapping.reference_mapping()[method]['source_root'])
        if method in mapping.OURS:
            files += [('Seed42 MCQ responses', old_source / folder / 'reader/responses.jsonl')
                      for folder in ('weighted_evaluation', 'complementary_evaluation')]
        else:
            files.append(('Seed42 MCQ responses', old_source / ('online/qa_responses.jsonl' if method == 'hippomm' else 'qa_responses.jsonl')))
        for name, path in files:
            if path.exists():
                provenance.append([row['label'], name, str(path), audit.sha(path)])
    shared = [output / 'plan.json', *report_paths, mapping.REFERENCE / 'storage_summary.json',
              JUDGE42 / 'audit.json', JUDGE42 / 'completion.json', output / 'ours/independent_mcq_paired_summary.json',
              output / 'ours/independent_mcq_paired_records.jsonl',
              output / 'metrics/paired_mcq_audit.json', output / 'metrics/paired_mcq_records.jsonl',
              output / 'metrics/paired_mcq_report_validation.json',
              output / 'metrics/simplemem_source_attribution/annotation_lock.json',
              output / 'metrics/simplemem_source_attribution/manual_review.jsonl',
              output / 'metrics/simplemem_source_attribution/recall_summary.json', Path(__file__), Path(audit.__file__)]
    for path in shared:
        if path.exists():
            provenance.append(['ALL', path.name, str(path), audit.sha(path)])
    notes = workbook.create_sheet('Protocol_notes'); notes.append(['Item', 'Definition / limitation'])
    values = [
        ('Status', 'Complete only after both10-method independent audits pass, all3710 MCQs/Open answers/Judge grades exist, and fresh SimpleMem attribution is sealed and scored.' if ready else
                   'DRAFT: pending values are blank. Current successful stages do not establish final completion.'),
        ('Controlled comparison', 'Both seeds use the same provider alias deepseek-v4.1-flash, exact original DeepSeek seed42 captions, canonical371 questions and original native method configurations. Memory/retrieval/reader use1024; judge and local Qwen source attribution remain42.'),
        ('Attribution limit', 'Requests came from different batches/times: MCQ42 controls include earlier Oct25/29 and Nov4 runs; Open42 and this replication are different batches on Nov6. The provider supplied no fixed weight revision or stable system fingerprint. Differences cannot strictly be attributed only to seed; UUIDs, timestamps, scheduling and numeric kernels may also vary.'),
        ('Ours memory and retrieval', 'Fresh construction verified identical core graphs/bases/master vectors and identical Top5 retrieval to42. Similar aggregate accuracy still permits changed individual choices; see paired MCQ transitions and source artifact links.'),
        ('Accuracy', 'Fixed-option MCQ accuracy=correct/371; not the official four-rotation EM. Native invalid answers, permanent provider input rejections and token caps remain in the denominator.'),
        ('Two seeds', 'Mean=(accuracy42+accuracy1024)/2; sample SD=abs(accuracy1024-accuracy42)/sqrt(2). Two seeds provide a limited stability check, not statistical significance or general robustness.'),
        ('Recall', 'Source-round hits/1041, deduplicated within question. GA Top10 and MemoryOS dynamic context keep Recall@5 blank. A-Mem follows native up-to-five notes, including any genuine short/empty context.'),
        ('HippoMM', 'Context source coverage includes fast-path summary provenance and is not literal Recall@5. It is kept in a separate column; both seeds use logical MiB=bytes/1048576.'),
        ('SimpleMem', 'Fresh seed1024 generated entries require new two-pass fixed-Qwen42 source annotation, literal-quote validation, genuine gold-blind flagged-entry review, and an immutable seal before scoring. No seed42 memory annotations are reused. This is reconstructed provenance rather than native source IDs/human gold.'),
        ('Storage', 'Logical bytes of native persisted core memory/1048576; no model weights, questions, answer/query caches or logs. GA/A-Mem are snapshots without complete persisted retrieval indices. MemoryOS decimal embeddings and SimpleMem retained database versions count. Native formats are not a universal theoretical storage comparison.'),
        ('Ours storage', 'Identical memory payload files; only source-provenance strings in memory.json change the counted manifest bytes. Small metadata byte deltas do not imply changed memory retention.'),
        ('Native failures', 'Coverage sheet lists real malformed outputs, memory/answer caps, empty answers and permanent provider outcomes. No answer-quality retries, silent sample exclusion or missing-score imputation are introduced by the audit/export.'),
        ('Aborted-run history', 'Native API error attempts from interrupted runs remain in append-only ledgers and preserved failure snapshots. Later identical-wire success is reported separately. Successful cached calls remain unchanged; reruns rebuild interrupted task state without erasing failed attempts.'),
        ('Source integrity', 'Artifacts tab contains absolute paths and SHA256 values. Export never calls an API, loads a GPU model, changes pinned source or marks progress complete.'),
    ]
    for item in values:
        notes.append(item)
    for sheet in workbook:
        sheet.freeze_panes = 'B2'; sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='234E70'); cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        sheet.row_dimensions[1].height = 35
        for cells in sheet.iter_rows(min_row=2):
            for cell in cells:
                cell.alignment = Alignment(vertical='top', wrap_text=sheet.title in ('Protocol_notes', 'Coverage_and_native_outputs'))
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000000'
        for col in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(col)].width = 24 if col > 1 else 48
    main.column_dimensions['J'].width = 95; notes.column_dimensions['B'].width = 150
    provenance.column_dimensions['C'].width = 115; provenance.column_dimensions['D'].width = 70
    storage.column_dimensions['B'].width = 115; storage.column_dimensions['D'].width = 70
    destination.parent.mkdir(parents=True, exist_ok=True); workbook.save(destination)
    reopened = load_workbook(destination, read_only=True, data_only=True)
    audit.require(all(reopened[s].max_row == 11 for s in ('Seed1024', 'Seed42_vs_1024', 'Paired_MCQ', 'Coverage_and_native_outputs')), 'Workbook row coverage differs')
    for i, item in enumerate(result, 2):
        audit.require(reopened['Seed1024'].cell(i, 7).value == item['seed1024_correct'], 'Workbook correct count roundtrip differs')
        for column, key in ((3, 'seed1024_accuracy'), (5, 'seed1024_llm_judge'), (6, 'seed1024_storage_mib')):
            original = item[key]; actual = reopened['Seed1024'].cell(i, column).value
            audit.require((original is None and actual is None) or (original is not None and abs(actual - original) < 1e-12), 'Workbook metric roundtrip differs')
    report = dict(status='complete' if ready else 'draft_partial', passed=ready, methods=10,
        mcq_records=reports[0].get('mcq_records_total', 0),
        open_answers=reports[1].get('open_answers_verified', 0), judge_scores=reports[1].get('judge_scores_verified', 0),
        path=str(destination), sha256=audit.sha(destination), sheets=reopened.sheetnames,
        implementation_sha256=audit.sha(__file__), audit_source_sha256={str(p): audit.sha(p) for p in report_paths if p.exists()})
    audit.dump(output / ('workbook_audit.json' if ready else 'workbook_draft_audit.json'), report)
    audit.dump(output / ('comparison.json' if ready else 'comparison_draft.json'), dict(status=report['status'], methods=result))
    csv_path = output / ('comparison.csv' if ready else 'comparison_draft.csv')
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0])); writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in result)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--dest', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    export(args.root, args.dest, args.allow_partial)
