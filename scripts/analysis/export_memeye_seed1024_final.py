"""Export the completed MemEye fixed-caption two-seed stability comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from scripts.evaluation import memeye_seed1024_suite as suite

ROOT, OUT = suite.ROOT, suite.OUT
DEST = OUT / 'MemEye_seed42_vs_seed1024.xlsx'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def seal_completion():
    reports = [OUT / 'recovery_20261106/final_audit_mcq_metrics.json',
               OUT / 'recovery_20261106/final_audit_open_judge.json']
    for path in reports:
        audit = read(path)
        assert audit.get('passed') is True or audit.get('status') == 'passed', str(path)
    plan = read(OUT / 'plan.json')
    suite.validate_sources(plan)
    state = read(OUT / 'progress.json')
    assert not state.get('running_stages'), 'A generation process is still registered as running'
    backup = OUT / 'recovery_20261106/progress_before_final_seal.json'
    if not backup.exists():
        suite.dump(backup, state)
    for stage in plan['stages']:
        if stage.get('marker'):
            assert suite.completed_marker(stage), stage['name']
            entry = state['stages'][stage['name']]
            if entry['status'] != 'complete':
                state.setdefault('final_recovered_stage_status', {})[stage['name']] = dict(entry)
                entry.update(status='complete', final_artifact_recovery=True,
                             marker_sha256=suite.sha(stage['marker']), finished_at_utc=suite.now())
        assert state['stages'][stage['name']]['status'] == 'complete', stage['name']
    result = suite.comparison(plan)
    assert result['mcq_methods_complete'] == result['judge_methods_complete'] == 10
    assert all(r['seed1024_storage_mib'] is not None for r in result['methods'])
    assert all(r['seed1024_recall'] is not None for r in result['methods'] if r['seed42_recall'] is not None)
    files = [*reports, OUT / 'plan.json', OUT / 'comparison.json', OUT / 'comparison.csv',
             DEST, OUT / 'workbook_audit.json']
    completion = {'status': 'complete', 'finished_at_utc': suite.now(), 'seed': 1024,
        'reference_seed': 42, 'judge_seed': 42, 'methods': 10, 'questions_per_method': 371,
        'mcq_question_records': 3710, 'open_answers': 3710, 'judge_question_scores': 3710,
        'missing_mcq_ids': 0, 'missing_open_ids': 0, 'missing_judge_scores': 0,
        'all_required_metrics_complete': True, 'independent_audits_passed': True,
        'source_sha256': {str(path): suite.sha(path) for path in files},
        'completion_note': 'Final judge format recovery validated from preserved raw outputs; original partial evaluation files retained.'}
    suite.dump(OUT / 'completion_manifest.json', completion)
    state.update(status='complete', current_stage=None, running_stages=[], unfinished_stages=[],
                 unfinished_metric_methods=[], mcq_methods_complete=10, judge_methods_complete=10,
                 updated_at_utc=suite.now(), completion_manifest=str(OUT / 'completion_manifest.json'))
    suite.dump(OUT / 'progress.json', state)
    previous = OUT / 'recovery_20261106/completion.json'
    preserved = OUT / 'recovery_20261106/completion_before_final_seal.json'
    if previous.exists() and not preserved.exists():
        suite.dump(preserved, read(previous))
    suite.dump(previous, completion)
    print(json.dumps({'status': 'complete', 'mcq_methods': 10, 'judge_methods': 10,
                      'independent_audits_passed': True}))


def main():
    plan = read(OUT / 'plan.json')
    result = suite.comparison(plan)
    assert result['mcq_methods_complete'] == result['judge_methods_complete'] == 10
    assert all(m['seed1024_storage_mib'] is not None for m in result['methods'])
    assert all(m['seed1024_recall'] is not None for m in result['methods'] if m['seed42_recall'] is not None)
    workbook = Workbook()
    main_sheet = workbook.active
    main_sheet.title = 'Seed1024'
    main_sheet.append(['Method', 'Accuracy', 'Recall@5 (micro)', 'LLM Judge', 'Storage (MiB)',
                       'Correct', 'Questions', 'Metric note'])
    paired = workbook.create_sheet('Seed42_vs_1024')
    paired.append(['Method', 'Accuracy seed42', 'Accuracy seed1024', 'Delta accuracy (pp)',
        'Correct seed42', 'Correct seed1024', 'Recall seed42', 'Recall seed1024',
        'Judge seed42', 'Judge seed1024', 'Storage seed42 (MiB)', 'Storage seed1024 (MiB)',
        'Two-seed accuracy mean', 'Two-seed accuracy sample SD'])
    validation = workbook.create_sheet('Coverage')
    validation.append(['Method', 'MCQ records', 'Unique MCQ IDs', 'Open answers', 'Judge scores',
                       'Gold source links', 'Gold hits', 'Judge score sum', 'Judge grades'])
    provenance = workbook.create_sheet('Artifacts')
    provenance.append(['Method', 'Artifact', 'Absolute path', 'SHA256'])
    source_methods = {m['id']: m for m in plan['methods']}
    for row in result['methods']:
        mid = row['id']
        source = source_methods[mid]
        note = ('Context-source coverage, including fast summary provenance; not literal Top5.'
                if mid == 'hippomm' else 'Native Top10/dynamic context; Recall@5 not reported.'
                if mid in ('gen_agents', 'memoryos') else
                'Conservative post-hoc source attribution, independently sealed before scoring.'
                if mid == 'simplemem' else '')
        main_sheet.append([row['label'], row['seed1024_accuracy'], row['seed1024_recall'],
            row['seed1024_judge'], row['seed1024_storage_mib'], row['seed1024_correct'], 371, note])
        paired.append([row['label'], row['seed42_accuracy'], row['seed1024_accuracy'], row['accuracy_delta_pp'],
            row['seed42_correct'], row['seed1024_correct'], row['seed42_recall'], row['seed1024_recall'],
            row['seed42_judge'], row['seed1024_judge'], row['seed42_storage_mib'], row['seed1024_storage_mib'],
            row['two_seed_accuracy_mean'], row['two_seed_accuracy_sample_sd']])
        mcq = rows(source['mcq_responses'])
        if source.get('condition'):
            mcq = [r for r in mcq if r['condition'] == source['condition']]
        identities = [r.get('question_id', r.get('id')) for r in mcq]
        assert len(identities) == len(set(identities)) == 371
        actual_open = Path(source.get('source_open_root', source['open_root']))
        answers_path = actual_open / ('qa_responses.jsonl' if mid == 'hippomm' else 'reader/responses.jsonl')
        answers = rows(answers_path)
        assert len(answers) == len({r['id'] for r in answers}) == 371
        judge_root = Path(source['open_root']) / 'judge'
        judged = rows(judge_root / 'per_question.jsonl')
        assert len(judged) == len({r['id'] for r in judged}) == 371
        assert all(r['judge_status'] == 'ok' for r in judged)
        assert set(identities) == {r['id'] for r in answers} == {r['id'] for r in judged}
        assert sum(r['score'] for r in judged) / 371 == row['seed1024_judge']
        hits = round(row['seed1024_recall'] * 1041) if row['seed1024_recall'] is not None else None
        validation.append([row['label'], len(mcq), len(set(identities)), len(answers), len(judged),
                           1041 if hits is not None else None, hits, sum(r['score'] for r in judged),
                           json.dumps({str(g): sum(r['score'] == g for r in judged) for g in (0, .25, .5, .75, 1)})])
        files = {'MCQ responses': Path(source['mcq_responses']), 'MCQ summary': Path(source['mcq_summary']),
            'Metric summary': Path(source['metric_summary']), 'Open answers': answers_path,
            'Judge summary': judge_root / 'summary.json', 'Judge protocol': judge_root / 'protocol.json',
            'Judge per question': judge_root / 'per_question.jsonl'}
        for name, path in files.items():
            provenance.append([row['label'], name, str(path), suite.sha(path)])
    shared_files = [OUT / 'plan.json',
        OUT / 'recovery_20261106/final_audit_mcq_metrics.json',
        OUT / 'recovery_20261106/final_audit_open_judge.json',
        OUT / 'metrics/simplemem_source_attribution/annotation_lock.json',
        OUT / 'metrics/simplemem_source_attribution/manual_review.jsonl',
        OUT / 'metrics/simplemem_source_attribution/recall_summary.json']
    for path in shared_files:
        if path.exists():
            provenance.append(['ALL / shared evaluator', path.name, str(path), suite.sha(path)])
    notes = workbook.create_sheet('Protocol_notes')
    notes.append(['Item', 'Definition / limitation'])
    values = [
        ('Completion', '10 methods × 371 MCQ records; 3710 Open answers and 3710 scored question records. No missing-score imputation.'),
        ('Seeds', 'Memory construction/retrieval/reader seed1024; original historical and query-image captions fixed. Judge and SimpleMem source-attribution evaluator seed42.'),
        ('Accuracy', 'Fixed-option MCQ accuracy, correct/371; not the official four-rotation EM.'),
        ('Two seeds', 'Sample SD = abs(accuracy1024-accuracy42)/sqrt(2). Two seeds provide a limited stability check, not general robustness or statistical significance.'),
        ('Native variation', 'MemoryOS/SimpleMem UUIDs, wall-clock maintenance, asynchronous batching and numerical backends can vary; differences are not a pure causal effect of seed alone.'),
        ('Recall', 'Original source-round hits/1041, deduplicated within question. GA and MemoryOS are native Top10/dynamic contexts and retain blank Recall@5.'),
        ('HippoMM recall', '823/1041 context-source coverage includes fast-path summary provenance and is not literal Top5.'),
        ('SimpleMem recall', '350/1041 conservative reconstructed provenance; 641 annotated entries, 596 selected; 151 explicit gold-blind reviews. Annotation sealed before reading evaluation gold.'),
        ('Storage', 'Uncompressed persisted-memory logical bytes /1048576. Exclude model weights, queries, answers, logs and request replay caches. GA/A-Mem are saved snapshots; complete retrieval vectors were not persisted.'),
        ('HippoMM unit correction', 'Original168.82 is decimalMB:168822299bytes=168.822299MB=161.00149059295654MiB. Both seeds use the corrected MiB convention here.'),
        ('Ours storage delta', 'All memory payloads byte-identical; small byte-size differences arise solely from source-directory strings in manifests.'),
        ('Output caps', 'Within the MCQ workflow, MemoryOS139 and SimpleMem1 memory-generation calls hit native output limits; retained rather than silently replaced. MCQ final-answer format errors and token-limit hits are zero.'),
        ('Open answer limits', 'All3710 Open answers were judged as originally emitted: empty answers0; token-limit flags30 retained without replacement or missing-score imputation.'),
        ('Judge format recovery', 'VisRAG and HippoMM each had one malformed raw response after five attempts. Every explicit JSON grade in all five outputs agreed at0.75. A separate parser-bound recovery preserves full raw strings and uses no new API calls; original partial roots remain intact.'),
        ('HippoMM query recovery', '13 failed MCQ rewrite constraints were recovered via separately recorded bounded format adaptation, preserving original358 answers,250-event memory and5507 original cached calls.'),
    ]
    for item in values:
        notes.append(item)
    for sheet in workbook:
        sheet.freeze_panes = 'B2'
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 32
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='234E70')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=sheet.title in ('Protocol_notes', 'Coverage'))
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000000'
        for index in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(index)].width = 25 if index > 1 else 47
    main_sheet.column_dimensions['H'].width = 85
    provenance.column_dimensions['C'].width = 115
    provenance.column_dimensions['D'].width = 69
    notes.column_dimensions['B'].width = 150
    main_sheet.column_dimensions['E'].width = 20
    for cell in main_sheet['E'][1:]:
        cell.number_format = '0.000'
    for cell in paired['D'][1:]:
        cell.number_format = '+0.000;-0.000;0.000'
        cell.font = Font(color='16803C' if cell.value > 0 else 'B42318' if cell.value < 0 else '404040')
    workbook.save(DEST)
    reopened = load_workbook(DEST, read_only=True, data_only=True)
    assert reopened['Seed1024'].max_row == reopened['Seed42_vs_1024'].max_row == reopened['Coverage'].max_row == 11
    for i, row in enumerate(result['methods'], 2):
        assert reopened['Seed1024'].cell(i, 6).value == row['seed1024_correct']
        assert abs(reopened['Seed1024'].cell(i, 4).value - row['seed1024_judge']) < 1e-14
    suite.dump(OUT / 'workbook_audit.json', {'status': 'complete', 'sheets': reopened.sheetnames,
        'methods': 10, 'questions_per_method': 371, 'mcq_records': 3710, 'open_answers': 3710,
        'judge_question_scores': 3710, 'path': str(DEST), 'sha256': suite.sha(DEST),
        'source_plan_sha256': suite.sha(OUT / 'plan.json'), 'implementation_sha256': suite.sha(Path(__file__))})
    print(DEST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true', help='Require both independent audits and seal final completion.')
    args = parser.parse_args()
    main()
    if args.seal:
        seal_completion()
