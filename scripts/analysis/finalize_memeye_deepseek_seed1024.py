"""Seal a completed, audited run. The default drycheck never writes files.

This tool neither runs stages nor repairs failed/skipped progress entries.
Only --apply may preserve the generation progress and set final completion,
after the original 47-stage dispatcher has exited and every evidence gate passes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
METHODS = ('semrag', 'ours_1', 'ours_2', 'ours_1_ours_2', 'gen_agents', 'memoryos',
           'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm')
STAGES = frozenset('''ours_rebuild ours_retrieve clip_image_dialogue_index hippomm_index
gen_agents_prepare memoryos_prepare a_mem_prepare simplemem_prepare ours_mcq gen_agents_mcq
memoryos_mcq a_mem_mcq simplemem_mcq clip_image_dialogue_mcq hippomm_mcq ours_open_candidates
ours_open_retrieve ours_open_answers gen_agents_open_retrieve gen_agents_open_answers
gen_agents_open_collect memoryos_open_retrieve memoryos_open_answers memoryos_open_collect
a_mem_open_retrieve a_mem_open_answers a_mem_open_collect simplemem_open_retrieve
simplemem_open_answers simplemem_open_collect clip_image_dialogue_open_answers
clip_image_dialogue_open_collect hippomm_open_prepare hippomm_open_answers semrag_judge42
ours_1_judge42 ours_2_judge42 ours_1_ours_2_judge42 gen_agents_judge42 memoryos_judge42
simplemem_judge42 a_mem_judge42 clip_image_dialogue_judge42 hippomm_judge42
metrics_before_attribution simplemem_attribution_prepare simplemem_attribution_run'''.split())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def process_active(pid):
    """Stopped processes are active; only absent/dead/zombie PIDs count as exited."""
    require(type(pid) is int and pid > 0, 'Missing valid dispatcher PID')
    path = Path('/proc') / str(pid) / 'stat'
    try:
        state = path.read_text().rsplit(')', 1)[1].strip().split()[0]
    except FileNotFoundError:
        return False
    return state not in ('Z', 'X', 'x')


def same_number(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    return isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual - expected) < 1e-12


class Inspector:
    def __init__(self, root, suite_locked=False):
        self.root = Path(root).resolve()
        self.suite_locked = suite_locked
        self.files = {}
        self.stats = {}
        self.issues = []
        self.gates = []
        self.plan = self.progress = self.mcq = self.opened = self.metrics = self.paired = None

    def sha(self, path):
        path = Path(path).resolve()
        stat = path.stat(); key = str(path)
        current = (stat.st_size, stat.st_mtime_ns)
        if key not in self.stats:
            with path.open('rb') as handle:
                value = hashlib.file_digest(handle, 'sha256').hexdigest()
            require((path.stat().st_size, path.stat().st_mtime_ns) == current,
                    'File changed during hashing: ' + key)
            self.stats[key] = current; self.files[key] = value
        else:
            require(self.stats[key] == current, 'Evidence changed during preflight: ' + key)
        return self.files[key]

    def read(self, path):
        self.sha(path)
        return json.loads(Path(path).read_text())

    def rows(self, path):
        self.sha(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def hashes(self, values):
        require(isinstance(values, dict) and values, 'Missing evidence SHA256 map')
        for path, expected in values.items():
            require(isinstance(expected, str) and len(expected) == 64 and self.sha(path) == expected,
                    'Current evidence hash mismatch: ' + str(path))

    def gate(self, name, function):
        try:
            function()
            self.gates.append(dict(gate=name, passed=True))
        except Exception as error:
            self.issues.append(dict(gate=name, error_type=type(error).__name__, error=str(error)))
            self.gates.append(dict(gate=name, passed=False))

    def generation(self):
        self.plan = self.read(self.root / 'plan.json')
        self.progress = self.read(self.root / 'progress.json')
        p, state = self.plan, self.progress
        require(p['model'] == state['model'] == 'deepseek-v4.1-flash' and
                p['seed'] == state['seed'] == 1024 and p['caption_seed'] == p['judge_seed'] == state['judge_seed'] == 42,
                'Model/seed controls differ')
        require(state['status'] == 'generation_complete_pending_final_audit',
                'Original dispatcher has not reached generation_complete_pending_final_audit')
        require(state.get('current_stage') is None and state.get('unfinished_stages') == [],
                'Dispatcher still records an active/unfinished stage')
        require(not process_active(state.get('pid')), 'Original dispatcher PID is still active (including stopped processes)')
        if not self.suite_locked:
            with (self.root / 'suite.lock').open('r') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError('Original dispatcher still holds suite.lock') from error
        self.hashes(p['source_sha256'])
        consent = self.read(self.root / 'export_authorization.json')
        require(consent['approved'] is True and consent['source'] == 'explicit_user_reply' and
                consent['model'] == 'deepseek-v4.1-flash' and
                consent['destination'] == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions' and
                consent['manifest_sha256'] == self.sha(self.root / 'data_export_manifest.json'),
                'Recorded explicit dataset-export authorization no longer matches its manifest')

    def stages(self):
        # Read independently so this gate can diagnose true stage states even while generation is active.
        plan = self.plan or self.read(self.root / 'plan.json')
        state = self.progress or self.read(self.root / 'progress.json')
        stages = plan['stages']; names = [s['name'] for s in stages]
        require(len(names) == len(set(names)) == 47 and set(names) == STAGES,
                'Expected the exact 47-stage original suite')
        require(set(state['stages']) == STAGES, 'Progress stage coverage differs from original47')
        bad = [name for name in names if state['stages'][name].get('status') != 'complete']
        require(not bad, 'Original suite must actually complete/adopt stages: ' + ', '.join(bad))
        for item in stages:
            name = item['name']
            require(set(item.get('depends_on', [])) <= STAGES, 'Unknown dependency: ' + name)
            if item.get('marker'):
                path = Path(item['marker']).resolve()
                require(path.is_relative_to(self.root), 'Stage marker is outside new run: ' + name)
                marker = self.read(path)
                require(marker.get(item['marker_key']) in item['marker_values'], 'Actual marker not complete: ' + name)
            elif name.endswith('_open_collect'):
                args = item['argv']; path = Path(args[args.index('--root') + 1]).resolve()
                require(path.is_relative_to(self.root), 'Collected Open source outside new run')
                complete = self.read(path / 'reader/completion.json')
                response = path / 'reader/responses.jsonl'; answers = self.rows(response)
                inputs = self.rows(path / 'judge_inputs.jsonl')
                require(complete['status'] == 'complete' and complete['questions_answered'] == 371 and
                        complete['responses_sha256'] == self.sha(response) and
                        complete['protocol_sha256'] == self.sha(path / 'reader/protocol.json'),
                        'Collect lacks real sealed native answers: ' + name)
                by_id = {r['id']: r for r in answers}
                require(len(answers) == len(inputs) == len(by_id) == len({r['id'] for r in inputs}) == 371 and
                        set(by_id) == {r['id'] for r in inputs}, 'Collect ID coverage differs: ' + name)
                require(all(by_id[r['id']]['seed'] == 1024 and r['answer_model'] == 'deepseek-v4.1-flash' and
                            r['source_responses_sha256'] == self.sha(response) and
                            r['source_api_request_sha256'] == by_id[r['id']]['request_sha256'] and
                            r['model_output'] == by_id[r['id']]['raw_response'] for r in inputs),
                        'Judge input differs from actual collected Open answer: ' + name)
            elif name == 'hippomm_open_prepare':
                path = self.root / 'visual_methods/hippomm/open'
                p = self.read(path / 'protocol.json'); questions = self.rows(path / 'questions.jsonl')
                require(p['method_id'] == 'hippomm' and p['answer_model'] == 'deepseek-v4.1-flash' and
                        p['seed'] == 1024 and p['evaluator_seed'] == 42 and p['questions'] == 371 and
                        len(questions) == len({q['id'] for q in questions}) == 371 and
                        p['questions_sha256'] == self.sha(path / 'questions.jsonl') and
                        Path(p['memory_dir']).resolve() == self.root / 'visual_methods/hippomm/offline',
                        'HippoMM Open preparation evidence differs')
                self.hashes(p['source_sha256'])
            else:
                raise ValueError('Unrecognized markerless stage: ' + name)

    def independent_audits(self):
        self.mcq = self.read(self.root / 'audit/final_audit_mcq_metrics.json')
        self.opened = self.read(self.root / 'audit/final_audit_open_judge.json')
        for value in (self.mcq, self.opened):
            require(value['status'] == 'passed' and value['passed'] is True,
                    'Both independent all10 audits must pass')
            self.hashes(value['artifact_sha256'])
            require(len(value['methods']) == 10 and {m['method'] for m in value['methods']} == set(METHODS) and
                    all(m['status'] == 'passed' for m in value['methods']), 'Independent method coverage differs')
        require(self.mcq['methods_complete'] == self.mcq['metrics_complete'] == 10 and
                self.mcq['mcq_records_total'] == 3710 and self.mcq['metric_index_complete'] is True and
                self.mcq['metric_index_unique_methods'] == 10 and not self.mcq['failures'] and
                self.mcq['seed'] == 1024 and self.mcq['reference_seed'] == self.mcq['judge_seed'] == 42,
                'MCQ/metric audit is incomplete')
        require(self.mcq['source_plan_sha256'] == self.sha(self.root / 'plan.json') and
                self.mcq['implementation_sha256'] == self.sha(ROOT / 'scripts/analysis/audit_memeye_deepseek_seed1024.py') and
                self.mcq['metric_index_sha256'] == self.sha(self.root / 'metrics/metric_summary.json'),
                'MCQ audit code/plan/index has changed')
        require(self.opened['methods_passed'] == self.opened['expected_methods'] == 10 and
                self.opened['open_answers_verified'] == self.opened['judge_scores_verified'] == 3710 and
                all(m['open_answers_verified'] == m['judge_scores_verified'] == 371 for m in self.opened['methods']),
                'Open/Judge audit is incomplete')
        open_code = str(ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_audit_open_judge.py')
        require(self.opened['plan_sha256_at_read'] == self.sha(self.root / 'plan.json') and
                self.opened['artifact_sha256'].get(open_code) == self.sha(open_code) and
                self.opened['answer_seed'] == 1024 and self.opened['judge_seed'] == 42 and
                self.opened['real_api_calls_by_audit'] == 0 and self.opened['errors'] == self.opened['pending'] == [],
                'Open/Judge audit code/plan/seed/completion fields differ')

    def metrics_index(self):
        self.metrics = self.read(self.root / 'metrics/metric_summary.json')
        require(self.metrics['status'] == self.metrics['mcq_storage_status'] == 'complete',
                'Full metric index is partial')
        methods = self.metrics['methods']
        require(len(methods) == 10 and {m['method'] for m in methods} == set(METHODS), 'Metric index must have ten unique methods')
        for item in methods:
            require(item['status'] == 'complete' and item['questions'] == 371 and
                    item['seed'] == 1024 and item['model'] == 'deepseek-v4.1-flash' and
                    same_number(item['accuracy'], item['correct'] / 371), 'Incomplete method metric: ' + item['method'])
            components = item['storage_components']; paths = [c['path'] for c in components]
            require(len(paths) == len(set(paths)) and components, 'Duplicate or missing storage components')
            require(sum(c['bytes'] for c in components) == item['storage_bytes'] and
                    same_number(item['storage_mib'], item['storage_bytes'] / 1048576), 'Actual storage byte/MiB sum differs')
            for c in components:
                require(Path(c['path']).stat().st_size == c['bytes'] and self.sha(c['path']) == c['sha256'],
                        'Persisted storage bytes/hash changed: ' + c['path'])
            if item['method'] == 'hippomm':
                require(item.get('recall_at5_micro') is None and item.get('context_source_micro_recall') is not None,
                        'HippoMM context coverage must remain distinct from literal Recall@5')

    def paired_mcq(self):
        self.paired = self.read(self.root / 'metrics/paired_mcq_audit.json')
        report = self.paired
        require(report['status'] == 'complete' and report['methods_expected'] == report['methods_complete'] == 10 and
                report['methods_pending'] == report['methods_failed_audit'] == 0 and report['paired_records'] == 3710 and
                set(report['methods']) == set(METHODS), 'Full paired MCQ audit is not complete')
        self.hashes(report['source_sha256'])
        path = Path(report['paired_records_path']).resolve()
        require(path == self.root / 'metrics/paired_mcq_records.jsonl' and self.sha(path) == report['paired_records_sha256'],
                'Paired record source differs')
        groups = defaultdict(list)
        for row in self.rows(path):
            groups[row['method']].append(row)
        require(set(groups) == set(METHODS), 'Paired record method coverage differs')
        for method, item in report['methods'].items():
            require(item['status'] == 'complete' and item['questions'] == 371, 'Partial paired method: ' + method)
            self.hashes(item['seed42_source']['source_sha256']); self.hashes(item['seed1024_source']['source_sha256'])
            records = groups[method]
            require(len(records) == len({r['id'] for r in records}) == 371, 'Paired IDs incomplete: ' + method)
            checks = dict(seed42_correct=sum(r['seed42_correct'] for r in records),
                seed1024_correct=sum(r['seed1024_correct'] for r in records),
                both_correct=sum(r['seed42_correct'] and r['seed1024_correct'] for r in records),
                seed42_only_correct=sum(r['seed42_correct'] and not r['seed1024_correct'] for r in records),
                seed1024_only_correct=sum(not r['seed42_correct'] and r['seed1024_correct'] for r in records),
                both_wrong=sum(not r['seed42_correct'] and not r['seed1024_correct'] for r in records),
                choice_changed=sum(r['seed42_choice'] != r['seed1024_choice'] for r in records),
                native_prediction_changed=sum(r['native_prediction_changed'] for r in records))
            require(all(item[k] == value for k, value in checks.items()), 'Paired aggregates differ: ' + method)
            require(all(r['choice_changed'] == (r['seed42_choice'] != r['seed1024_choice']) and
                        r['seed42_choice'] in ('A', 'B', 'C', 'D', None) and
                        r['seed1024_choice'] in ('A', 'B', 'C', 'D', None) for r in records), 'Paired normalized choices differ')
            if self.mcq:
                independent = next(m for m in self.mcq['methods'] if m['method'] == method)
                require(independent['correct'] == checks['seed1024_correct'] and
                        independent['changed_predicted_answers_vs_seed42'] == checks['choice_changed'] and
                        independent['changed_native_predictions_vs_seed42'] == checks['native_prediction_changed'],
                        'Independent MCQ and paired audit disagree: ' + method)

    def attribution(self):
        folder = self.root / 'metrics/simplemem_source_attribution'
        p = self.read(folder / 'protocol.json'); lock = self.read(folder / 'annotation_lock.json')
        score = self.read(folder / 'recall_summary.json')
        source = self.root / 'text_methods/simplemem'
        native = self.read(source / 'protocol.json')
        require(p['seed'] == 42 and p['memory_generation_seed'] == 1024 and p['fresh_source_only'] is True and
                p['gold_blind'] is True and Path(p['source_root']).resolve() == source and
                p['source_experiment_signature'] == native['experiment_signature'], 'SimpleMem attribution is not fresh/fixedQwen42')
        require(lock['status'] == 'sealed_before_scoring' and score['status'] == 'complete' and
                score['questions'] == 371 and score['gold_round_references'] == 1041 and
                score['source_experiment_signature'] == native['experiment_signature'] and
                score['annotation_lock_sha256'] == self.sha(folder / 'annotation_lock.json') and
                score['responses_sha256'] == self.sha(source / 'qa_responses.jsonl') and
                same_number(score['recall_at5_micro'], score['gold_round_hits'] / 1041),
                'SimpleMem attribution requires sealed complete fresh scoring')
        self.hashes(p['input_hashes']); self.hashes(lock['input_hashes'])
        require(lock['entries'] == p['annotation_entries'] == score['annotations']['entries'] and
                lock['selected_entries'] == p['selected_entries'] == score['annotations']['selected_entries'] and
                lock['manual_review_entries'] == score['annotations']['manual_review_entries'], 'Attribution coverage/seal counts differ')
        if self.mcq:
            item = next(m for m in self.mcq['methods'] if m['method'] == 'simplemem')
            require(item['recall_numerator'] == score['gold_round_hits'] and item['recall_denominator'] == 1041,
                    'Independent SimpleMem recall differs from sealed score')

    def workbook(self):
        from openpyxl import load_workbook
        report = self.read(self.root / 'workbook_audit.json')
        data = self.read(self.root / 'comparison.json')
        require(report['status'] == data['status'] == 'complete' and report['passed'] is True and
                report['methods'] == 10 and report['mcq_records'] == report['open_answers'] == report['judge_scores'] == 3710,
                'Final workbook/JSON not complete')
        self.hashes(report['audit_source_sha256'])
        require(report['implementation_sha256'] == self.sha(ROOT / 'scripts/analysis/export_memeye_deepseek_seed1024_final.py'),
                'Workbook exporter changed after export')
        path = Path(report['path']).resolve()
        require(path.is_relative_to(self.root) and path.suffix == '.xlsx' and 'DRAFT' not in path.stem.upper() and
                self.sha(path) == report['sha256'], 'Final XLSX missing/changed/draft')
        items = data['methods']
        require(len(items) == 10 and {i['method'] for i in items} == set(METHODS) and
                all(i['status'] == 'complete' for i in items), 'Comparison JSON is incomplete')
        csv_path = self.root / 'comparison.csv'; self.sha(csv_path)
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        require(len(csv_rows) == 10 and {i['method'] for i in csv_rows} == set(METHODS), 'Comparison CSV coverage differs')
        csv_by_method = {i['method']: i for i in csv_rows}
        for item in items:
            for key, value in item.items():
                serialized = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else '' if value is None else str(value)
                require(csv_by_method[item['method']][key] == serialized, 'Comparison CSV/JSON values differ: ' + key)
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            require(book.sheetnames == report['sheets'] and all(book[s].max_row == 11 for s in
                ('Seed1024', 'Seed42_vs_1024', 'Paired_MCQ', 'Coverage_and_native_outputs')), 'Workbook sheet coverage differs')
            for row, item in enumerate(items, 2):
                require(book['Seed1024'].cell(row, 1).value == item['label'] and item['questions'] == 371,
                        'Workbook row identity differs')
                for col, key in ((3, 'seed1024_accuracy'), (4, 'seed1024_recall_at5_micro'), (5, 'seed1024_llm_judge'),
                                 (6, 'seed1024_storage_mib'), (7, 'seed1024_correct'), (8, 'questions'),
                                 (9, 'seed1024_context_source_micro_recall')):
                    require(same_number(book['Seed1024'].cell(row, col).value, item[key]), 'Workbook/JSON numeric value differs: ' + key)
                for col, key in ((2, 'both_correct'), (3, 'seed42_only_correct'), (4, 'seed1024_only_correct'),
                    (5, 'both_wrong'), (6, 'changed_predicted_answers'), (9, 'changed_native_predictions')):
                    require(book['Paired_MCQ'].cell(row, col).value == item[key], 'Workbook paired values differ')
                comparison_keys = ('seed42_accuracy', 'seed1024_accuracy', 'accuracy_delta_pp', 'seed42_correct',
                    'seed1024_correct', 'seed42_recall_at5_micro', 'seed1024_recall_at5_micro', 'seed42_llm_judge',
                    'seed1024_llm_judge', 'seed42_storage_mib', 'seed1024_storage_mib', 'two_seed_accuracy_mean',
                    'two_seed_accuracy_sample_sd', 'seed42_context_source_micro_recall', 'seed1024_context_source_micro_recall')
                for col, key in enumerate(comparison_keys, 2):
                    require(same_number(book['Seed42_vs_1024'].cell(row, col).value, item[key]),
                            'Workbook comparison values differ: ' + key)
                require(same_number(item['two_seed_accuracy_mean'], (item['seed42_accuracy'] + item['seed1024_accuracy']) / 2) and
                        same_number(item['two_seed_accuracy_sample_sd'], abs(item['seed1024_accuracy'] - item['seed42_accuracy']) / math.sqrt(2)),
                        'Two-seed summary formula differs')
                if self.mcq and self.opened and self.metrics and self.paired:
                    method = item['method']; mcq = next(m for m in self.mcq['methods'] if m['method'] == method)
                    opened = next(m for m in self.opened['methods'] if m['method'] == method)
                    metric = next(m for m in self.metrics['methods'] if m['method'] == method)
                    pair = self.paired['methods'][method]
                    require(item['seed1024_correct'] == mcq['correct'] == metric['correct'] and
                            item['seed1024_storage_bytes'] == mcq['storage_bytes'] == metric['storage_bytes'] and
                            same_number(item['seed1024_llm_judge'], opened['judge']['llm_judge']) and
                            item['seed42_correct'] == pair['seed42_correct'], 'Workbook disagrees with final evidence: ' + method)
            for values in book['Artifacts'].iter_rows(min_row=2, values_only=True):
                if values[2]:
                    require(self.sha(values[2]) == values[3], 'Workbook cited artifact changed: ' + values[2])
            if self.metrics:
                labels = {i['method']: i['label'] for i in items}
                expected = {(labels[m['method']], c['path'], c['bytes'], c['sha256'])
                            for m in self.metrics['methods'] for c in m['storage_components']}
                stored = list(book['Storage_files'].iter_rows(min_row=2, values_only=True))
                require(len(stored) == len(expected) and set(stored) == expected,
                        'Workbook persisted storage component rows differ from audited actual bytes')
        finally:
            book.close()

    def recovery(self):
        # Preserve every earlier failed attempt/snapshot and current CPU recovery record.
        archive = self.root / 'recovery/simplemem_first_failure_20261106T063907Z'
        first = self.read(archive / 'recovery_manifest.json')
        require(first['passed'] is True and first['file_count'] == len(first['files']) and first['api_calls_by_archive'] == 0,
                'Original SimpleMem failure snapshot absent/unverified')
        for item in first['files']:
            path = archive / item['archive_relative']
            require(path.stat().st_size == item['bytes'] and self.sha(path) == item['sha256'], 'Historical failure snapshot changed')
        diagnostic = self.root / 'metrics/recovery_diagnostics'
        evidence = list(diagnostic.glob('*_open_failure_*/recovery_evidence.json'))
        require(len(evidence) == 4, 'Require explicit CPU recovery evidence for all four text Open tails')
        helpers = list((ROOT / 'scripts/evaluation').glob('recover_memeye_deepseek1024_*_open.py'))
        helpers += list((ROOT / 'scripts/evaluation').glob('run_memeye_deepseek1024_text_open_recovery.py'))
        require(helpers, 'CPU Open recovery helper implementation missing')
        helper_hashes = {self.sha(path) for path in helpers}
        methods = set()
        for path in evidence:
            item = self.read(path); methods.add(item['method'])
            require(item['status'] == 'cpu_recovered' and item['questions'] == 371 and
                    item['actual_query_derivation_passed_all_371'] is True and
                    item['native_task_global_retrieval_and_context_hashes_passed'] is True and
                    item['frozen_protocol_and_question_bytes_preserved'] is True and item['source_hashes_passed'] is True and
                    item['no_native_retrieval_or_generation_rerun'] is True and item['api_calls'] == item['gpu_model_loads'] == 0,
                    'Text native Open recovery exceeds approved derived-field boundary')
            expected_ids = {'Social_Chat_Memory_Test/Q' + str(i) for i in (1, 2, 4, 8, 24, 25, 37, 38)}
            require(item['different_frozen_fields'] == {'retrieval_query': 8} and item['unchanged_other_records'] == 363 and
                    {m['id'] for m in item['mismatches']} == expected_ids and
                    all(m['fields'] == ['retrieval_query'] for m in item['mismatches']),
                    'Recovery altered fields outside the exact eight approved derived-query comparisons')
            self.hashes(item['preserved_input_sha256']); self.hashes(item['output_sha256'])
            require(item['native_marker'] == self.read(self.root / 'open/text_methods' / item['method'] / 'complete.json'),
                    'Recovery marker no longer matches the actual adopted native completion')
            require(item['adapter_sha256'] in helper_hashes and
                    item['source_code_sha256'] == self.sha(ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_text_methods.py'),
                    'Recovery helper or original source changed')
            self.sha(path.parent / 'adapted_native_tail.py')
            if item.get('base_recovery_adapter_sha256'):
                require(item['base_recovery_adapter_sha256'] in helper_hashes, 'Recovery base helper changed')
        require(methods == {'gen_agents', 'memoryos', 'a_mem', 'simplemem'}, 'Recovery method coverage differs')
        # Include helper implementations as well as all historical parent-progress snapshots.
        for folder in (self.root / 'recovery', diagnostic):
            for path in folder.rglob('*'):
                if path.is_file():
                    self.sha(path)
        for path in (self.root / 'parallel').glob('*'):
            if path.is_file() and path.suffix in ('.json', '.log'):
                self.sha(path)
        for path in self.root.glob('*progress*.json'):
            if path.name != 'progress.json':
                self.sha(path)

    def check(self):
        self.sha(__file__)
        for name, method in (('generation_exit_and_source_hashes', self.generation), ('all47_actual_stages', self.stages),
            ('independent_all10_audits', self.independent_audits), ('all10_metric_index', self.metrics_index),
            ('full_paired_mcq', self.paired_mcq), ('fresh_simplemem_seal_and_score', self.attribution),
            ('final_workbook_json_csv', self.workbook), ('preserved_recovery_evidence', self.recovery)):
            self.gate(name, method)
        for path, expected in self.stats.items():
            try:
                stat = Path(path).stat()
                require((stat.st_size, stat.st_mtime_ns) == expected, 'Evidence changed before preflight finished: ' + path)
            except (ValueError, OSError) as error:
                self.issues.append(dict(gate='stable_evidence_snapshot', error=str(error)))
        return dict(status='ready' if not self.issues else 'not_ready', passed=not self.issues, dry_run=True,
            root=str(self.root), expected_stages=47, expected_methods=10, expected_questions_per_method=371,
            checked_at_utc=datetime.now(timezone.utc).isoformat(), gates=self.gates, issues=self.issues,
            evidence_files=len(self.files), evidence_sha256=self.files)


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def apply(root):
    root = Path(root).resolve()
    # Apply is the sole mutating path. Hold the same dispatcher lock across recheck and writes.
    with (root / 'suite.lock').open('r+') as suite:
        fcntl.flock(suite, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inspector = Inspector(root, suite_locked=True); report = inspector.check()
        require(report['passed'], 'Finalization refused: ' + json.dumps(report['issues'], ensure_ascii=False))
        require(not (root / 'completion_manifest.json').exists(), 'Completion manifest already exists; refusing overwrite')
        original = (root / 'progress.json').read_bytes()
        require(hashlib.sha256(original).hexdigest() == inspector.files[str(root / 'progress.json')], 'Progress changed after preflight')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = root / 'audit' / ('progress_before_finalization_' + stamp + '.json')
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open('xb') as handle:
            handle.write(original)
        evidence = dict(report['evidence_sha256']); evidence[str(backup)] = hashlib.sha256(original).hexdigest()
        # Current progress is intentionally replaced; its original bytes remain in the backup.
        evidence.pop(str(root / 'progress.json'))
        completion = dict(status='complete', passed=True, model='deepseek-v4.1-flash', seed=1024,
            caption_seed=42, judge_seed=42, stages=47, methods=10, mcq_records=3710, open_answers=3710, judge_scores=3710,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), generation_progress_backup=str(backup),
            generation_progress_sha256=hashlib.sha256(original).hexdigest(), generation_pid=inspector.progress['pid'],
            generation_status_before='generation_complete_pending_final_audit', gates=report['gates'],
            evidence_sha256=evidence, implementation_sha256=inspector.files[str(Path(__file__).resolve())],
            policy='All original stages genuinely complete; original dispatcher exited; current complete independent audits, fresh sealed attribution and consistent final exports. Historical failures and CPU recovery evidence retained.')
        manifest = root / 'completion_manifest.json'; atomic_json(manifest, completion)
        state = dict(inspector.progress, status='complete', current_stage=None, unfinished_stages=[],
            completed_at_utc=completion['completed_at_utc'], completion_manifest=str(manifest),
            completion_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            generation_progress_backup=str(backup), generation_progress_sha256=completion['generation_progress_sha256'])
        require(state['stages'] == inspector.progress['stages'], 'Finalizer must not change any stage state')
        atomic_json(root / 'progress.json', state)
        return dict(status='complete', passed=True, completion_manifest=str(manifest),
                    completion_manifest_sha256=state['completion_manifest_sha256'], evidence_files=len(evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        report = apply(args.root)
    else:
        report = Inspector(args.root).check()
    # Large SHA maps go into the completion manifest only on apply; drycheck prints concise findings.
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence_sha256'}, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
