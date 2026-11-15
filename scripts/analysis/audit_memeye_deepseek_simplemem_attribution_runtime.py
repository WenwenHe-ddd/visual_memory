"""CPU-only, gold-blind audit of fresh SimpleMem attribution runtime artifacts.

No model imports, API calls, evaluation questions, gold or score files. Native
pure parsing/collection/review functions are extracted from their hash-pinned
source AST and run with reads restricted to audited snapshots and writes held
in memory. Only the requested audit report is written.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
DEFAULT = RUN / 'metrics/simplemem_source_attribution'
NATIVE = ROOT / 'scripts/analysis/simplemem_source_attribution.py'
REVIEW = ROOT / 'scripts/analysis/simplemem_attribution_recall.py'
ENGINE = ROOT / 'scripts/memory/reflexion_memory.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def forbidden(path):
    name = Path(path).name.lower()
    return (name in {'qa_responses.jsonl', 'qa_manifest.jsonl', 'judge_inputs.jsonl', 'backbone_audit.json',
                     'gold.jsonl', 'questions.jsonl', 'reader_manifest.jsonl', 'metric_summary.json'} or
            name.startswith(('recall_summary', 'recall_per_question')) or 'gold' in name or
            name.endswith(('_mcq.jsonl', '_mcq_records.jsonl')))


def extract(path, names, namespace):
    """Execute only named pure functions; never import the model-bearing module."""
    tree = ast.parse(Path(path).read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in body} == set(names), 'Native pure function coverage changed')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}; self.cache = {}; self.stamps = {}; self.skipped = {}
        self.checks = Counter(); self.pending = []; self.errors = []

    def data(self, path):
        path = Path(path).resolve(); key = str(path)
        require(not forbidden(path), 'Gold-blind boundary forbids opening: ' + key)
        if key not in self.cache:
            before = path.stat(); value = path.read_bytes(); after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'Artifact changed while being read: ' + key)
            self.files[key] = hashlib.sha256(value).hexdigest(); self.cache[key] = value
            self.stamps[key] = (after.st_size, after.st_mtime_ns)
        return self.cache[key]

    def read(self, path):
        return json.loads(self.data(path))

    def rows(self, path):
        return [json.loads(line) for line in self.data(path).decode().splitlines() if line.strip()]

    def sha(self, path):
        self.data(path)
        return self.files[str(Path(path).resolve())]

    def hashes(self, values):
        for path, expected in values.items():
            if forbidden(path):
                self.skipped[str(path)] = dict(recorded_sha256=expected,
                    reason='Evaluation QA/score-bearing artifact deliberately not opened by this gold-blind audit')
                continue
            require(self.sha(path) == expected, 'Pinned input changed: ' + str(path))
            self.checks['pinned_allowed_input_hash'] += 1

    def pure_namespace(self):
        for path in (NATIVE, REVIEW, ENGINE, Path(__file__)):
            self.sha(path)
        tree = ast.parse(self.data(NATIVE).decode())
        rules = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'RULES' for t in n.targets))
        namespace = dict(json=json, unicodedata=unicodedata, Counter=Counter, defaultdict=defaultdict,
            RULES=rules, print=lambda *args, **kwargs: None)
        extract(NATIVE, ('normalize', 'messages', 'parse_and_validate'), namespace)
        extract(REVIEW, ('duplicate_evidence_sources',), namespace)
        # Confirm the pinned actual LocalQwen code contains its declared fixed-seed/greedy path.
        engine = ast.parse(self.data(ENGINE).decode())
        cls = next(n for n in engine.body if isinstance(n, ast.ClassDef) and n.name == 'LocalQwen')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        generation = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        seeds = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'manual_seed']
        require(len(seeds) == 1 and ast.unparse(seeds[0].args[0]) == "protocol['seed']", 'Native engine seed binding changed')
        calls = [n for n in ast.walk(generation) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'generate']
        require(len(calls) == 1 and any(k.arg == 'do_sample' and isinstance(k.value, ast.Constant)
                and k.value.value is False for k in calls[0].keywords), 'Native engine greedy generation changed')
        self.checks['native_engine_seed_and_greedy_code_binding'] += 1
        return namespace

    def rebuild_jobs(self, source, native, entry_index):
        """Independently recover window order/aliases from construction records.

        Selection flags come from the sealed pre-generation entry_index; this
        audit deliberately does not reopen evaluation QA to choose entries.
        """
        observations = self.read(source / 'observations.json')
        selected = defaultdict(set)
        for identity, item in entry_index.items():
            if item['selected']:
                selected[item['task']].add(identity)
        jobs, rebuilt_index = [], {}
        for task in native['tasks']:
            memories = self.read(source / task / 'final_memory.json')
            entries = {e['entry_id']: e for e in memories}
            require(len(entries) == len(memories), 'Duplicate native memory IDs')
            windows = self.read(source / task / 'runtime/construction_windows.json')
            dialogue = self.read(source / task / 'runtime/dialogue_sources.json')
            rounds = {r['id']: r for r in observations[task]}
            previous, dependencies = [], {}
            for window in windows:
                for identity in window['entry_ids']:
                    dependencies[identity] = list(previous)
                if window['entry_ids']:
                    previous = window['entry_ids'][:3]
            required, pending = set(selected[task]), list(selected[task])
            while pending:
                for dep in dependencies[pending.pop()]:
                    if dep not in required:
                        required.add(dep); pending.append(dep)
            for wi, window in enumerate(windows):
                current = [identity for identity in window['entry_ids'] if identity in required]
                if not current:
                    continue
                present = defaultdict(set)
                for did in window['dialogue_ids']:
                    origin = dialogue[str(did)]; present[origin['round_id']].add(origin['speaker'])
                sources = []
                for ri, rid in enumerate(window['source_round_ids'], 1):
                    row = rounds[rid]; user_present = row['speaker'] in present[rid]
                    sources.append(dict(alias=f'R{ri}', kind='round', round_id=task + '/' + rid,
                        timestamp=row['timestamp'], speaker=row['speaker'], user=row['user'] if user_present else '',
                        caption='\n'.join(row.get('raw', {}).get('image_caption', [])) if user_present else '',
                        assistant=row['assistant'] if 'assistant' in present[rid] else ''))
                for pi, identity in enumerate(dependencies[current[0]], 1):
                    sources.append(dict(alias=f'P{pi}', kind='prior_entry', entry_id=identity,
                                        memory=entries[identity]['lossless_restatement']))
                for start in range(0, len(current), 8):
                    key = f'{task}__window{wi:02d}__group{start // 8:02d}'
                    targets = [dict(alias=f'M{i}', entry_id=identity,
                        memory=entries[identity]['lossless_restatement'], selected=identity in selected[task])
                        for i, identity in enumerate(current[start:start + 8], 1)]
                    jobs.append(dict(id=key, task=task, window_index=wi, sources=sources, targets=targets))
                    for target in targets:
                        require(target['entry_id'] not in rebuilt_index, 'Duplicate attribution target')
                        rebuilt_index[target['entry_id']] = dict(task=task, job_id=key, **target)
        require(rebuilt_index == entry_index, 'Entry index differs from exact fresh construction/order/dependency closure')
        self.checks['native_job_source_target_alias_order_rebuilt'] = len(jobs)
        return jobs

    def inspect_result(self, job, index, p, result, call, pure):
        expected = dict(messages=pure['messages'](job, index), max_new_tokens=4096,
                        do_sample=False, signature=p['signature'], **{'pass': index})
        require(result['job_id'] == job['id'] and result['pass'] == index and
                result['signature'] == p['signature'] and result['call_key'] == digest(expected) and
                call['request'] == expected and call['output'] == result['output'],
                'Native pass/job/request/signature/raw-output identity differs: ' + job['id'])
        payload = json.loads(expected['messages'][1]['content'])
        original_sources = [{k: v for k, v in s.items() if k not in ('round_id', 'entry_id', 'kind')} for s in job['sources']]
        original_targets = [dict(id=t['alias'], statement=t['memory']) for t in job['targets']]
        if index == 2:
            original_sources.reverse(); original_targets.reverse()
        require(payload == dict(original_sources=original_sources, target_memories=original_targets),
                'Native source/target reversal must preserve aliases and source text')
        out = result['output']
        require(set(out) == {'raw_response', 'input_tokens', 'generated_tokens', 'hit_token_limit'} and
                type(out['input_tokens']) is int and type(out['generated_tokens']) is int and
                0 < out['generated_tokens'] <= 4096 and out['input_tokens'] > 0 and
                out['input_tokens'] + 4096 <= p['context_budget'] and type(out['hit_token_limit']) is bool and
                (not out['hit_token_limit'] or out['generated_tokens'] == 4096), 'Native output token metadata differs')
        derived = {}
        try:
            if out['hit_token_limit']:
                raise ValueError('annotation output truncated')
            derived['annotations'], derived['quote_issues'] = pure['parse_and_validate'](out['raw_response'], job)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            derived['parse_error'] = str(error)
        saved = {k: result[k] for k in ('annotations', 'quote_issues', 'parse_error') if k in result}
        require(derived == saved and set(result) == {'job_id', 'call_key', 'output', 'pass', 'signature'} | set(derived),
                'Saved annotations/quote issues/parse error differ from native raw reparse: ' + job['id'])
        self.checks['raw_pass_request_output_and_reparse'] += 1
        return dict(job_id=job['id'], **{'pass': index}, call_key=result['call_key'],
            parse_error=derived.get('parse_error'), quote_issue_targets=len(derived.get('quote_issues', {})),
            token_limit=out['hit_token_limit'], generated_tokens=out['generated_tokens'], input_tokens=out['input_tokens'])

    def derivations(self, p, jobs, results, pure):
        captured = {}
        def read(path):
            path = Path(path)
            if path == self.root / 'protocol.json':
                return p
            for index in (1, 2):
                if path.parent == self.root / f'pass{index}':
                    return results[(path.stem, index)]
            raise ValueError('Unexpected native derivation read: ' + str(path))
        def read_rows(path):
            if Path(path) == self.root / 'jobs.jsonl':
                return jobs
            require(str(path) in captured, 'Native derivation attempted an unapproved input read')
            return captured[str(path)]['value']
        def write_rows(path, value):
            captured[str(path)] = dict(value=value,
                data=''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in value).encode())
        def write_json(path, value):
            captured[str(path)] = dict(value=value, data=(json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())
        def sha(path):
            require(str(path) in captured, 'Unexpected native derivation hash read')
            return hashlib.sha256(captured[str(path)]['data']).hexdigest()
        namespace = dict(pure, read=read, read_rows=read_rows, write_rows=write_rows,
            write_json=write_json, sha=sha, verify_inputs=lambda *args: None)
        extract(NATIVE, ('collect',), namespace)
        extract(REVIEW, ('review',), namespace)
        namespace['collect'](self.root); namespace['review'](self.root)
        expected_files = ('annotations.jsonl', 'needs_review.jsonl', 'review_queue.jsonl', 'review_status.json')
        for name in expected_files:
            path = self.root / name
            if not path.exists():
                self.pending.append('Native derivation not yet persisted: ' + name)
                continue
            expected = captured[str(path)]['data']
            require(self.data(path) == expected, 'Native collect/review derived artifact differs byte-for-byte: ' + name)
            self.checks['native_collect_review_exact_derived_file'] += 1
        annotations = captured[str(self.root / 'annotations.jsonl')]['value']
        queue = captured[str(self.root / 'review_queue.jsonl')]['value']
        require(len(annotations) == len({r['entry_id'] for r in annotations}) == p['annotation_entries'] == 662 and
                sum(r['selected'] for r in annotations) == p['selected_entries'] == 607, 'Native derived annotation coverage differs')
        return dict(annotations=len(annotations), selected_entries=p['selected_entries'], review_entries=len(queue),
            review_reasons=dict(Counter(reason for r in queue for reason in r['review_reasons'])),
            expected_derived_sha256={name: hashlib.sha256(captured[str(self.root / name)]['data']).hexdigest() for name in expected_files})

    def run(self):
        report = dict(schema='memeye-simplemem-attribution-runtime-audit-v1', status='pending', passed=False,
            gold_blind=True, api_calls=0, gpu_model_loads=0,
            root=str(self.root), expected_jobs_per_pass=106, expected_calls=212,
            read_boundary='Native source memories/construction records, jobs/index/protocol, local raw calls/passes and gold-blind derived annotations only; no evaluation QA, gold or score is opened.')
        try:
            p = self.read(self.root / 'protocol.json'); pure = self.pure_namespace()
            require(digest({k: v for k, v in p.items() if k != 'signature'}) == p['signature'], 'Attribution protocol signature changed')
            require(p['seed'] == 42 and p['memory_generation_seed'] == 1024 and p['fresh_source_only'] is True and
                    p['gold_blind'] is True and p['rules'] == pure['RULES'] and p['max_new_tokens'] == 4096 and
                    p['generation_batch_size'] == 8 and p['max_padded_batch_tokens'] == 131072, 'Evaluator seed/native settings changed')
            self.hashes(p['input_hashes'])
            source = Path(p['source_root']).resolve()
            require(source == RUN / 'text_methods/simplemem', 'Attribution is not bound to fresh seed1024 source')
            native = self.read(source / 'protocol.json')
            require(native['seed'] == 1024 and native['mode'] == 'simplemem' and native['model'] == 'deepseek-v4.1-flash' and
                    digest({k: v for k, v in native.items() if k != 'experiment_signature'}) == native['experiment_signature'] and
                    p['source_experiment_signature'] == native['experiment_signature'], 'Native memory1024 signature/model binding changed')
            evaluator = self.read(Path(p['evaluator_control_root']) / 'protocol.json')
            require(all(p[k] == evaluator[k] for k in ('reader_model', 'reader_model_path', 'seed', 'dtype', 'context_budget')),
                    'Fixed Qwen42 evaluator control changed')
            environment = self.read(self.root / 'generation_environment.json')
            require(environment['device_map'] == p['reader_device_map'] and environment['input_device'] == 'cuda:1',
                    'Saved local runtime device map differs from this fixed protocol')
            jobs = self.rows(self.root / 'jobs.jsonl'); entry_index = self.read(self.root / 'entry_index.json')
            require(self.sha(self.root / 'jobs.jsonl') == p['jobs_sha256'] and
                    self.sha(self.root / 'entry_index.json') == p['entry_index_sha256'], 'Jobs/index input hashes differ')
            require(len(jobs) == len({j['id'] for j in jobs}) == 106 and
                    all(set(j) == {'id', 'task', 'window_index', 'sources', 'targets'} for j in jobs), 'Job coverage or gold-blind field schema differs')
            require(jobs == self.rebuild_jobs(source, native, entry_index), 'Job/source order or exact window aliases differ')
            self.checks['protocol_seed42_and_fresh_memory_seed1024_binding'] += 1
            results, details, expected_keys = {}, [], set()
            pass_counts = {}
            for index in (1, 2):
                paths = {path.stem: path for path in (self.root / f'pass{index}').glob('*.json')}
                require(set(paths) <= {j['id'] for j in jobs}, 'Unexpected local pass job file')
                pass_counts[str(index)] = len(paths)
                for job in jobs:
                    if job['id'] not in paths:
                        continue
                    result = self.read(paths[job['id']]); key = result['call_key']
                    call = self.read(self.root / 'calls' / (key + '.json'))
                    require(digest(call['request']) == key, 'Raw native call filename hash differs')
                    details.append(self.inspect_result(job, index, p, result, call, pure))
                    results[(job['id'], index)] = result; expected_keys.add(key)
                if len(paths) != 106:
                    self.pending.append(f'Pass{index} completed {len(paths)}/106 jobs')
            report.update(pass_jobs_verified=pass_counts, calls_verified=len(details), call_details=details,
                parse_error_calls=sum(bool(d['parse_error']) for d in details),
                quote_issue_targets=sum(d['quote_issue_targets'] for d in details),
                token_limited_calls=sum(d['token_limit'] for d in details), runtime_passed=len(details) == 212)
            if len(details) == 212:
                actual_keys = {path.stem for path in (self.root / 'calls').glob('*.json')}
                require(actual_keys == expected_keys and len(expected_keys) == 212, 'Native raw call set is not exactly106×2')
                report.update(self.derivations(p, jobs, results, pure))
                report['derived_artifacts_passed'] = self.checks['native_collect_review_exact_derived_file'] == 4
                lock_path = self.root / 'annotation_lock.json'
                if not lock_path.exists():
                    self.pending.append('Gold-blind manual annotation lock not yet sealed')
                    report['annotation_lock_passed'] = False
                else:
                    lock = self.read(lock_path)
                    require(lock['status'] == 'sealed_before_scoring' and lock['entries'] == 662 and lock['selected_entries'] == 607,
                            'Annotation seal does not bind all fresh entries')
                    self.hashes(lock['input_hashes'])
                    required = {str((self.root / name).resolve()) for name in
                        ('protocol.json', 'jobs.jsonl', 'annotations.jsonl', 'review_queue.jsonl', 'frozen_source_annotations.jsonl')}
                    required |= {str((self.root / 'calls' / (key + '.json')).resolve()) for key in expected_keys}
                    required |= {str((self.root / f'pass{index}' / (job['id'] + '.json')).resolve()) for index in (1, 2) for job in jobs}
                    require(required <= set(lock['input_hashes']), 'Annotation lock omits raw calls/passes or derived annotations')
                    report['annotation_lock_passed'] = True
                    report['annotation_lock_sha256'] = self.sha(lock_path)
            for path, stamp in self.stamps.items():
                stat = Path(path).stat()
                require((stat.st_size, stat.st_mtime_ns) == stamp, 'Audited snapshot changed during CPU verification: ' + path)
            report['status'] = 'pending' if self.pending else 'passed'; report['passed'] = not self.pending
        except Exception as error:
            self.errors.append(dict(type=type(error).__name__, message=str(error)))
            report.update(status='failed', passed=False)
        report.update(audited_at_utc=datetime.now(timezone.utc).isoformat(), pending=self.pending, errors=self.errors,
            checks=dict(self.checks), source_sha256={str(path): self.files.get(str(path)) for path in (Path(__file__).resolve(), NATIVE, REVIEW, ENGINE)},
            artifact_sha256=self.files, deliberately_unopened_input_hashes=self.skipped,
            limitations='Logical raw-request/runtime-source provenance is checked. No separate GPU execution trace exists. Selection membership is bound to pre-generation entry_index flags; evaluation QA selection and final recall scoring are intentionally outside this gold-blind audit.')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--report', type=Path, default=RUN / 'parallel/simplemem_attribution_runtime_audit.json')
    args = parser.parse_args()
    require(not forbidden(args.report), 'Refusing a question/gold/score report destination')
    report = Audit(args.root).run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('artifact_sha256', 'call_details', 'deliberately_unopened_input_hashes')}))
    return 0 if report['passed'] else 2 if report['status'] == 'pending' else 1


if __name__ == '__main__':
    raise SystemExit(main())
