"""Synthetic CPU-only tests for native runtime reparse and collection gates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SOURCE = Path(__file__).with_name('audit_memeye_deepseek_simplemem_attribution_runtime.py')
spec = importlib.util.spec_from_file_location('runtime_audit_fixture', SOURCE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class RuntimeAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='attribution-runtime-fixture-')
        self.root = Path(self.temp.name); self.audit = module.Audit(self.root)
        self.pure = self.audit.pure_namespace()
        self.quote = 'The bright orange bird landed beside the old wooden fence.'
        self.job = dict(id='fixture', task='fixture_task', window_index=0, sources=[
            dict(alias='R1', kind='round', round_id='fixture/R1', timestamp='today', speaker='user',
                 user=self.quote, assistant='', caption=''),
            dict(alias='R2', kind='round', round_id='fixture/R2', timestamp='later', speaker='user',
                 user=self.quote, assistant='', caption=''),
            dict(alias='P1', kind='prior_entry', entry_id='prior', memory=self.quote)],
            targets=[dict(alias='M1', entry_id='fixture0', memory='A bright orange bird beside a fence.', selected=True)])
        self.protocol = dict(signature='fixture-signature', context_budget=128000,
                             annotation_entries=662, selected_entries=607)

    def tearDown(self):
        self.temp.cleanup()

    def result(self, job, index, cap=False):
        values = [dict(id=t['alias'], status='resolved', evidence=[dict(source='P1', field='memory', quote=self.quote)],
                       ambiguous_sources=[], note='') for t in job['targets']]
        output = dict(raw_response=json.dumps(values), input_tokens=500, generated_tokens=4096 if cap else 100,
                      hit_token_limit=cap)
        request = dict(messages=self.pure['messages'](job, index), max_new_tokens=4096, do_sample=False,
                       signature=self.protocol['signature'], **{'pass': index})
        result = dict(job_id=job['id'], call_key=module.digest(request), output=output,
                      signature=self.protocol['signature'], **{'pass': index})
        if cap:
            result['parse_error'] = 'annotation output truncated'
        else:
            result['annotations'], result['quote_issues'] = self.pure['parse_and_validate'](output['raw_response'], job)
        return result, dict(request=request, output=output)

    def test_original_forward_reverse_requests_and_raw_reparse(self):
        for index in (1, 2):
            result, call = self.result(self.job, index)
            self.audit.inspect_result(self.job, index, self.protocol, result, call, self.pure)
        self.assertEqual(self.audit.checks['raw_pass_request_output_and_reparse'], 2)

    def test_pass2_alias_renumbering_or_request_change_rejected(self):
        result, call = self.result(self.job, 2)
        payload = json.loads(call['request']['messages'][1]['content'])
        payload['original_sources'][0]['alias'] = 'R1'
        call['request']['messages'][1]['content'] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, 'request/signature/raw-output identity'):
            self.audit.inspect_result(self.job, 2, self.protocol, result, call, self.pure)

    def test_saved_annotation_mutation_rejected_after_raw_reparse(self):
        result, call = self.result(self.job, 1)
        result['annotations']['M1']['note'] = 'not in original response'
        with self.assertRaisesRegex(ValueError, 'raw reparse'):
            self.audit.inspect_result(self.job, 1, self.protocol, result, call, self.pure)

    def test_truncated_valid_json_keeps_native_parse_error(self):
        result, call = self.result(self.job, 1, cap=True)
        self.audit.inspect_result(self.job, 1, self.protocol, result, call, self.pure)
        result['annotations'] = {}
        with self.assertRaisesRegex(ValueError, 'raw reparse'):
            self.audit.inspect_result(self.job, 1, self.protocol, result, call, self.pure)

    def test_forbidden_evaluation_files_cannot_be_opened(self):
        for name in ('qa_responses.jsonl', 'questions.jsonl', 'gold.jsonl', 'recall_summary.json'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'boundary forbids opening'):
                self.audit.data(self.root / name)

    def test_all662_derived_annotations_and_duplicate_prior_review_queue(self):
        jobs, results, ordinal = [], {}, 0
        for number in range(106):
            job = copy.deepcopy(self.job); job['id'] = f'job{number:03d}'; job['targets'] = []
            for alias in range(1, (7 if number < 26 else 6) + 1):
                job['targets'].append(dict(alias=f'M{alias}', entry_id=f'entry{ordinal:03d}',
                    memory='A bright orange bird beside a fence.', selected=ordinal < 607))
                ordinal += 1
            jobs.append(job)
            for index in (1, 2):
                results[(job['id'], index)] = self.result(job, index)[0]
        self.assertEqual(ordinal, 662)
        # Persist only synthetic fixtures through the original pure derivations.
        def read(path):
            path = Path(path)
            return self.protocol if path.name == 'protocol.json' else results[(path.stem, int(path.parent.name[-1]))]
        def rows(path):
            return jobs if Path(path).name == 'jobs.jsonl' else [json.loads(s) for s in Path(path).read_text().splitlines()]
        def write_rows(path, values):
            Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))
        def write_json(path, value):
            Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        namespace = dict(self.pure, read=read, read_rows=rows, write_rows=write_rows, write_json=write_json,
            verify_inputs=lambda *args: None, sha=lambda p: module.hashlib.sha256(Path(p).read_bytes()).hexdigest())
        module.extract(module.NATIVE, ('collect',), namespace)
        module.extract(module.REVIEW, ('review',), namespace)
        namespace['collect'](self.root); namespace['review'](self.root)
        summary = self.audit.derivations(self.protocol, jobs, results, self.pure)
        self.assertEqual(summary['annotations'], 662); self.assertEqual(summary['review_entries'], 662)
        self.assertEqual(summary['review_reasons'], {'quoted_content_has_multiple_possible_sources': 662,
                                                     'uses_previous_window_memory': 662})
        self.assertEqual(self.audit.checks['native_collect_review_exact_derived_file'], 4)
        self.assertEqual(self.audit.pending, [])
        # Byte-level mutation is rejected even if JSON rows would parse identically.
        with (self.root / 'annotations.jsonl').open('a') as handle:
            handle.write('\n')
        fresh = module.Audit(self.root)
        with self.assertRaisesRegex(ValueError, 'byte-for-byte'):
            fresh.derivations(self.protocol, jobs, results, fresh.pure_namespace())


if __name__ == '__main__':
    unittest.main(verbosity=2)
