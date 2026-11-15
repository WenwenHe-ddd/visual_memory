"""Offline cache-integrity tests; never call a model or modify historical runs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_open_method_judge as judge


def synthetic_cache(root, grade=1):
    """All 371 IDs share one wire request; summary counts still use 371 rows."""
    root.mkdir()
    canonical = {f'Synthetic/Q{i}': {'id': f'Synthetic/Q{i}', 'task': 'Synthetic',
        'point': 'X1/Y1', 'question': 'Where does the person live?', 'ground_truth': 'Paris'}
        for i in range(371)}
    input_rows = [{**q, 'model_output': 'Paris', 'source_record_id': qid,
                   'answer_model': judge.base.ANSWER_MODEL, 'em': 1}
                  for qid, q in canonical.items()]
    request = judge.shared.build_request(input_rows[0], judge.shared.PROMPT.read_text())
    digest = judge.transport.request_sha256(request)
    requests = [{'request_sha256': digest, 'request': request,
        **{key: input_rows[0][key] for key in ('question', 'ground_truth', 'model_output', 'point')},
        'question_ids': list(canonical)}]
    mapped = [{**row, 'request_sha256': digest} for row in input_rows]
    score = {'request_sha256': digest, 'returned_model': judge.transport.DEEPSEEK_MODEL,
        'requested_model': judge.transport.DEEPSEEK_MODEL, 'finish_reason': 'stop',
        'refusal': None, 'seed': 42, 'enable_thinking': False, 'reasoning_content_present': False,
        'reasoning_tokens': None, 'raw_response': json.dumps({'score': grade, 'reasoning': 'Synthetic fixture.'}),
        'score': grade, 'reasoning': 'Synthetic fixture.', 'score_origin': 'new_api',
        'response_id': 'synthetic-response'}
    attempts = [{**{k: v for k, v in score.items() if k not in ('score', 'reasoning', 'score_origin')},
                 'status': 'ok', 'attempt': 1, 'retryable': False}]
    protocol = {'dataset': 'MemEye', 'task': 'open_ended_answering', 'method': 'synthetic',
        'answer_model': judge.base.ANSWER_MODEL, 'questions': 371, 'source_open_em_correct': 371,
        'unique_requests': 1, **judge.shared.judge_settings(),
        'source_sha256': {}, 'implementation_sha256': {}}
    judge.dump(root / 'protocol.json', protocol)
    for name, values in [('requests.jsonl', requests), ('question_map.jsonl', mapped),
                         ('reused_scores.jsonl', []), ('scores.jsonl', [score]),
                         ('attempts.jsonl', attempts),
                         ('per_question.jsonl', [{**row, **score, 'judge_status': 'ok'} for row in mapped])]:
        judge.write_rows(root / name, values)
    (root / 'judge_prompt.txt').write_bytes(judge.shared.PROMPT.read_bytes())
    report = {'status': 'complete', 'dataset': 'MemEye', 'task': 'open_ended_answering',
        'method': 'synthetic', 'answer_model': judge.base.ANSWER_MODEL,
        'questions_expected': 371, 'questions_scored': 371, 'questions_missing': 0,
        'unique_requests': 1, 'unique_requests_scored': 1, 'score_sum': grade * 371,
        'llm_judge': grade, 'llm_judge_percent': grade * 100,
        'score_distribution': {str(g): 371 if g == grade else 0 for g in judge.shared.GRADES},
        'source_open_em_correct': 371, 'source_open_em': 1., 'missing_ids': [],
        'judge_model': judge.transport.DEEPSEEK_MODEL, 'grades': list(judge.shared.GRADES),
        'api_attempt_count': 1, 'failed_api_attempt_count': 0,
        'artifact_sha256': {name: judge.sha(root / name) for name in
            ('protocol.json', 'requests.jsonl', 'question_map.jsonl', 'reused_scores.jsonl',
             'scores.jsonl', 'attempts.jsonl')}}
    judge.dump(root / 'summary.json', report)
    return canonical


def rehash(root, name):
    summary = judge.read(root / 'summary.json')
    summary['artifact_sha256'][name] = judge.sha(root / name)
    judge.dump(root / 'summary.json', summary)


class SyntheticCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix='memeye_judge_synthetic_', dir='/tmp')
        self.root = Path(self.tmp.name) / 'cache'
        self.canonical = synthetic_cache(self.root)
        self.template = judge.shared.PROMPT.read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return judge.load_completed_cache(self.root, self.template, self.canonical)

    def test_complete_deduplicated_cache_keeps_371_denominator(self):
        entries, hashes = self.load()
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0][2]['question_ids']), 371)
        self.assertIn(str(self.root / 'summary.json'), hashes)

    def test_partial_cache_rejected(self):
        report = judge.read(self.root / 'summary.json'); report['status'] = 'partial'
        judge.dump(self.root / 'summary.json', report)
        with self.assertRaisesRegex(ValueError, 'complete caches'):
            self.load()

    def test_noncanonical_gold_rejected_even_with_updated_artifact_hash(self):
        mapped = judge.read_rows(self.root / 'question_map.jsonl')
        mapped[0]['ground_truth'] = 'London'
        judge.write_rows(self.root / 'question_map.jsonl', mapped)
        rehash(self.root, 'question_map.jsonl')
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.load()

    def test_raw_grade_mismatch_rejected(self):
        values = judge.read_rows(self.root / 'scores.jsonl'); values[0]['score'] = .5
        judge.write_rows(self.root / 'scores.jsonl', values)
        rehash(self.root, 'scores.jsonl')
        with self.assertRaisesRegex(ValueError, 'raw judge output'):
            self.load()

    def test_summary_sum_mismatch_rejected(self):
        report = judge.read(self.root / 'summary.json'); report['score_sum'] -= .25
        judge.dump(self.root / 'summary.json', report)
        with self.assertRaisesRegex(ValueError, 'score_sum'):
            self.load()

    def test_wire_mutation_rejected_even_when_hash_manifest_updated(self):
        requests = judge.read_rows(self.root / 'requests.jsonl')
        requests[0]['request']['temperature'] = .5
        judge.write_rows(self.root / 'requests.jsonl', requests)
        rehash(self.root, 'requests.jsonl')
        with self.assertRaisesRegex(ValueError, 'wire request'):
            self.load()

    def test_changed_settings_rejected(self):
        protocol = judge.read(self.root / 'protocol.json'); protocol['seed'] = 7
        judge.dump(self.root / 'protocol.json', protocol)
        rehash(self.root, 'protocol.json')
        with self.assertRaisesRegex(ValueError, 'settings'):
            self.load()

    def test_cross_cache_conflicting_grade_rejected(self):
        second = Path(self.tmp.name) / 'second'
        synthetic_cache(second, grade=.5)
        with self.assertRaisesRegex(ValueError, 'Conflicting grades'):
            judge.collect_caches([self.root, second], self.template, self.canonical)

    def test_unhashed_score_edit_rejected(self):
        values = judge.read_rows(self.root / 'scores.jsonl'); values[0]['response_id'] = 'modified'
        judge.write_rows(self.root / 'scores.jsonl', values)
        with self.assertRaisesRegex(ValueError, 'artifact differs'):
            self.load()


class RealPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix='memeye_judge_real_prepare_', dir='/tmp')
        self.args = SimpleNamespace(method_id='new_method', method_label='New method label',
            input=judge.CANONICAL_ROOT / 'judge_inputs.jsonl', output=Path(self.tmp.name) / 'judge',
            source_protocol=judge.CANONICAL_ROOT / 'reader/protocol.json', cache_root=None,
            max_attempts=5, limit=0, workers=8, env_file=Path('/unused'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_baseline_exact_reuse_resume_and_public_metric(self):
        with patch.object(judge.transport, 'chat_request', side_effect=AssertionError('No API allowed')):
            p, requests, mapped = judge.prepare(self.args)
            self.assertEqual(p['method'], 'New method label')
            self.assertEqual((len(mapped), len(requests), p['cached_unique_requests']), (371, 369, 369))
            self.assertEqual((p, requests, mapped), judge.prepare(self.args))
            report = judge.base.analyze(self.args, p, requests, mapped)
            public = judge.publish_summary(self.args, p, report)
            self.assertEqual(public['llm_judge'], 139.25 / 371)
            self.assertFalse(any('em' in key for key in public))

    def test_resume_cannot_add_cache(self):
        judge.prepare(self.args)
        self.args.cache_root = [judge.DEFAULT_CACHE, Path(self.tmp.name) / 'extra']
        with self.assertRaisesRegex(ValueError, 'Cache roots are frozen'):
            judge.prepare(self.args)

    def test_reused_cache_can_be_reused_without_nested_provenance(self):
        p, requests, mapped = judge.prepare(self.args)
        judge.base.analyze(self.args, p, requests, mapped)
        canonical, _ = judge.load_canonical()
        cached, _, _ = judge.collect_caches([judge.DEFAULT_CACHE, self.args.output],
                                           judge.shared.PROMPT.read_text(), canonical)
        self.assertEqual(len(cached), 369)
        for score in cached.values():
            self.assertEqual(len(score['reuse_sources']), 2)
            self.assertNotIn('source_response', json.dumps(score))
            self.assertEqual({source['source_score_origin'] for source in score['reuse_sources']},
                             {'new_api', 'baseline_cache'})

    def test_changed_prediction_is_not_reused(self):
        values = judge.read_rows(self.args.input)
        values[0]['model_output'] = 'A unique synthetic answer used only by this offline test.'
        values[0]['em'] = 0
        path = Path(self.tmp.name) / 'changed.jsonl'
        judge.write_rows(path, values); self.args.input = path
        p, requests, mapped = judge.prepare(self.args)
        self.assertEqual(p['new_unique_requests'], 1)
        changed_hash = mapped[0]['request_sha256']
        reused = judge.read_rows(self.args.output / 'reused_scores.jsonl')
        self.assertNotIn(changed_hash, {r['request_sha256'] for r in reused})

    def test_input_canonical_mismatch_rejected(self):
        values = judge.read_rows(self.args.input); values[0]['question'] += ' Changed.'
        path = Path(self.tmp.name) / 'changed.jsonl'
        judge.write_rows(path, values); self.args.input = path
        with self.assertRaisesRegex(ValueError, 'canonical'):
            judge.prepare(self.args)


if __name__ == '__main__':
    unittest.main()
