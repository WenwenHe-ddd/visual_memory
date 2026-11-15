"""Scoring invariants and routing for the pipeline's ridge-z debug adapter."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from scripts import visual_memory_pipeline as pipeline
from scripts.memory import online_retrieval as adapter
from scripts.memory.online_zscore import score_master, ZScoreWeighting


class ZReweightPipelineTests(unittest.TestCase):
    base = ['online', '--memory', 'caption_visual_master', '--online-weighting', 'z_score']

    def test_dispatch_and_explicit_hyperparameters(self):
        with patch.object(adapter, 'online') as online, patch('scripts.memory.caption_visual_master.online') as old:
            pipeline.main(self.base + ['--device','cpu','--question-ids','Q10',
                '--whitening-lambda','.1','--reweight-alpha','1','--memory-dir','/tmp/memory'])
        old.assert_not_called(); online.assert_called_once()
        args=online.call_args.args[0]
        self.assertEqual((args.whitening_lambda,args.reweight_alpha,args.question_ids,args.top_k),(.1,1.,['Q10'],5))
        self.assertEqual(args.memory_dir,Path('/tmp/memory'))
        self.assertEqual(args.output,pipeline.ROOT/'results/debug_online_retrieval/caption_visual_master/multi_path/z_score')

    def test_default_hyperparameters_and_invalid_parameters_fail_before_dispatch(self):
        with patch.object(adapter,'online') as online:
            pipeline.main(self.base)
            args=online.call_args.args[0]
            self.assertEqual((args.whitening_lambda,args.reweight_alpha),(.1,1.))
            self.assertIsNone(args.question_ids)  # All source questions, never the old three-question default.
            online.reset_mock()
            for flags in [['--whitening-lambda','0'],['--whitening-lambda','nan'],['--whitening-lambda','inf'],
                          ['--reweight-alpha','-1'],['--reweight-alpha','nan'],['--reweight-alpha','inf'],
                          ['--stage','run'],['--reader-backend','api'],['--dataset','mem_gallery']]:
                with self.subTest(flags=flags),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                    pipeline.main(self.base+flags)
                self.assertEqual(exc.exception.code,2)
            online.assert_not_called()
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            pipeline.main(['online','--memory','caption_visual_master','--whitening-lambda','.1'])

    def test_alpha_zero_recovers_raw_and_positive_correction_is_bounded(self):
        rng=np.random.default_rng(14);x=rng.normal(size=(6,12));x/=np.linalg.norm(x,axis=1,keepdims=True)
        q=rng.normal(size=12);q/=np.linalg.norm(q)
        for ridge in [.01,.1,1.]:
            with self.subTest(ridge=ridge):
                zero=score_master(q,x,ridge,0)
                np.testing.assert_array_equal(zero['weighted'],x@q)
                result=score_master(q,x,ridge,1)
                self.assertLessEqual(abs(result['correction']).max(),np.std(x@q)+1e-15)
                # Independent covariance inverse verifies the adapter's query projection.
                centered=x-x.mean(0);cov=centered.T@centered/(len(x)-1)
                expected=x@np.linalg.solve(cov+ridge*np.eye(12),ridge*q)
                np.testing.assert_allclose(result['white'],expected,atol=1e-13)

    def test_singleton_and_equal_scores_have_no_correction(self):
        for x in [np.array([[1.,0.]]),np.array([[1.,0.],[1.,0.]])]:
            result=score_master(np.array([.8,.6]),x,.01,1)
            np.testing.assert_array_equal(result['weighted'],result['raw'])

    def test_question_ids_are_not_silently_taken_from_another_domain(self):
        questions=[{'id':'Home/Q10'},{'id':'Other/Q10'}]
        self.assertEqual(adapter.select_questions(questions,['Home/Q10']),[questions[0]])
        for ids in [['Q10'],['missing'],['Home/Q10','Home/Q10']]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):adapter.select_questions(questions,ids)

    def test_both_memories_use_the_same_component_and_explicit_none_route(self):
        with patch.object(adapter, 'online') as online, patch.object(pipeline, 'prepare_online') as legacy:
            for memory in ['natural', 'caption_visual_master']:
                for weighting in ['none', 'z_score']:
                    pipeline.main(['online', '--memory', memory, '--online-weighting', weighting])
                    args = online.call_args.args[0]
                    self.assertEqual((args.mode, args.online_weighting), (memory, weighting))
                    expected = args.question_source.parent / 'memory' if memory == 'caption_visual_master' else args.question_source.parent
                    self.assertEqual(args.memory_dir, expected)
            legacy.assert_not_called()
            pipeline.main(['online', '--memory', 'natural', '--retrieval-scoring', 'z_reweight'])
            self.assertEqual(online.call_args.args[0].online_weighting, 'z_score')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.main(self.base + ['--online-weighting', 'none', '--reweight-alpha', '0'])

    def test_natural_loading_needs_no_master_graph_candidates_or_clip(self):
        from types import SimpleNamespace
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = {'image_id': 'Home/M1#image1', 'memory_id': 'Home/M1', 'dialogue': 'dialogue', 'caption': 'caption'}
            (root/'inputs.json').write_text(json.dumps({'leaves': [leaf]}))
            (root/'protocol.json').write_text('{}')
            args = SimpleNamespace(mode='natural', memory_dir=root)
            with patch.object(adapter, 'load_clip_text_encoder', side_effect=AssertionError('Natural loaded CLIP')):
                memory, _, _, devices = adapter.load_memory(args, [{'id': 'Home/Q1'}])
            queries, groups = memory.plan({'retrieval_query': 'unchanged question with cached image caption'})
            self.assertEqual(queries, ['unchanged question with cached image caption'])
            self.assertEqual([g.indices for g in groups], [(0,)])
            self.assertEqual(devices, {})

    def test_weighting_preserves_queries_texts_and_all_group_members_before_topk(self):
        rng = np.random.default_rng(21)
        vectors = rng.normal(size=(5, 8)); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = rng.normal(size=(4, 8)); queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        leaves = [{'image_id': f'Home/M{i}#image1', 'memory_id': f'Home/M{i}',
                   'dialogue': f'dialogue {i}', 'caption': f'caption {i}', 'master_id': 'm1' if i < 3 else 'm2'} for i in range(5)]
        masters = [{'id': key, 'n': len(ids), 'member_ids': [leaves[i]['image_id'] for i in ids]}
                   for key, ids in [('m1', [0, 1, 2]), ('m2', [3, 4])]]
        question = {'id': 'Home/Q1', 'question': 'question', 'retrieval_query': 'query including cached image caption',
                    'options': {k: k for k in 'ABCD'}, 'gold_ids': ['Home/M0'], 'gold_choice': 'A'}
        statements = {k: f'Complete statement for {k}.' for k in 'ABCD'}
        candidates = {'Home/Q1': {**question, 'candidate_statements': statements}}
        memories = [adapter.NaturalCaptionMemory(leaves), adapter.CaptionVisualMasterMemory(
            leaves, masters, np.zeros((2, 8)), candidates, None, None, 77)]
        for memory in memories:
            with self.subTest(memory=memory.name), patch.object(adapter, 'choose_master_answers',
                    return_value=(np.zeros((2, 4)), np.array([1, 3]))) as choose:
                calls = []
                def encode(model, texts, prefix=''):
                    calls.append((list(texts), prefix))
                    return vectors.copy() if list(texts) == memory.texts else queries[:len(texts)].copy()
                weight = ZScoreWeighting(.1, 1)
                with patch.object(adapter, 'encode_text', side_effect=encode), patch.object(
                        ZScoreWeighting, 'apply', autospec=True, side_effect=ZScoreWeighting.apply) as apply:
                    raw = adapter.retrieve_question(question, memory, None, 'prefix ', 2)
                    self.assertEqual(apply.call_count, 0)
                    result = adapter.retrieve_question(question, memory, None, 'prefix ', 2, weight)
                    sizes = [len(call.args[2]) for call in apply.call_args_list]
                    self.assertEqual(sizes, [5] if memory.name == 'natural' else [3, 2])
                self.assertEqual(calls[:2], calls[2:])
                self.assertEqual(raw['conditions'][0], result['conditions'][0])
                self.assertEqual(raw['conditions'][0]['top5_memory_ids'], raw['conditions'][1]['top5_memory_ids'])
                if memory.name == 'natural':
                    choose.assert_not_called()
                    self.assertEqual(calls[0][0], [question['retrieval_query']])
                else:
                    self.assertEqual(result['groups'][0]['query'], question['retrieval_query'] + '\nCandidate answer: ' + statements['B'])
                    self.assertEqual(result['groups'][1]['selected_statement'], statements['D'])
                    # Full scan can use the same master memory without option expansion.
                    choose.reset_mock()
                    with patch.object(adapter, 'encode_text', side_effect=encode):
                        full_scan = adapter.retrieve_question(question, memory, None, 'prefix ', 2, weight,
                                                              adapter.FullScanRetrieval())
                    choose.assert_not_called()
                    self.assertEqual([g['query'] for g in full_scan['groups']], [question['retrieval_query']] * 2)
                    self.assertEqual([g['size'] for g in full_scan['groups']], [3, 2])
                # Gold labels are strictly evaluation-only.
                with patch.object(adapter, 'encode_text', side_effect=encode):
                    changed = adapter.retrieve_question({**question, 'gold_ids': ['Home/M4'], 'gold_choice': 'D'},
                                                        memory, None, 'prefix ', 2, weight)
                self.assertEqual(changed['leaf_scores'], result['leaf_scores'])
                self.assertEqual(changed['groups'], result['groups'])


if __name__=='__main__':unittest.main()
