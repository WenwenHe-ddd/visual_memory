"""CPU checks for candidate isolation, paired payloads and open-answer scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation.mem_gallery_master_reader import (
    CONDITIONS, build_reader_manifest, ensure_protocol, normalize_answer_universal,
    parse_candidates, score_answer, run_reader, analyze,
)


class WordTokenizer:
    def __call__(self, text, truncation=False):
        assert truncation is False
        return {'input_ids': [0] + list(range(len(text.split()))) + [1]}


class MemGalleryReaderTests(unittest.TestCase):
    def test_candidate_requires_four_distinct_short_statements(self):
        statements = ['A red sofa.', 'A blue chair.', 'A green table.', 'A yellow cabinet.']
        self.assertEqual(parse_candidates(json.dumps(statements), WordTokenizer())[0], statements)
        for invalid in [statements[:3], statements[:3] + [statements[0].upper()],
                        statements[:3] + ['word ' * 76], statements[:3] + ['D1:IMG_001']]:
            with self.assertRaises(ValueError):
                parse_candidates(json.dumps(invalid), WordTokenizer())
        with self.assertRaises(ValueError):
            parse_candidates('```json\n' + json.dumps(statements) + '\n```', WordTokenizer())

    def test_exact_four_line_candidate_formats_preserve_statement_content(self):
        statements = ['A red sofa.', 'A blue chair.', 'A green table.', 'A yellow cabinet.']
        cases = [('four_lines_unmarked', '\n\n'.join(statements)),
                 ('four_lines_numbered', '\n'.join(f'{i}. {text}' for i, text in enumerate(statements, 1))),
                 ('four_lines_bulleted', '\n'.join('- ' + text for text in statements))]
        for expected, raw in cases:
            parsed, lengths, detected = parse_candidates(raw, WordTokenizer(), return_format=True)
            self.assertEqual(parsed, statements)
            self.assertEqual(detected, expected)
            self.assertEqual(lengths, [5] * 4)
        malformed = ['[\n"A red sofa",\n"A blue chair",\n"A green table",\n"A cabinet"',
                     'Here are the candidates:\nA red sofa.\nA blue chair.\nA green table.',
                     'A red sofa.\n2. A blue chair.\nA green table.\nA yellow cabinet.',
                     '1. A red sofa.\n2. A blue chair.\n4. A green table.\n5. A yellow cabinet.',
                     '\n'.join('"' + text + '",' for text in statements)]
        for raw in malformed:
            with self.assertRaises(ValueError):
                parse_candidates(raw, WordTokenizer())

    def inputs(self):
        memories = [{'id': f'domain/D1:{i}', 'task': 'domain',
                     'reader_text': f'Original dialogue and caption {i}.'} for i in range(1, 7)]
        question = {'id': 'domain/QA_0000', 'task': 'domain', 'point': 'VS',
                    'question': 'Which image contains the chair?',
                    'query_image_paths': ['/unread/query.jpg'],
                    'query_image_caption': 'RETRIEVAL-ONLY CAPTION',
                    'retrieval_query': 'RETRIEVAL-ONLY EXPANSION'}
        retrieved = {'id': question['id'], 'task': 'domain', 'candidate_statements': ['SECRET HYPOTHESIS'],
                     'conditions': [{'name': name, 'top5_memory_ids': [row['id'] for row in memories[:5]],
                                     'top5_scores': [.9, .8, .7, .6, .5]} for name in CONDITIONS]}
        return [question], memories, [retrieved]

    def test_identical_payloads_share_hash_and_exclude_retrieval_hypotheses(self):
        manifest = build_reader_manifest(*self.inputs(), image_hash=lambda path: 'image-sha')
        self.assertEqual(len(manifest), 2)
        self.assertEqual(manifest[0]['payload_sha256'], manifest[1]['payload_sha256'])
        for row in manifest:
            self.assertNotIn('RETRIEVAL-ONLY', row['reader_text'])
            self.assertNotIn('SECRET HYPOTHESIS', row['reader_text'])
            self.assertIn('Return the image_id', row['reader_text'])
            self.assertEqual(row['query_images'], [{'path': '/unread/query.jpg', 'sha256': 'image-sha'}])

    def test_ordered_evidence_and_image_pixels_affect_payload_hash(self):
        questions, memories, retrieved = self.inputs()
        retrieved[0]['conditions'][1]['top5_memory_ids'].reverse()
        rows = build_reader_manifest(questions, memories, retrieved, image_hash=lambda path: 'image-sha')
        self.assertNotEqual(rows[0]['payload_sha256'], rows[1]['payload_sha256'])
        changed = build_reader_manifest(questions, memories, retrieved, image_hash=lambda path: 'different-pixels')
        self.assertNotEqual(rows[0]['payload_sha256'], changed[0]['payload_sha256'])

    def test_reader_rejects_gold_and_cross_domain_evidence(self):
        questions, memories, retrieved = self.inputs()
        questions[0]['answer'] = 'D1:IMG_002'
        with self.assertRaises(ValueError):
            build_reader_manifest(questions, memories, retrieved, image_hash=lambda path: 'sha')
        del questions[0]['answer']
        memories[0]['task'] = 'another-domain'
        with self.assertRaises(ValueError):
            build_reader_manifest(questions, memories, retrieved, image_hash=lambda path: 'sha')

    def test_resume_rejects_changed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            first = ensure_protocol(path, {'seed': 42, 'source_hash': 'first'})
            self.assertEqual(ensure_protocol(path, {'seed': 42, 'source_hash': 'first'}), first)
            with self.assertRaises(ValueError):
                ensure_protocol(path, {'seed': 42, 'source_hash': 'changed'})

    def test_official_normalization_and_stemming(self):
        self.assertEqual(normalize_answer_universal('The D1:IMG_001 and 3.14.'), 'd1 img_001 3.14')
        exact = score_answer('D1:IMG_001.', 'D1:IMG_001')
        self.assertEqual(exact['em'], 1.0)
        self.assertEqual(exact['f1'], 1.0)
        self.assertEqual(score_answer('Running dogs', 'run dog')['f1'], 1.0)
        self.assertEqual(score_answer('red sofa', 'red chair')['f1'], .5)
        self.assertEqual(score_answer('', 'red chair')['f1'], 0.0)

    def test_reader_fresh_dedup_resume_and_gold_free_generation(self):
        questions, memories, retrieved = self.inputs()
        questions[0]['query_image_paths'] = []
        second = dict(questions[0], id='domain/QA_0001', question='Which image contains the sofa?')
        questions.append(second)
        retrieved.append({'id': second['id'], 'task': 'domain', 'conditions': [
            {'name': CONDITIONS[0], 'top5_memory_ids': [row['id'] for row in memories[:5]], 'top5_scores': [.9] * 5},
            {'name': CONDITIONS[1], 'top5_memory_ids': [row['id'] for row in memories[1:]], 'top5_scores': [.8] * 5}]})

        def fake_generate(runtime, rows, max_new_tokens):
            self.assertEqual(max_new_tokens, 256)
            return [{'raw_response': 'D1:IMG_001', 'input_tokens': 10, 'generated_tokens': 5,
                     'hit_token_limit': False, 'batch_elapsed_seconds': 0.1, 'batch_size': len(rows)} for row in rows]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            for name, rows in [('inputs/questions.jsonl', questions), ('inputs/memories.jsonl', memories),
                               ('retrieval_results.jsonl', retrieved)]:
                (root / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            # There is deliberately no gold file during either reader invocation.
            module = 'scripts.evaluation.mem_gallery_master_reader.'
            with patch(module + 'model_provenance', return_value={'model': 'test'}), \
                 patch(module + 'load_runtime', return_value=(None, None, {})) as loader, \
                 patch(module + 'generate_batch', side_effect=fake_generate):
                first = run_reader(root)
                self.assertEqual(first['logical_records'], 4)
                self.assertEqual(first['fresh_inferences'], 3)
                self.assertEqual(first['shared_records'], 1)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(run_reader(root), first)
                self.assertEqual(loader.call_count, 1)
            gold = [{'id': row['id'], 'point': row['point'], 'answer': 'D1:IMG_001',
                     'gold_ids': [memories[0]['id']]} for row in questions]
            (root / 'inputs/gold.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in gold))
            result = analyze(root)
            self.assertEqual(result['overall']['questions'], 2)
            self.assertEqual(result['overall'][CONDITIONS[0]]['em'], 1.0)
            self.assertEqual(result['overall'][CONDITIONS[1]]['recall_at_5_micro'], .5)


if __name__ == '__main__':
    unittest.main()
