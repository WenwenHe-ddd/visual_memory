"""Input-boundary checks for the open-ended Mem-Gallery comparison."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.data.mem_gallery_inputs import CAPTION_FIELD, prepare


class MemGalleryInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'data/Mem-Gallery'
        self.cache = self.base / 'captions/test'
        self.out = self.root / 'prepared'

        def write(relative, value):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = value if isinstance(value, bytes) else json.dumps(value).encode()
            path.write_bytes(content)
            return hashlib.sha256(content).hexdigest()

        self.write = write
        history = 'data/Mem-Gallery/data/image/Scene/D1_IMG_001.jpg'
        query = 'data/Mem-Gallery/data/image/Scene/QA_IMG_001.jpg'
        hhash, qhash = write(history, b'history fixture image'), write(query, b'query fixture image')
        raw = {
            'multi_session_dialogues': [{'session_id': 'D1', 'date': '2024-01-01', 'dialogues': [
                {'round': 'D1:1', 'user': 'Here is the lamp.', 'assistant': 'It is blue.',
                 'input_image': ['../image/Scene/D1_IMG_001.jpg'], 'image_id': ['D1:IMG_001'],
                 'image_caption': ['Official historical caption.']},
                {'round': 'D1:2', 'user': 'Remember Monday.', 'assistant': 'Okay.'},
            ]}],
            'human-annotated QAs': [
                {'question': 'What is this?', 'point': 'TTL', 'session_id': ['D1'],
                 'question_image': '../image/Scene/QA_IMG_001.jpg',
                 'image_caption': 'Official query caption.', 'answer': 'SECRET_GOLD_ANSWER',
                 'clue': ['D1:1', 'D1:2']},
                {'question': 'Was a fridge discussed?', 'point': 'AR', 'session_id': ['D1'],
                 'answer': 'Not mentioned.', 'clue': []},
            ],
        }
        derived = json.loads(json.dumps(raw))
        derived['multi_session_dialogues'][0]['dialogues'][0][CAPTION_FIELD] = ['Qwen historical caption.']
        derived['human-annotated QAs'][0][CAPTION_FIELD] = 'Qwen query caption must not be used.'
        raw_path = 'data/Mem-Gallery/data/dialog/Scene.json'
        raw_hash = write(raw_path, raw)
        derived_hash = write('data/Mem-Gallery/captions/test/dialog_with_qwen_caption/Scene.json', derived)
        audit_hash = write('data/Mem-Gallery/captions/test/independent_audit.json', {'status': 'passed'})
        write('data/Mem-Gallery/captions/test/artifact_manifest.json', {
            'dialog_with_qwen_caption/Scene.json': derived_hash,
            'independent_audit.json': audit_hash,
        })
        write('data/Mem-Gallery/source_manifest.json', {
            'repo': 'fixture/gallery', 'revision': 'fixed', 'files': [{'path': raw_path, 'sha256': raw_hash}],
        })
        write('data/Mem-Gallery/image_manifest.json', {
            'revision': 'fixed', 'files': [
                {'path': 'data/image/Scene/D1_IMG_001.jpg', 'sha256': hhash},
                {'path': 'data/image/Scene/QA_IMG_001.jpg', 'sha256': qhash},
            ],
        })

    def run_prepare(self):
        return prepare(self.out, root=self.root, caption_dir=self.cache)

    def rows(self, name):
        return [json.loads(line) for line in (self.out / name).read_text().splitlines()]

    def test_gold_isolation_caption_policy_and_complete_memory_pool(self):
        result = self.run_prepare()
        memories, questions, gold = (self.rows(name) for name in
                                     ['memories.jsonl', 'questions.jsonl', 'gold.jsonl'])
        candidates = self.rows('sanitized_candidate_inputs.jsonl')
        self.assertEqual([m['is_visual'] for m in memories], [True, False])
        self.assertEqual([m['insertion_index'] for m in memories], [0, 1])
        self.assertEqual(memories[0]['caption'], 'Qwen historical caption.')
        self.assertEqual(memories[1]['caption'], '')
        self.assertIsNone(memories[1]['node_id'])
        self.assertIn('Image ID: D1:IMG_001', memories[0]['reader_text'])
        self.assertNotIn('Image ID:', memories[0]['retrieval_text'])
        self.assertEqual(questions[0]['query_image_caption'], 'Official query caption.')
        self.assertEqual(gold[0]['gold_ids'], ['Scene/D1:1', 'Scene/D1:2'])
        self.assertEqual(gold[1]['gold_ids'], [])
        self.assertEqual(result['counts']['questions_without_gold'], 1)
        for question in questions:
            self.assertFalse({'answer', 'clue', 'session_id'} & question.keys())
        for candidate in candidates:
            self.assertEqual(set(candidate), {'id', 'question', 'query_image_caption'})
        self.assertNotIn('SECRET_GOLD_ANSWER', json.dumps(questions + candidates + memories))
        self.assertEqual([q['id'] for q in questions], [g['id'] for g in gold])

    def test_rerun_preserves_artifacts_and_rejects_different_output(self):
        first = self.run_prepare()
        stats = {p.name: p.stat().st_mtime_ns for p in self.out.iterdir()}
        self.assertEqual(first, self.run_prepare())
        self.assertEqual(stats, {p.name: p.stat().st_mtime_ns for p in self.out.iterdir()})
        questions = self.out / 'questions.jsonl'
        questions.write_text('altered output\n')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            self.run_prepare()
        self.assertEqual(questions.read_text(), 'altered output\n')

    def test_changed_source_image_is_rejected_before_creating_output(self):
        (self.base / 'data/image/Scene/D1_IMG_001.jpg').write_bytes(b'changed source image')
        with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_derived_original_field_changes_are_rejected_even_with_updated_cache_hash(self):
        derived = self.cache / 'dialog_with_qwen_caption/Scene.json'
        data = json.loads(derived.read_text())
        data['human-annotated QAs'][0]['answer'] = 'tampered original answer'
        digest = self.write(str(derived.relative_to(self.root)), data)
        manifest = self.cache / 'artifact_manifest.json'
        hashes = json.loads(manifest.read_text())
        hashes['dialog_with_qwen_caption/Scene.json'] = digest
        self.write(str(manifest.relative_to(self.root)), hashes)
        with self.assertRaisesRegex(ValueError, 'Derived JSON changed original fields'):
            self.run_prepare()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
