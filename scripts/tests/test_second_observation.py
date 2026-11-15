"""Check image/caption alignment and mathematical/truncation safeguards."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.tests.second_observation import select_captions, sha256, similarities


class SecondObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = [self.root / 'a.png', self.root / 'b.png']
        for i, path in enumerate(self.images):
            path.write_bytes(f'image-content-{i}'.encode())

    def source(self, rows):
        path = self.root / 'captions.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_hash_mapping_ignores_record_order(self):
        source = self.source([{'id': sha256(p), 'caption': p.stem} for p in reversed(self.images)])
        self.assertEqual([r['caption'] for r in select_captions(self.images, [source])], ['a', 'b'])

    def test_explicit_text_pairing(self):
        sources = [self.root / 'a.txt', self.root / 'b.txt']
        for source in sources:
            source.write_text(source.stem)
        self.assertEqual([r['caption'] for r in select_captions(self.images, sources)], ['a', 'b'])

    def test_relative_paths_are_relative_to_source(self):
        source = self.source([{'image_path': p.name, 'caption': p.stem} for p in self.images])
        self.assertEqual(len(select_captions(self.images, [source])), 2)

    def test_missing_and_ambiguous_matches_fail(self):
        row = {'id': sha256(self.images[0]), 'caption': 'caption'}
        for rows in ([row], [row, row]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'exactly one'):
                select_captions(self.images, [self.source(rows)])

    def test_keyed_file_cannot_be_positionally_misassigned(self):
        source = self.source([{'id': 'incorrect-hash', 'caption': 'caption'}])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            select_captions(self.images, [source, source])

    def test_reject_generation_truncation(self):
        source = self.source([{'id': sha256(p), 'caption': 'caption', 'hit_token_limit': True}
                              for p in self.images])
        with self.assertRaisesRegex(ValueError, 'generation token limit'):
            select_captions(self.images, [source])

    def test_cosine_and_embedding_truncation(self):
        model = SimpleNamespace(max_seq_length=4,
                                tokenizer=SimpleNamespace(encode=lambda text, **kwargs: text.split()),
                                encode=lambda texts, **kwargs: [[3, 0], [0, 4]])
        records = [{'caption': 'first'}, {'caption': 'second'}]
        self.assertEqual(similarities(model, records, 2), [[1, 0], [0, 1]])
        records[0]['caption'] = 'one two three four five'
        with self.assertRaisesRegex(ValueError, 'refusing silent truncation'):
            similarities(model, records, 2)


if __name__ == '__main__':
    unittest.main()
