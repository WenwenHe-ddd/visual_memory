import unittest
from dataclasses import dataclass, field

from scripts.evaluation import mem_gallery_qwen_text_methods as run


@dataclass
class Result:
    items: list
    expansion_candidates: list = field(default_factory=list)

    def to_dict(self):
        return dict(items=self.items, expansion_candidates=self.expansion_candidates)


class GalleryTextTests(unittest.TestCase):
    def test_reader_matches_original_payload(self):
        q = dict(question='Which image?', point='VS')
        memories = {'a': dict(reader_text='Date: today\nImage ID: D1:IMG_001')}
        context = 'Memory 1: ' + memories['a']['reader_text']
        self.assertEqual(run.reader_text(q, context), run.gallery.reader_text(q, ['a'], memories))

    def test_reader_preserves_empty_context_and_refusal_instruction(self):
        q = dict(question='Unknown?', point='AR')
        actual = run.reader_text(q, '')
        self.assertEqual(actual, run.gallery.reader_text(q, [], {}))
        self.assertIn('Not mentioned.', actual)

    def test_bounded_context_keeps_native_order(self):
        original = Result([dict(id=x) for x in ['b', 'a', 'b', 'd', 'c', 'f', 'e']], ['b', 'e'])
        limited, detail = run.bounded_simplemem(original)
        self.assertEqual(detail['context_memory_ids'], ['b', 'a', 'd', 'c', 'f'])
        self.assertEqual(limited.expansion_candidates, ['b'])
        self.assertEqual(len(original.items), 7)

    def test_sparse_and_empty_retrieval_are_not_filled(self):
        for count in [0, 1, 3]:
            limited, detail = run.bounded_simplemem(Result([dict(id=str(x)) for x in range(count)]))
            self.assertEqual(len(limited.items), count)
            self.assertEqual(detail['source_empty_preserved'], count == 0)

    def test_gallery_scoring_preserves_image_ids(self):
        self.assertEqual(run.gallery.score_answer('D2:IMG_003.', 'D2:IMG_003')['em'], 1)
        self.assertEqual(run.gallery.score_answer('D2:IMG_003', 'D2:IMG_004')['em'], 0)
        self.assertEqual(run.gallery.score_answer('The blue shirt', 'blue shirt')['em'], 1)


if __name__ == '__main__':
    unittest.main()
