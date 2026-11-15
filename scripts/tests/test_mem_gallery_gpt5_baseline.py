"""Check the controlled input boundary against the existing Natural baseline."""
import unittest
from scripts.evaluation import mem_gallery_gpt5_baseline as run


class NaturalBaselineTests(unittest.TestCase):
    def test_formats_match_original_with_only_caption_substitution(self):
        observations = run.read(run.SOURCE / 'observations.json')
        current = run.memory_records(observations)
        previous = {r['id']: r for r in run.rows(run.REFERENCE / 'inputs/memories.jsonl')}
        self.assertEqual(len(current), 3962)
        for m in current:
            old = previous[m['id']]
            for key in ('id', 'task', 'round', 'timestamp', 'session_id', 'dialogue', 'is_visual', 'insertion_index'):
                self.assertEqual(m[key], old[key], (m['id'], key))
            for key in ('retrieval_text', 'reader_text'):
                expected = old[key].replace(old['caption'], m['caption']) if old['is_visual'] else old[key]
                self.assertEqual(m[key], expected, (m['id'], key))

    def test_ranked_context_and_reader_match_original(self):
        memories = run.memory_records(run.read(run.SOURCE / 'observations.json'))
        mmap = {m['id']: m for m in memories}
        q = run.rows(run.SOURCE / 'inputs/questions.jsonl')[0]
        ids = [m['id'] for m in memories if m['task'] == q['task']][:5][::-1]
        context = run.context_for(ids, mmap)
        payload = run.ga.reader_payload(dict(q, answer='SECRET_GOLD', clue='SECRET_CLUE'), context)
        self.assertEqual(payload['user_text'], run.ga.gallery.reader_text(q, ids, mmap))
        self.assertNotIn('SECRET_', payload['user_text'])
        self.assertEqual(payload['query_images'], [dict(path=p, sha256=run.sha(p)) for p in q['query_image_paths']])


if __name__ == '__main__':
    unittest.main()
