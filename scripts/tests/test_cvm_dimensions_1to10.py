"""Cross-implementation checks for the dimension ablation's replay adapter."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.analysis.compare_caption_visual_dimensions import PartitionReplay
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory, _unit


class DimensionReplayTests(unittest.TestCase):
    def replay(self):
        rng = np.random.default_rng(37)
        basis = np.linalg.qr(rng.normal(size=(10, 10)))[0]
        core = CaptionWardVisualMemory(basis[:, :2])
        replays = {d: PartitionReplay(basis[:, :d]) for d in range(1, 11)}
        vectors = {}
        # Explicitly exercises merges, splits, partial migration, and unchanged groups.
        partitions = [[[0]], [[0], [1]], [[0, 1, 2]], [[0, 2], [1, 3]],
                      [[0, 1, 3], [2, 4]], [[0, 1, 3], [2, 4, 5]]]
        for step, partition in enumerate(partitions):
            visual = _unit(rng.normal(size=10), 'test')
            key = str(step)
            text = _unit(rng.normal(size=3), 'text')
            choice = {'partition': partition, 'selected_k': len(partition),
                      'candidate_scores': [], 'selection_reason': 'test fixed partitions'}
            with patch('scripts.graph.caption_ward_visual.choose_caption_partition', return_value=choice):
                audit = core.add(key, key, text, visual, dialogue='dialogue', timestamp=step)
            normalized = _unit(visual, 'match core')
            vectors[key] = normalized
            row = {'image_id': key, 'caption': key, 'dialogue': 'dialogue', 'timestamp': step}
            for d, memory in replays.items():
                memory.add(row, normalized, audit)
                for master in memory.masters.values():
                    projected = sum(np.array(memory.image_nodes[i]['visual_coords']) for i in master['member_ids'])
                    np.testing.assert_allclose(memory.basis.T @ master['visual_sum'], projected, atol=2e-14)
                    if d == 10:
                        # Full dimensional representation recovers every true aggregate after splits.
                        np.testing.assert_allclose(master['visual_sum'], sum(vectors[i] for i in master['member_ids']), atol=2e-14)
                np.testing.assert_allclose(sum(m['visual_sum'] for m in memory.masters.values()), sum(vectors.values()), atol=2e-14)
            for mid in core.masters:
                np.testing.assert_allclose(core.masters[mid]['visual_sum'], replays[2].masters[mid]['visual_sum'], atol=2e-14)
        return replays

    def test_split_merge_conservation_and_two_dimensional_parity(self):
        self.replay()

    def test_saved_coordinates_have_requested_dimension_and_nested_basis(self):
        import json
        memories = self.replay()
        with tempfile.TemporaryDirectory() as directory:
            for d, memory in memories.items():
                target = Path(directory) / str(d)
                memory.save(target)
                graph = json.loads((target / 'graph.json').read_text())
                self.assertEqual(graph['leaf_visual_dim'], d)
                self.assertTrue(all(len(n['visual_coords']) == d for n in graph['image_nodes']))
                np.testing.assert_array_equal(np.load(target / 'basis.npy'), memories[10].basis[:, :d])


if __name__ == '__main__':
    unittest.main()
