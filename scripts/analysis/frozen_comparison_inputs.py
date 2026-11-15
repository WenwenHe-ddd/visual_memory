"""Read fixed candidate sets retained by independent visual diagnostics.

This module does not build graphs or rerun the retired retrieval method. Historical
selections and responses come only from the independent experiment's saved cache.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'data/independent_visual_inputs'
COMPARISON = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/qa_reuse_source.json'
Q10 = ROOT / 'reports/home_q10_top5_embeddings_20261020'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def memories():
    return {row['id']: row for row in rows(INPUTS / 'retrieval_memories.jsonl')}


def questions():
    from scripts.analysis.evaluate_adaptive_changed_qa import reader_text, digest
    memory = {mid: row['reader_text'] for mid, row in memories().items()}
    cache = json.loads(COMPARISON.read_text())
    selected = {row['id']: row for row in cache['records']}
    result = []
    for q in rows(INPUTS / 'qa_manifest.jsonl'):
        old = selected[q['id']]
        text = reader_text(q, old['ordered_memory_ids'], memory)
        images = [{'path': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                  for p in q['query_image_paths']]
        assert images == old['query_images']
        payload = {'system': cache['source_reader_protocol']['reader_system_prompt'],
                   'user_text': text, 'query_images': images}
        assert digest(payload) == old['payload_sha256']
        result.append({**q, 'top_memory_ids': old['ordered_memory_ids'],
                       'reader_text': text, 'payload_sha256': old['payload_sha256']})
    return result


def q10_memories():
    """Preserve the diagnostic's fixed text variants and previously saved vectors."""
    import numpy as np
    canonical = memories()
    saved = json.loads((Q10 / 'inputs.json').read_text())
    with np.load(Q10 / 'embeddings.npz', allow_pickle=False) as archive:
        ids = archive['memory_ids'].tolist()
        vectors = archive['text_dialogue_unique_caption'].copy()
    assert ids == [r['memory_id'] for r in saved]
    result = {}
    for row, vector in zip(saved, vectors, strict=True):
        mid = row['memory_id']
        assert row['dialogue_full_caption'] == canonical[mid]['reader_text']
        assert row['full_caption'] == canonical[mid]['caption']
        result[mid] = {**canonical[mid], 'retrieval_text': row['retrieval_text'],
                       'embedding': vector.tolist()}
    return result
