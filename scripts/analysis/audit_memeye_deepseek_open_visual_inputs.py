"""Independent CLIP native payload/source audit; no API calls or gold loading."""
from pathlib import Path
import json
import hashlib
import copy
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import clip_image_dialogue_api as native

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def main():
    root = ROOT / 'results/memeye_deepseek_open_20261106/visual_methods/clip_image_dialogue'
    source = ROOT / 'results/memeye_gpt5_open_20261105/visual_methods/clip_image_dialogue'
    p = json.loads((root / 'protocol.json').read_text())
    values = rows(root / 'reader_manifest.jsonl')
    previous = {r['id']: r for r in rows(source / 'reader_manifest.jsonl')}
    canonical = {r['id']: r for r in rows(ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl')}
    assert len(values) == len(previous) == len(canonical) == 371
    assert len({r['id'] for r in values}) == 371
    assert p['answer_model'] == 'deepseek-v4.1-flash' and p['source_mcq_correct'] == 212
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected
    wires = []
    query_image_questions = 0
    for row in values:
        qid = row['id']; prev = previous[qid]; q = canonical[qid]
        assert not set(row) & {'answer', 'options', 'gold_choice', 'ground_truth', 'clue', 'query_image_captions'}
        assert [row[k] for k in ('id', 'task', 'point', 'question', 'query_image_paths')] == [q[k] for k in ('id', 'task', 'point', 'question', 'query_image_paths')]
        assert row['reader_messages'] == prev['reader_messages']
        assert row['retrieval_trace'] == prev['retrieval_trace']
        assert row['retrieval_trace']['retrieval_query'] == row['question']
        assert row['reader_image_hashes'] == prev['reader_image_hashes']
        for name, expected in row['reader_image_hashes'].items():
            assert sha(name) == expected
        messages = copy.deepcopy(row['reader_messages'])
        for m in messages:
            if isinstance(m['content'], list):
                for part in m['content']:
                    if part['type'] == 'image':
                        assert sha(part['path']) == part.pop('sha256')
        wire = native.build_request(messages, row['reader_image_hashes'], p['answer_model'])
        assert wire['seed'] == 42 and wire['max_tokens'] == 16
        wire['max_tokens'] = 2048
        assert {k: v for k, v in wire.items() if k != 'messages'} == p['settings']
        assert len(row['retrieval_trace']['retrieved_images']) == 5
        query_image_questions += bool(row['query_image_paths'])
        wires.append({'id': qid, 'native_open_wire_sha256': native.digest(wire), 'image_count': len(row['reader_image_paths'])})
    assert query_image_questions == 8
    result = {'status': 'complete', 'questions': 371, 'native_interleaved_reader_wires': 371,
        'original_source_mcq_correct': 212, 'model': p['answer_model'], 'open_reader_max_tokens': 2048,
        'query_image_questions': query_image_questions, 'original_pixels_verified': True,
        'independent_model_agnostic_open_clip_rankings_identical': True,
        'comparison_manifest_sha256': sha(source / 'reader_manifest.jsonl'),
        'manifest_sha256': sha(root / 'reader_manifest.jsonl'), 'protocol_sha256': sha(root / 'protocol.json'),
        'wires': wires, 'gold_accessed': False}
    out = root / 'independent_input_audit.json'
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if out.exists(): assert out.read_text() == text
    out.write_text(text)
    print(json.dumps({k: v for k,v in result.items() if k != 'wires'}))
if __name__ == '__main__': main()
