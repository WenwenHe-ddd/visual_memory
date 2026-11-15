"""Probe the actual SigLIP2 image-to-image index after a completed M2A run.

This is an exact-image interface smoke test, not a multimodal QA benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.vendor'))
sys.path.insert(0, str(ROOT / '.vendor_m2a'))
sys.path.insert(0, str(ROOT / 'scripts/vendor_m2a/runtime'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def run(source, output):
    import torch
    from PIL import Image
    from pymilvus import MilvusClient
    from agent.stores.semantic import SemanticStore
    from types import SimpleNamespace
    from transformers import AutoModel, AutoProcessor
    if not (source / 'completion_manifest.json').exists():
        raise ValueError('The main experiment must finish and close its database first')
    output.mkdir(parents=True, exist_ok=False)
    model_path = ROOT / 'models/siglip2-base-patch16-384'
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to('cuda:0').eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    memories = json.loads((source / 'semantic_after_qa.json').read_text())
    paths = list(dict.fromkeys(m['image_path'] for m in memories if m['image_path']))[:3]
    if len(paths) < 1:
        raise ValueError('No semantic images to probe')
    db = MilvusClient(str(source / 'semantic.db'))
    for collection in ('memory', 'image_embs'):
        db.load_collection(collection)
    store = SemanticStore.__new__(SemanticStore)
    store.db, store.log = db, []

    def encode(path):
        with Image.open(path) as im:
            inputs = processor(images=[im.convert('RGB')], return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            vec = model.get_image_features(**inputs)
            if not isinstance(vec, torch.Tensor):
                vec = vec.pooler_output
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)[0].cpu().tolist()
        if len(vec) != 768:
            raise ValueError('Unexpected SigLIP2 dimension')
        return vec

    store.multimodal_embedder = SimpleNamespace(encode_image=encode)
    results = []
    try:
        for path in paths:
            matches = store.hybrid_search(query_image_path=path, top_k=3)
            hits = [{'id': h.memory_id, 'image_path': h.image_path} for h in matches]
            results.append({'query_image': path,
                'query_image_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'dimension': 768, 'hits': hits,
                'same_image_in_top3': any(h['image_path'] == path for h in hits),
                'same_image_top1': bool(hits and hits[0]['image_path'] == path)})
    finally:
        db.close()
    report = {'test': 'exact-image SigLIP2 search interface smoke; not image-query QA accuracy',
              'source': str(source), 'queries': len(results),
              'same_image_top1': sum(r['same_image_top1'] for r in results),
              'same_image_top3': sum(r['same_image_in_top3'] for r in results),
              'actual_upstream_hybrid_search_calls': store.log, 'results': results}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    run(a.source, a.output)
