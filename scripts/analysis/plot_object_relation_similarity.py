"""Compare Brand caption and object/relation text similarities on the same images."""

import csv
import json
import os
from pathlib import Path
import re

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
TASK = 'Brand_Memory_Test'
OUT = ROOT / 'results/object_relation_memory'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def natural_key(name):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r'(\d+)', name)]


def describe(values):
    return {'pairs': int(values.size), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'min': float(values.min()),
            'max': float(values.max()), 'std': float(values.std())}


def save_heatmap(matrix, labels, groups, title, path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(11.7, 11), layout='constrained')
    heatmap = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.tick_params(length=0, pad=5)
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color='white', linewidth=1.1, alpha=0.85)
            ax.axvline(i - 0.5, color='white', linewidth=1.1, alpha=0.85)
    ax.set_title(title + '\nBrand | 30 images | BGE-small-en-v1.5',
                 loc='left', fontsize=16, pad=16)
    ax.set_xlabel('Same image order and color scale in both heatmaps', labelpad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(heatmap, ax=ax, fraction=0.042, pad=0.025, ticks=np.linspace(0, 1, 6))
    bar.set_label('Cosine similarity', labelpad=10)
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)


def main():
    natural = ROOT / 'results/natural_observation'
    config = json.loads((natural / 'protocol.json').read_text())
    captions = {r['id']: r for r in read_jsonl(natural / 'captions.jsonl')}
    memories = [r for r in read_jsonl(natural / 'memories.jsonl') if r['task'] == TASK]
    records = [r for r in read_jsonl(OUT / 'extractions.jsonl')
               if any(ref['task'] == TASK for ref in r['references'])]
    records.sort(key=lambda row: natural_key(row['cache_name']))
    assert len(records) == 30 and len({r['image_id'] for r in records}) == 30
    assert {r['image_id'] for r in records} == {i for m in memories for i in m['images']}
    for row in records:
        assert row['caption'] == captions[row['image_id']]['caption']
        assert captions[row['image_id']]['hit_token_limit'] is False
        assert row['objects'] and row['relations']
        assert all(isinstance(s, str) and s.strip() for s in row['relations'])
    labels = [r['cache_name'] for r in records]
    groups = [label.rsplit('_', 1)[0] for label in labels]
    caption_texts = [r['caption'] for r in records]
    relation_texts = ['Objects: ' + ', '.join(r['objects'])
                      + '\nRelations:\n' + '\n'.join(r['relations']) for r in records]
    texts = caption_texts + relation_texts
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    model = SentenceTransformer(config['retriever_model_path'], device='cpu', local_files_only=True)
    lengths = [len(model.tokenizer.encode(text, truncation=False)) for text in texts]
    assert max(lengths) <= model.max_seq_length, 'Text exceeds encoder limit; do not silently truncate'
    vectors = model.encode(texts, batch_size=16, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    upper = np.triu_indices(len(records), 1)
    same_brand = np.array([groups[i] == groups[j] for i, j in zip(*upper)])
    report = {
        'task': TASK, 'images': len(records), 'memory_records': len(memories),
        'model': 'BAAI/bge-small-en-v1.5', 'model_path': config['retriever_model_path'],
        'dimension': int(vectors.shape[1]), 'seed': config['seed'], 'device': 'cpu',
        'metric': 'cosine similarity of L2-normalized text embeddings',
        'natural_input': 'full original natural caption only',
        'object_relation_input': 'Objects: comma-separated final objects; Relations: newline-separated final relation sentences',
        'query_prefix': '', 'max_seq_length': model.max_seq_length, 'truncated_inputs': 0,
        'display_order': 'brand name then numeric image suffix; filenames are not embedded',
        'color_scale': [0, 1], 'diagonal': 'self similarity, included in plots and excluded from pair statistics',
        'pair_statistics': 'strict upper triangle; descriptive only, pairs share images',
        'scope': 'text semantic similarity; not graph matching or factual accuracy',
        'sources': [str(natural / 'captions.jsonl'), str(natural / 'memories.jsonl'),
                    str(natural / 'protocol.json'), str(OUT / 'extractions.jsonl')],
        'inputs': [{'image': label, 'brand': group, 'caption': cap, 'object_relation_text': rel,
                    'caption_tokens': lengths[i], 'object_relation_tokens': lengths[len(records) + i]}
                   for i, (label, group, cap, rel) in enumerate(zip(labels, groups, caption_texts, relation_texts))],
        'statistics': {},
    }
    for key, title, embedding, token_counts in (
            ('natural_caption', 'Natural captions', vectors[:30], lengths[:30]),
            ('object_relation', 'Objects + relation sentences', vectors[30:], lengths[30:])):
        matrix = np.clip(embedding @ embedding.T, -1, 1)
        assert matrix.shape == (30, 30) and np.isfinite(matrix).all()
        assert np.allclose(matrix, matrix.T, atol=1e-6)
        assert np.allclose(np.diag(matrix), 1, atol=1e-5)
        assert matrix.min() >= 0, 'Negative cosine would fall outside the shared plot scale'
        np.fill_diagonal(matrix, 1)
        prefix = OUT / ('Brand_' + key + '_similarity')
        with prefix.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', *labels])
            writer.writerows([label, *[f'{value:.8f}' for value in row]]
                             for label, row in zip(labels, matrix))
        save_heatmap(matrix, labels, groups, title, prefix.with_suffix('.png'))
        pairs = matrix[upper]
        report['statistics'][key] = {
            'all_pairs': describe(pairs), 'same_brand': describe(pairs[same_brand]),
            'different_brand': describe(pairs[~same_brand]),
            'same_minus_different_brand_mean': float(pairs[same_brand].mean() - pairs[~same_brand].mean()),
            'tokens_min': min(token_counts), 'tokens_max': max(token_counts),
            'matrix': str(prefix.with_suffix('.csv')), 'heatmap': str(prefix.with_suffix('.png'))}
    (OUT / 'Brand_similarity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['statistics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
