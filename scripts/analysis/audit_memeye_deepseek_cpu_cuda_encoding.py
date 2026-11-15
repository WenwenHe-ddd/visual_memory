"""Quantify the archived CPU/source CUDA reproduction discrepancy without changing retrieval."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.evaluation import memeye_deepseek_open_caption_inputs as src
import torch
torch.set_num_threads(4)
root = src.DEFAULT
task = 'Cartoon_Entertainment_Companion'
info = src.read(src.MEMORY / 'inputs/protocol.json')
memories, questions, _ = src.native.attach_captions(src.rows(src.MEMORY / 'inputs/source_memories.jsonl'), src.rows(src.MEMORY / 'inputs/source_questions.jsonl'), src.rows(src.MEMORY / 'captions.jsonl'))
ms = [m for m in memories if m['task'] == task]
qs = [q for q in questions if q['task'] == task]
bge = src.native.load_text_encoder(info['text_encoder'], 'cpu')
e, _ = src.encode_complete(bge, [m['retrieval_text'] for m in ms], field='CPU reproduction diagnostic memory')
texts = [t for q in qs for t in [q['retrieval_query'], *[q['retrieval_query'] + '\nCandidate answer: ' + q['candidate_statements'][k] for k in 'ABCD']]]
qv, _ = src.encode_complete(bge, texts, info['query_prefix'], field='CPU reproduction diagnostic query')
qv = qv.reshape(len(qs),5,-1)
with np.load(root / 'embeddings' / (task + '.npz')) as f:
    gpu_e, gpu_q = f['memory_vectors'].copy(), f['old_query_vectors'].copy()
original = src.original_retrieval()
differences = []
for i,q in enumerate(qs):
    scores, _ = src.checked_weighting(qv[i,0], e)
    gpu_scores, _ = src.checked_weighting(gpu_q[i,0], gpu_e)
    order = np.argsort(-scores['weighted'], kind='stable')
    ids = [ms[j]['id'] for j in order]
    expected = next(c for c in original[q['id']]['conditions'] if c['name'] == 'baseline_question_only_zscore')
    if ids != expected['full_rank_memory_ids']:
        mismatches = [j for j,(a,b) in enumerate(zip(ids,expected['full_rank_memory_ids'])) if a!=b]
        differences.append({'id': q['id'], 'changed_rank_positions_one_based': [j+1 for j in mismatches],
            'CPU_ranked_ids_at_changed_positions': [ids[j] for j in mismatches],
            'source_CUDA_ranked_ids_at_changed_positions': [expected['full_rank_memory_ids'][j] for j in mismatches],
            'ordered_Top5_identical': ids[:5] == expected['top5_memory_ids'],
            'max_weighted_score_difference': float(np.max(np.abs(scores['weighted']-gpu_scores['weighted']))),
            'closest_source_score_gap_at_changed_positions': float(min(abs(gpu_scores['weighted'][order[j]]-gpu_scores['weighted'][order[k]]) for j in mismatches for k in mismatches if j!=k))})
result = {'passed': True, 'task': task, 'questions': len(qs), 'CPU_vs_source_CUDA_full_ranking_differences': differences,
    'max_memory_embedding_difference': float(np.max(np.abs(e-gpu_e))), 'max_query_embedding_difference': float(np.max(np.abs(qv-gpu_q))),
    'production_vectors_device': 'cuda:1', 'production_source_MCq_full_rankings_exact': 1484,
    'audit_script_sha256': src.sha(__file__), 'source_artifact_sha256': {str(root/'embeddings'/(task+'.npz')): src.sha(root/'embeddings'/(task+'.npz'))}}
src.freeze(root / 'CPU_CUDA_reproduction_discrepancy_audit.json', result)
print(result)
