"""Prepare a pinned, answer-free caption/query manifest and separate scoring gold."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'natural_observation'
OUT.mkdir(parents=True, exist_ok=True)

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def jsonl(name, rows):
    (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def main():
    data = ROOT / 'data/MemEye/data'
    images, memories, queries, gold, files = {}, [], [], [], []
    counts, cells = [], collections.Counter()
    def image_id(rel):
        p = data / 'image' / rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest not in images:
            images[digest] = {'id': digest, 'path': str(p), 'source_paths': [rel]}
        elif rel not in images[digest]['source_paths']:
            images[digest]['source_paths'].append(rel)
        return digest
    for f in sorted((data / 'dialog').glob('*_Open.json')):
        task = f.stem.removesuffix('_Open')
        d = json.loads(f.read_text())
        files.append({'path': str(f.relative_to(ROOT)), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
        task_ids, nrounds, nimage_rounds = set(), 0, 0
        for session in d['multi_session_dialogues']:
            for r in session['dialogues']:
                nrounds += 1
                if not r.get('input_image'):
                    continue
                mid = task + '/' + r['round']
                assert mid not in task_ids
                task_ids.add(mid)
                nimage_rounds += 1
                memories.append({'id': mid, 'task': task, 'round': r['round'],
                                 'session_id': session['session_id'], 'date': session.get('date'),
                                 'images': [image_id(x) for x in r['input_image']]})
        for q in d['human-annotated QAs']:
            qid = task + '/' + q['question_id']
            x, y = q['point'][0][0], q['point'][1][0]
            assert x in ('X1','X2','X3','X4') and y in ('Y1','Y2','Y3')
            qimages = q.get('question_image', [])
            if isinstance(qimages, str): qimages = [qimages]
            queries.append({'id': qid, 'task': task, 'question': q['question'],
                            'images': [image_id(x) for x in qimages]})
            clues = list(dict.fromkeys(task + '/' + c for c in q['clue']))
            assert clues and all(c in task_ids for c in clues), (qid, clues)
            g = {'id': qid, 'task': task, 'x': x, 'y': y, 'clue_ids': clues}
            for field in ('latest_clue_round', 'stale_clue_rounds', 'visual_state_probe_include'):
                if field in q: g[field] = q[field]
            gold.append(g)
            cells[x,y] += 1
        counts.append({'task': task, 'rounds': nrounds, 'image_rounds': nimage_rounds,
                       'questions': len(d['human-annotated QAs'])})
    assert len(queries) == len({q['id'] for q in queries}) == 371
    assert len(memories) == 438
    provenance = json.loads((ROOT/'results/download_provenance.json').read_text())
    config = {
      'study': 'MemEye natural observation of cue sharing and caption retrieval',
      'provenance': provenance, 'source_files': files,
      'caption_model_path': str(ROOT/'models/Qwen3-VL-8B-Instruct'),
      'retriever_model_path': '/storage/hewenwen/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
      'retriever': 'BAAI/bge-small-en-v1.5',
      'retriever_revision': '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
      'query_prefix': 'Represent this sentence for searching relevant passages: ',
      'seed': 42, 'do_sample': False, 'dtype': 'bfloat16',
      'image_max_pixels': 1638400, 'caption_max_new_tokens': 256,
      'cue_max_new_tokens': 768, 'batch_size': 4,
      'caption_prompt': 'Describe this image in one factual English paragraph of about 80 to 120 words. Describe the main scene, visible people and objects, and notable visible details. State only what is visible; do not guess identities, intentions, or facts outside the image. Output only the description.',
      'cue_prompt': '''Extract short retrieval cues explicitly present in the input. Return JSON only: {"cues":[{"category":"scene|person|object|attribute|time","text":"EXACT contiguous span from INPUT"}]}. Use at most 12 cues. Split a complex reference into independently reusable short cues. Never return a whole descriptive clause or long noun phrase. Most cues are 1-3 words; preserve complete proper names.\nScene: a named place or activity setting. Person: a name, role or person category; preserve labels like Player 0. Object: an object head noun, brand or product name. Extract the object head and brand separately when both appear. Attribute: an explicitly stated color, pattern, number or visible text value. Do not extract an unknown value asked for by the question. Time: an explicit date or temporal reference. Do not infer unstated entities. Exclude generic words image, photo, photograph, picture, screenshot, question, information, detail and thing.\nExample INPUT: What color was the mug beside Anna in the kitchen on Monday?\nOutput: {"cues":[{"category":"object","text":"mug"},{"category":"person","text":"Anna"},{"category":"scene","text":"kitchen"},{"category":"time","text":"Monday"}]}\nExample INPUT: What is the text on the red Cedar Coffee bag?\nOutput: {"cues":[{"category":"object","text":"Cedar Coffee"},{"category":"object","text":"bag"},{"category":"attribute","text":"red"}]}\nExtract nouns such as advertisement even if there are no specific named entities. Avoid overlapping duplicates. Do not answer the question.\nINPUT:\n''',
      'memory_unit': 'one image-bearing dialogue round; concatenate captions if multiple images',
      'indexed_text': 'newly generated image captions only; no original dialogue, supplied captions, dates, IDs, gold answers or explanations',
      'retrieval_scope': 'complete history of the same original task; session_id in QA is NOT a query timestamp',
      'query_image_policy': 'append the ordinary caption of the provided query image; flag and report text-only subset separately',
      'counting': {
        'matcher': 'deterministic normalized lexical phrase matching, independent of retrieval scores',
        'primary_categories': ['scene','person','object'],
        'secondary_categories': ['attribute','time'],
        'exclude_from_competitors': 'all annotated clue memories for that question',
        'all_cue_load': 'mean number of other memories matching each primary query cue',
        'supported_cue_load': 'same mean restricted to query cues found in at least one gold caption; undefined if none',
        'gold_coverage': 'fraction of primary query cues found in at least one gold caption; not answer-detail coverage',
        'zero_cue_policy': 'retain question in retrieval results; cue-load correlation undefined',
        'zero_competitor_policy': 'retain retrieval results but omit cue-load correlation when every memory is annotated gold',
      },
      'metrics': ['per_gold_rank','first_gold_rank','mean_gold_rank','mean_gold_rank_fraction',
                  'recall@1','recall@5','recall@10','all_annotated_hit@5','all_annotated_hit@10'],
      'interpretation': 'annotated-evidence retrieval only; all_annotated_hit is coverage, not a guarantee of answering or Y3 reasoning',
      'groupings': ['all','X','Y','X_by_Y','task','text_only_queries'],
      'associations': 'descriptive Spearman; partial rank correlation controls task, X, Y, gold count, primary cue count, gold caption length and cue coverage',
      'uncertainty': 'bootstrap whole task histories; only 8 task clusters; no causal claim or hypothesis-test p values',
      'small_cell_rule': 'report n and estimates; n<10 flagged exploratory',
    }
    dump('protocol.json', config)
    jsonl('images.jsonl', list(images.values()))
    jsonl('memories.jsonl', memories)
    jsonl('queries.jsonl', queries)
    jsonl('gold.jsonl', gold)
    audit = {'tasks': counts, 'questions': len(queries), 'memory_rounds': len(memories),
             'unique_image_contents_including_query_images': len(images),
             'questions_with_image': sum(bool(q['images']) for q in queries),
             'xy_counts': {f'{x}/{y}': n for (x,y),n in sorted(cells.items())},
             'all_clues_map_to_image_rounds': True,
             'note': 'Pinned data labels differ from arXiv v1 Table 5; use actual JSON labels.'}
    dump('data_audit.json', audit)
    print(json.dumps(audit, indent=2))

if __name__ == '__main__': main()
