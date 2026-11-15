"""Visual memory construction and evaluation for retained independent methods."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.evaluation import mcq_reader as qa_reader


ROOT = Path(__file__).resolve().parents[1]
NATURAL = ROOT / 'results/natural_observation'
BENCHMARK = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
QUERY_CACHE = ROOT / 'results/mcq_fifth_non_target_rank/query_embeddings.npz'
OUTPUT = ROOT / 'results/visual_memory_pipeline'
MODES = ('natural', 'fc_t')
TEXT_MEMORY_METHODS = ('gen_agents', 'memoryos', 'a_mem', 'simplemem')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_modes(mode):
    return MODES if mode == 'all' else (mode,)


def output_dir(args, mode):
    root = args.output_root
    if getattr(args, 'query_image_captions', False):
        root = root / 'query_image_caption'
    return (root / mode if mode == 'fc_t' else root / f'top{args.top_k}' / mode)


def retrieval_queries(questions, natural_root, include_image_captions=False):
    """Return answer-free queries and pinned cached captions; never generate captions."""
    captions = {}
    sources = []
    if include_image_captions and any(q['query_image_paths'] for q in questions):
        path = natural_root / 'captions.jsonl'
        rows = read_jsonl(path)
        captions = {r['id']: r['caption'] for r in rows}
        if len(captions) != len(rows):
            raise ValueError('duplicate cached caption IDs')
        sources.append(path)
    queries, image_captions = {}, {}
    for q in questions:
        selected = []
        if include_image_captions:
            for image_path in q['query_image_paths']:
                path = Path(image_path)
                caption = captions.get(sha(path))
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError('missing cached Qwen query-image caption: ' + str(path))
                selected.append(caption)
                sources.append(path)
        text = q['question']
        if selected:
            # Match A-Mem/SimpleMem retrieval_query's image-caption formatting.
            text += "\nquestion's image:\nimage_caption: " + ' '.join(selected)
        queries[q['id']] = text
        image_captions[q['id']] = selected
    return queries, image_captions, list(dict.fromkeys(sources))


def memory_records(mode, natural_root, memory_order, tasks):
    """Return ordinary caption retrieval and reader evidence in source order."""
    if mode != 'natural':
        raise ValueError('unsupported retrieval mode: ' + mode)
    captions = {r['id']: r['caption'] for r in read_jsonl(natural_root / 'captions.jsonl')}
    records = []
    for memory in read_jsonl(natural_root / 'memories.jsonl'):
        if memory['task'] in tasks:
            text = ' '.join(captions[image] for image in memory['images'])
            records.append((memory['id'], memory['task'], text, text))
    if not records or any(not text.strip() for _, _, text, _ in records):
        raise ValueError('missing memory caption')
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)) or any(mid not in memory_order for mid in ids):
        raise ValueError('invalid memory IDs')
    records.sort(key=lambda r: memory_order[r[0]])
    return records, [natural_root / 'captions.jsonl', natural_root / 'memories.jsonl']


def format_reader(question, selected, mode=None):
    lines = [f'Memory {i}: ' + caption.replace(chr(10), ' ')
             for i, (_, _, caption) in enumerate(selected, 1)]
    heading = 'Retrieved memory captions'
    return (heading + ' (ranked by relevance):\n\n'
            + '\n\n'.join(lines) + '\n\nQuestion:\n' + question['question']
            + '\n\nOptions:\n' + '\n'.join(
                label + '. ' + question['options'][label] for label in 'ABCD')
            + '\n\nAnswer with one letter only.')


def rank_task(query, candidates, vectors, order, top_k):
    query = np.asarray(query, dtype=np.float32)
    query /= np.linalg.norm(query)
    scores = [(mid, float(np.dot(query, vectors[mid]))) for mid in candidates]
    scores.sort(key=lambda row: (-round(row[1], 6), order[row[0]]))
    return scores[:top_k]



def offline(args):
    if args.memory == 'caption_visual_master':
        if getattr(args, 'dataset', 'memeye') == 'mem_gallery':
            from scripts.memory import mem_gallery_master_answer
            mem_gallery_master_answer.offline(args)
            return
        if getattr(args, 'visual_basis', 'pca-2d') == 'random-2d':
            from scripts.memory import caption_visual_master_random_2d
            caption_visual_master_random_2d.offline(args)
            return
        from scripts.memory import caption_visual_master
        caption_visual_master.offline(args)
        return
    if args.memory == 'visual_embedding_graph':
        from scripts.graph.build_visual_embedding_graph import build
        build(args)
        return
    if args.memory == 'object_relation':
        from scripts.memory.object_relation_memory import build
        build(args)
        return
    if args.memory in ('natural', 'fc_t', 'reflexion', *TEXT_MEMORY_METHODS):
        if args.natural_root.resolve() != NATURAL.resolve():
            raise ValueError('the existing natural caption writer uses ' + str(NATURAL))
        needed = [NATURAL / name for name in
                  ('protocol.json', 'images.jsonl', 'memories.jsonl')]
        if not all(path.exists() for path in needed):
            subprocess.run([sys.executable, str(ROOT / 'scripts/data/prepare.py')],
                           check=True, cwd=ROOT)
        images = read_jsonl(NATURAL / 'images.jsonl')
        existing = read_jsonl(NATURAL / 'captions.jsonl') if (NATURAL / 'captions.jsonl').exists() else []
        image_ids = {r['id'] for r in images}
        caption_ids = {r['id'] for r in existing}
        if len(image_ids) != len(images) or len(caption_ids) != len(existing):
            raise ValueError('duplicate image or caption IDs in natural memory')
        if caption_ids - image_ids:
            raise ValueError('natural caption store contains unknown image IDs')
        if existing:
            from scripts.data.generate import signature
            expected_signature = signature(json.loads((NATURAL / 'protocol.json').read_text()))
            if any(row['protocol_signature'] != expected_signature for row in existing):
                raise ValueError('natural captions use another generation configuration')
        if caption_ids != image_ids:
            command = [sys.executable, str(ROOT / 'scripts/data/generate.py'), '--stage', 'caption']
            if args.limit:
                command += ['--limit', str(args.limit)]
            subprocess.run(command, check=True, cwd=ROOT)
        else:
            print(f'natural captions already complete: {len(images)} images', flush=True)

def prepare_online(args):
    questions = read_jsonl(args.benchmark_manifest)
    if args.task:
        questions = [q for q in questions if q['task'] == args.task]
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    ids = [q['id'] for q in questions]
    if not questions or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate benchmark question set')
    tasks = {q['task'] for q in questions}
    natural_config = json.loads((args.natural_root / 'protocol.json').read_text())
    baseline_protocol = json.loads((args.benchmark_manifest.parent / 'protocol.json').read_text())
    if 'fc_t' in selected_modes(args.mode):
        from scripts.memory.full_context_memory import prepare_full_context
        prepare_full_context(args, questions, natural_config, baseline_protocol)
    if args.mode == 'fc_t':
        return
    memory_order = {m['id']: i for i, m in enumerate(
        read_jsonl(args.natural_root / 'memories.jsonl'))}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(natural_config['retriever_model_path'],
                                device='cpu', local_files_only=True)
    aligned = getattr(args, 'query_image_captions', False)
    query_texts, query_captions, caption_sources = retrieval_queries(questions, args.natural_root, aligned)
    encoded_texts = [natural_config['query_prefix'] + query_texts[q['id']] for q in questions]
    query_lengths = [len(model.tokenizer.encode(text)) for text in encoded_texts]
    if not aligned and args.query_embeddings and args.query_embeddings.exists() and not args.encode_queries:
        cache = np.load(args.query_embeddings)
        queries = dict(zip(cache['query_ids'], cache['query']))
        if not set(ids) <= set(queries):
            raise ValueError('cached query embeddings do not cover benchmark questions')
        query_source = str(args.query_embeddings)
        query_files = [args.query_embeddings]
    else:
        # The legacy cache contains question-only vectors and cannot serve aligned queries.
        encoded = model.encode(encoded_texts, batch_size=32, normalize_embeddings=True)
        queries = dict(zip(ids, encoded))
        query_source = ('freshly encoded question plus cached Qwen query-image captions with BGE prefix' if aligned
                        else 'encoded original question text with natural protocol BGE prefix')
        query_files = []
    for mode in selected_modes(args.mode):
        if mode == 'fc_t':
            continue
        records, memory_files = memory_records(mode, args.natural_root, memory_order, tasks)
        text_by_id = {mid: text for mid, _, text, _ in records}
        reader_text_by_id = {mid: reader_text for mid, _, _, reader_text in records}
        candidates = {task: [mid for mid, owner, _, _ in records if owner == task]
                      for task in tasks}
        required = args.top_k
        if any(len(items) < required for items in candidates.values()):
            raise ValueError(f'{mode}: fewer than Top-{required} memories in a task')
        lengths = [len(model.tokenizer.encode(text)) for text in text_by_id.values()]
        vectors = model.encode(list(text_by_id.values()), batch_size=32,
                               normalize_embeddings=True)
        vectors = dict(zip(text_by_id, vectors))
        rows = []
        for question in questions:
            ranked = rank_task(queries[question['id']], candidates[question['task']],
                               vectors, memory_order, args.top_k)
            selected = [(mid, score, reader_text_by_id[mid]) for mid, score in ranked]
            top = [mid for mid, _, _ in selected]
            reader_text = format_reader(question, selected, mode)
            payload = {'system': baseline_protocol['reader_system_prompt'],
                       'user_text': reader_text,
                       'query_images': [{'path': p, 'sha256': sha(Path(p))}
                                        for p in question['query_image_paths']]}
            payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()
            gold = set(question['gold_ids'])
            rows.append({**{key: question[key] for key in
                            ('id', 'task', 'x', 'y', 'question', 'options',
                             'gold_choice', 'gold_ids', 'query_image_paths')},
                         'top_memory_ids': top,
                         'top_query_scores': [score for _, score, _ in selected],
                         'reader_text': reader_text,
                         'retrieval_query': query_texts[question['id']],
                         'query_image_captions': query_captions[question['id']],
                         'payload_sha256': payload_sha,
                         'any_gold_topk': bool(gold & set(top)),
                         'all_gold_topk': gold <= set(top)})
        files = [ROOT / 'scripts/visual_memory_pipeline.py',
                 ROOT / 'scripts/evaluation/mcq_reader.py',
                 args.natural_root / 'protocol.json',
                 args.natural_root / 'memories.jsonl',
                 args.benchmark_manifest, args.benchmark_manifest.parent / 'protocol.json']
        files += memory_files + query_files + caption_sources
        protocol = {
            'study': 'MemEye two-stage visual memory MCQ',
            'mode': mode, 'retrieval_top_k': args.top_k,
            'questions': len(rows), 'memory_nodes': len(records),
            'retrieval_scope': 'all picture memories in the original question task',
            'retrieval_text': 'ordinary image caption',
            'reader_memory_text': 'ordinary image caption',
            'retriever': natural_config['retriever'],
            'retriever_revision': natural_config['retriever_revision'],
            'query_embedding_source': query_source,
            'query_policy': ('question plus cached Qwen query-image captions; exclude options and gold' if aligned
                             else 'original question text only; exclude options, gold and query images'),
            'query_prefix': natural_config['query_prefix'],
            'query_image_captions_for_retrieval': aligned,
            'query_image_caption_questions': sum(bool(v) for v in query_captions.values()),
            'query_text_max_tokens': max(query_lengths),
            'query_text_truncated_count': sum(n > model.max_seq_length for n in query_lengths),
            'reader_query_image_policy': 'native query images; cached query-image captions are not appended to reader text',
            'reader_model': baseline_protocol['reader_model'],
            'reader_model_path': baseline_protocol['reader_model_path'],
            'reader_system_prompt': baseline_protocol['reader_system_prompt'],
            'seed': baseline_protocol['seed'], 'do_sample': False,
            'dtype': baseline_protocol['dtype'],
            'max_new_tokens': baseline_protocol['max_new_tokens'],
            'batch_size_text': baseline_protocol['batch_size_text'],
            'batch_size_query_images': baseline_protocol['batch_size_query_images'],
            'query_image_max_pixels': baseline_protocol['query_image_max_pixels'],
            'input_hashes': {str(p.resolve()): sha(p) for p in files},
            'retrieval_text_max_tokens': max(lengths),
            'retrieval_text_truncated_count': sum(n > model.max_seq_length for n in lengths),
            'retrieval_embedding_limit': model.max_seq_length,
        }
        signature = hashlib.sha256(json.dumps(protocol, sort_keys=True,
                                              ensure_ascii=False).encode()).hexdigest()
        protocol['experiment_signature'] = signature
        target = output_dir(args, mode)
        target.mkdir(parents=True, exist_ok=True)
        old = read_jsonl(target / 'qa_responses.jsonl') if (target / 'qa_responses.jsonl').exists() else []
        row_by_id = {q['id']: q for q in rows}
        if any(r['experiment_signature'] != signature or
               r['payload_sha256'] != row_by_id[r['id']]['payload_sha256'] for r in old):
            raise ValueError('existing responses have different inputs: ' + str(target))
        qa_reader.dump(target / 'protocol.json', protocol)
        qa_reader.jsonl(target / 'qa_manifest.jsonl', rows)
        print(f'{mode} Top-{args.top_k}: {len(rows)} questions, '
              f'{sum(q["any_gold_topk"] for q in rows)} with any gold memory', flush=True)


def run_online(args):
    for mode in selected_modes(args.mode):
        qa_reader.OUT = output_dir(args, mode)
        qa_reader.run()


def analyze_online(args):
    combined = {'top_k': args.top_k, 'modes': {}}
    paired_questions = None
    for mode in selected_modes(args.mode):
        target = output_dir(args, mode)
        protocol = json.loads((target / 'protocol.json').read_text())
        questions = read_jsonl(target / 'qa_manifest.jsonl')
        responses = read_jsonl(target / 'qa_responses.jsonl')
        if len(responses) != len(questions) or len({r['id'] for r in responses}) != len(questions):
            raise ValueError('incomplete or duplicate QA responses: ' + str(target))
        answers = {r['id']: r for r in responses}
        if {q['id'] for q in questions} != set(answers):
            raise ValueError('QA response ID set mismatch')
        current_questions = {q['id']: (q['task'], q['question'], q['options'],
                                       q['gold_choice'], q['query_image_paths'])
                             for q in questions}
        if paired_questions is not None and current_questions != paired_questions:
            raise ValueError('mode outputs contain different questions or option orders')
        paired_questions = current_questions
        rows = []
        for q in questions:
            a = answers[q['id']]
            if (a['experiment_signature'] != protocol['experiment_signature'] or
                    a['payload_sha256'] != q['payload_sha256']):
                raise ValueError('QA response provenance mismatch: ' + q['id'])
            rows.append({'id': q['id'], 'task': q['task'],
                         'gold_choice': q['gold_choice'],
                         'predicted_choice': a['predicted_choice'],
                         'correct': a['predicted_choice'] == q['gold_choice'],
                         'any_gold_topk': q['any_gold_topk'],
                         'all_gold_topk': q['all_gold_topk']})
        def counts(group):
            return {'n': len(group), 'correct': sum(r['correct'] for r in group),
                    'accuracy': sum(r['correct'] for r in group) / len(group),
                    'any_gold_topk': sum(r['any_gold_topk'] for r in group),
                    'all_gold_topk': sum(r['all_gold_topk'] for r in group)}
        summary = {**counts(rows), 'mode': mode, 'top_k': None if mode == 'fc_t' else args.top_k,
                   'format_errors': sum(a['status'] != 'ok' for a in responses),
                   'token_limit_hits': sum(a['hit_token_limit'] for a in responses),
                   'experiment_signature': protocol['experiment_signature'],
                   'by_task': {task: counts([r for r in rows if r['task'] == task])
                               for task in sorted({r['task'] for r in rows})}}
        if mode == 'fc_t':
            summary['context_truncated_questions'] = sum(bool(q['dropped_round_ids']) for q in questions)
            summary['max_input_tokens'] = max(q['input_tokens'] for q in questions)
            summary['evidence_coverage_note'] = 'gold coverage after FIFO; legacy topk field names do not indicate retrieval'
        qa_reader.dump(target / 'summary.json', summary)
        qa_reader.jsonl(target / 'per_question.jsonl', rows)
        combined['modes'][mode] = summary
    if args.mode == 'all':
        qa_reader.dump(args.output_root / f'top{args.top_k}' / 'comparison.json', combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


def add_caption_visual_arguments(parser, phase):
    """Arguments for the current Home examples; do not load encoder modules."""
    home = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design'
    pilot = ROOT / 'reports/home_caption_ward_visual_20261020'
    output = ROOT / 'results/debug_caption_master_answer'
    group = parser.add_argument_group('caption_visual_master: Home examples or full Mem-Gallery')
    group.add_argument('--dataset', choices=['memeye', 'mem_gallery'], default='memeye',
                       help='caption_visual_master dataset; memeye retains the existing Home debug defaults')
    group.add_argument('--device', default='cuda:0', help='caption_visual_master: CLIP/BGE device (default: cuda:0)')
    group.add_argument('--output', type=Path, default=output / phase, help='caption_visual_master output directory')
    if phase == 'offline':
        group.add_argument('--memory-source', type=Path,
                           help='export an existing domain offline result to a self-describing memory at --output')
        group.add_argument('--input-manifest', type=Path, default=pilot / 'input_manifest.json')
        group.add_argument('--models-protocol', type=Path, default=home / 'protocol.json')
        group.add_argument('--basis', type=Path, default=pilot / 'memory/basis.npy')
        group.add_argument('--visual-basis', choices=['pca-2d', 'random-2d'], default='pca-2d',
                           help='MemEye basis construction: existing PCA file (default) or fixed random orthogonal 2D basis')
        group.add_argument('--basis-seed', type=int,
                           help='random-2d only: nonnegative seed (default: 0); never selects a seed using evaluation results')
        group.add_argument('--image-files', nargs='+', help='MemEye Home example filenames; defaults to eight debug images')
    else:
        group.add_argument('--memory-path', type=Path, help='self-describing memory directory containing memory.json; reads offline_method automatically')
        group.add_argument('--retrieval', choices=['full_scan', 'multi_path'],
                           help='online query strategy, independent of z-score; default follows the stored offline method')
        group.add_argument('--memory-dir', type=Path, default=output / 'offline/memory')
        group.add_argument('--question-source', type=Path, default=home / 'inputs.json')
        group.add_argument('--candidates', type=Path, default=ROOT / 'reports/all_master_answer_query_20261021/candidates.jsonl')
        group.add_argument('--question-ids', nargs='+', help='MemEye Home examples; defaults to Q8/Q10/Q27')
        group.add_argument('--online-weighting', choices=['none', 'z_score', 'positive_variance'],
                           help='online component for natural/caption_visual_master; explicit none uses the same memory/query without correction')
        group.add_argument('--retrieval-scoring', choices=['master_answer_query', 'z_reweight'],
                           default='master_answer_query', help='legacy alias: z_reweight means --online-weighting z_score')
        group.add_argument('--whitening-lambda', type=float, help='z_score/positive_variance: positive ridge lambda (default: 0.1)')
        group.add_argument('--reweight-alpha', type=float, help='online weighting: nonnegative correction strength (default: 1)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='phase', required=True)
    offline_parser = stages.add_parser('offline', help='build natural captions, visual embedding memory or object/relation extractions')
    offline_parser.add_argument('--memory', choices=['clip_caption', 'caption_visual_master', 'visual_embedding_graph', 'natural', 'fc_t', 'reflexion', *TEXT_MEMORY_METHODS, 'object_relation', 'hippomm', 'hippomm_original', 'hippomm_api'], default='natural')
    offline_parser.add_argument('--natural-root', type=Path, default=NATURAL)
    offline_parser.add_argument('--graph-root', type=Path, help='visual embedding output directory')
    offline_parser.add_argument('--task', help='graph or extraction task to build; default all tasks')
    offline_parser.add_argument('--limit', type=int, default=0, help='max images to generate (natural/object_relation) or per graph task; 0 means all')
    offline_parser.add_argument('--from-start', action='store_true', help='rebuild graph or regenerate object_relation images (respecting --limit)')
    offline_parser.add_argument('--continue-on-error', action='store_true', help='object_relation: record an image output error and continue with remaining images')
    offline_parser.add_argument('--max-new-tokens', type=int, default=256)
    offline_parser.add_argument('--caption-max-new-tokens', type=int, default=256)
    offline_parser.add_argument('--extraction-stage', choices=['prepare', 'run'], default='run', help='object_relation: prepare image+caption requests without loading a model, or extract')
    offline_parser.add_argument('--object-relation-root', type=Path, default=ROOT / 'results/object_relation_memory')
    offline_parser.add_argument('--extraction-max-new-tokens', type=int, default=4096)
    offline_parser.add_argument('--visual-encoder', default=str(ROOT / 'models/clip-vit-base-patch32'), help='visual_embedding_graph: pretrained CLIP model directory or Hugging Face ID')
    offline_parser.add_argument('--visual-encoder-revision', help='pin a Hugging Face encoder revision')
    offline_parser.add_argument('--visual-device', default='cpu', help='CLIP inference device, e.g. cpu or cuda:0')
    offline_parser.add_argument('--visual-local-files-only', action='store_true', help='load CLIP only from local files/cache')
    add_caption_visual_arguments(offline_parser, 'offline')
    from scripts.memory.clip_caption import add_arguments as add_clip_caption_arguments
    add_clip_caption_arguments(offline_parser)
    from scripts.memory.hippomm import add_arguments as add_hippomm_arguments
    add_hippomm_arguments(offline_parser, 'offline')
    from scripts.memory.hippomm_original import add_arguments as add_hippomm_original_arguments
    add_hippomm_original_arguments(offline_parser)
    from scripts.memory.hippomm_api import add_arguments as add_hippomm_api_arguments
    add_hippomm_api_arguments(offline_parser)
    online_parser = stages.add_parser('online', help='retrieve and answer benchmark questions')
    online_parser.add_argument('--stage', choices=['prepare', 'run', 'analyze', 'all'], default='prepare')
    online_parser.add_argument('--mode', '--memory', choices=[*MODES, 'clip_caption', 'clip_image', 'clip_image_dialogue', 'caption_visual_master', 'reflexion', *TEXT_MEMORY_METHODS, 'all', 'hippomm', 'hippomm_original', 'hippomm_api'], help='legacy method selector; omit when using a self-describing --memory-path')
    online_parser.add_argument('--clip-image-source', type=Path, help='clip_image_dialogue: completed CLIP image experiment whose retrieval is kept fixed')
    online_parser.add_argument('--top-k', type=int, help='retrieval count (default: 5); A-Mem uses K primary hits plus linked neighbors; omit for fc_t/reflexion/other text cores')
    online_parser.add_argument('--simplemem-backend', choices=['official_text', 'omni_legacy'], default='official_text',
                               help='SimpleMem defaults to the official three-stage text API; omni_legacy selects the historical MemEye core')
    online_parser.add_argument('--simplemem-backbone', choices=['local-qwen', 'gpt-5'], default='local-qwen',
                               help='gpt-5: verified GPT-5 captions, rebuilt official memory, planning/reflection and answers')
    online_parser.add_argument('--simplemem-source-run', type=Path,
                               default=ROOT / 'results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028/top5/simplemem_official_text_context5',
                               help='read-only matched Qwen control for the GPT-5 SimpleMem experiment')
    online_parser.add_argument('--simplemem-caption-source', type=Path,
                               default=ROOT / 'results/memeye_gpt5_seed42_text_methods_20261025/caption_source',
                               help='complete, verified GPT-5 seed42 caption source')
    online_parser.add_argument('--simplemem-context-limit', type=int,
                               help='SimpleMem only: preserve upstream retrieval, then keep exactly N distinct whole memories for answering')
    online_parser.add_argument('--simplemem-preserve-empty', action='store_true',
                               help='with --simplemem-context-limit, preserve originally empty retrievals instead of filling or failing')
    online_parser.add_argument('--simplemem-semantic-top-k', type=int, default=10,
                               help='official SimpleMem: semantic candidates per query, including reflection (default: 10)')
    online_parser.add_argument('--simplemem-context-fusion', choices=['channel_rrf', 'ordered'], default='channel_rrf',
                               help='official SimpleMem: fuse all retrieval channels before the context cap; ordered preserves historical concatenation')
    online_parser.add_argument('--simplemem-reflection', action=argparse.BooleanOptionalAction, default=False,
                               help='official SimpleMem: enable completeness checks and additional semantic searches (default: disabled)')
    online_parser.add_argument('--a-mem-context-limit', type=int, help='A-Mem only: keep link expansion, deduplicate by memory ID, and answer with exactly this many complete notes')
    online_parser.add_argument('--context-budget', type=int, default=128000, help='FC(T) input plus output token budget; FIFO removes oldest rounds')
    online_parser.add_argument('--reflection-max-new-tokens', type=int, default=512)
    online_parser.add_argument('--memory-max-new-tokens', type=int, default=2048, help='cap for unspecified third-party memory LLM outputs')
    online_parser.add_argument('--memory-workers', type=int, default=8, help='independent task states; within-task sequence preserved')
    online_parser.add_argument('--memory-batch-size', type=int, default=8)
    online_parser.add_argument('--memory-device-map', type=Path,
                               help='optional JSON module-to-device map for the local text-memory Qwen model; recorded in the protocol')
    online_parser.add_argument('--reflexion-batch-size', type=int, default=4, help='batch independent tasks; preserve sequential feedback within each task')
    online_parser.add_argument('--task', help='evaluate questions from one task')
    online_parser.add_argument('--limit-questions', type=int, default=0)
    online_parser.add_argument('--natural-root', type=Path, default=NATURAL)
    online_parser.add_argument('--benchmark-manifest', type=Path, default=BENCHMARK)
    online_parser.add_argument('--query-embeddings', type=Path, default=QUERY_CACHE)
    online_parser.add_argument('--encode-queries', action='store_true', help='encode question text with BGE instead of using the query cache')
    online_parser.add_argument('--query-image-captions', action='store_true', help='add cached Qwen query-image captions to retrieval only; re-encode queries and isolate outputs under query_image_caption/')
    online_parser.add_argument('--output-root', type=Path, default=OUTPUT)
    online_parser.add_argument('--reader-backend', choices=['local', 'api'], default='local',
                               help='api: frozen-input MemEye caption_visual_master reader ablation')
    online_parser.add_argument('--api-model', default='gpt-4o-mini')
    online_parser.add_argument('--api-env-file', type=Path, default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    online_parser.add_argument('--api-source-reader', type=Path, default=ROOT / 'reports/all_master_answer_query_20261021/reader')
    online_parser.add_argument('--api-workers', type=int, default=3)
    online_parser.add_argument('--api-smoke', action='store_true', help='run one frozen text payload and one query-image payload, preserving results for full resume')
    add_caption_visual_arguments(online_parser, 'online')
    add_clip_caption_arguments(online_parser)
    add_hippomm_arguments(online_parser, 'online')
    add_hippomm_original_arguments(online_parser)
    add_hippomm_api_arguments(online_parser)
    backbone_parser = stages.add_parser('backbone', help='MemEye: replace all Qwen roles, regenerate captions, rebuild memory/retrieval and answer')
    backbone_parser.add_argument('--memory', choices=['caption_visual_master'], default='caption_visual_master')
    backbone_parser.add_argument('--dataset', choices=['memeye'], default='memeye')
    from scripts.evaluation.memeye_backbone import add_arguments as add_backbone_arguments
    add_backbone_arguments(backbone_parser)
    args = parser.parse_args(argv)
    if args.phase == 'online' and args.memory_path is not None:
        metadata = args.memory_path / 'memory.json'
        if metadata.is_file() and json.loads(metadata.read_text()).get('offline_method') == 'clip_caption':
            if args.mode not in (None, 'clip_caption'):
                parser.error('--memory disagrees with offline_method stored at --memory-path')
            args.mode = 'clip_caption'
    if ((args.phase == 'offline' and args.memory == 'clip_caption')
            or (args.phase == 'online' and args.mode == 'clip_caption')):
        from scripts.memory.clip_caption import pipeline as run_clip_caption
        return run_clip_caption(args, parser)
    if args.phase == 'offline':
        if args.basis_seed is not None and args.visual_basis != 'random-2d':
            parser.error('--basis-seed requires --visual-basis random-2d')
        if args.visual_basis == 'random-2d':
            if args.memory != 'caption_visual_master' or args.dataset != 'memeye':
                parser.error('--visual-basis random-2d currently requires --memory caption_visual_master --dataset memeye')
            if args.memory_source is not None:
                parser.error('random-2d must rebuild arrivals; --memory-source only exports an existing memory')
            if args.basis != ROOT / 'reports/home_caption_ward_visual_20261020/memory/basis.npy':
                parser.error('random-2d generates its basis; omit --basis')
            args.basis_seed = 0 if args.basis_seed is None else args.basis_seed
            if args.basis_seed < 0:
                parser.error('--basis-seed must be nonnegative')
            if args.output == ROOT / 'results/debug_caption_master_answer/offline':
                args.output = (ROOT / 'results/debug_caption_master_answer_random_2d'
                               / f'seed_{args.basis_seed}' / 'offline')
    if ((args.phase == 'offline' and args.memory == 'hippomm_api')
            or (args.phase == 'online' and args.mode == 'hippomm_api')):
        if args.dataset == 'mem_gallery':
            from scripts.memory.hippomm_gallery_api import pipeline as run_hippomm_api
        else:
            from scripts.memory.hippomm_api import pipeline as run_hippomm_api
        return run_hippomm_api(args, parser)
    if ((args.phase == 'offline' and args.memory == 'hippomm_original')
            or (args.phase == 'online' and args.mode == 'hippomm_original')):
        from scripts.memory.hippomm_original import pipeline as run_hippomm_original
        return run_hippomm_original(args, parser)
    if ((args.phase == 'offline' and args.memory == 'hippomm')
            or (args.phase == 'online' and args.mode == 'hippomm')):
        from scripts.memory.hippomm import pipeline as run_hippomm
        return run_hippomm(args, parser)
    if args.phase == 'backbone':
        if args.api_workers < 1 or args.api_limit < 0:
            parser.error('--api-workers must be positive and --api-limit must be nonnegative')
        from scripts.evaluation.memeye_backbone_completion import pipeline as run_full_backbone
        return run_full_backbone(args)
    if args.phase == 'offline' and args.memory_source is not None:
        if args.memory not in ('natural', 'caption_visual_master') or args.dataset != 'memeye':
            parser.error('--memory-source supports natural/caption_visual_master on MemEye')
        if args.task or args.limit or args.from_start or args.image_files:
            parser.error('--memory-source packages a complete existing domain; omit construction filters')
        if args.output == ROOT / 'results/debug_caption_master_answer/offline':
            parser.error('--memory-source requires an explicit fresh --output memory directory')
        from scripts.memory.memory_store import export_store
        manifest = export_store(args.memory_source, args.output, args.memory)
        print(json.dumps({'memory_path': str(args.output.resolve()), 'offline_method': manifest['offline_method'],
                          'leaves': manifest['leaf_count']}, ensure_ascii=False))
        return
    if args.phase == 'online':
        if args.memory_path is not None:
            if args.memory_dir != ROOT / 'results/debug_caption_master_answer/offline/memory':
                parser.error('use --memory-path instead of combining it with legacy --memory-dir')
            from scripts.memory.memory_store import read_manifest
            manifest = read_manifest(args.memory_path)
            if args.mode is not None and args.mode != manifest['offline_method']:
                parser.error('--memory disagrees with offline_method stored at --memory-path')
            args.mode = manifest['offline_method']
            args.memory_dir = args.memory_path
            if args.online_weighting is None:
                args.online_weighting = 'z_score' if args.retrieval_scoring == 'z_reweight' else 'none'
        if args.mode is None:
            parser.error('online requires --memory-path (or legacy --memory)')
        if args.retrieval is not None and args.online_weighting is None:
            args.online_weighting = 'z_score' if args.retrieval_scoring == 'z_reweight' else 'none'
        if args.retrieval_scoring == 'z_reweight':
            if args.online_weighting not in (None, 'z_score'):
                parser.error('--retrieval-scoring z_reweight conflicts with this --online-weighting selection')
            args.online_weighting = 'z_score'
        if args.online_weighting is not None:
            if args.mode not in ('natural', 'caption_visual_master') or args.dataset != 'memeye' or args.reader_backend != 'local':
                parser.error('--online-weighting requires natural/caption_visual_master, --dataset memeye and --reader-backend local')
            if args.stage != 'prepare':
                parser.error('--online-weighting supports --stage prepare (retrieval only); no reader is called')
            args.retrieval = args.retrieval or ('full_scan' if args.mode == 'natural' else 'multi_path')
            if args.retrieval == 'multi_path' and args.mode != 'caption_visual_master':
                parser.error('multi_path requires visual masters; this memory was built with natural')
            if args.task or args.limit_questions or args.query_image_captions or args.encode_queries:
                parser.error('--online-weighting uses the exact --question-source and optional --question-ids; omit legacy query/filter flags')
            if args.online_weighting == 'none' and (args.whitening_lambda is not None or args.reweight_alpha is not None):
                parser.error('ridge/alpha parameters require --online-weighting z_score or positive_variance')
            args.whitening_lambda = .1 if args.whitening_lambda is None else args.whitening_lambda
            args.reweight_alpha = 1. if args.reweight_alpha is None else args.reweight_alpha
            if not math.isfinite(args.whitening_lambda) or args.whitening_lambda <= 0:
                parser.error('--whitening-lambda must be finite and positive')
            if not math.isfinite(args.reweight_alpha) or args.reweight_alpha < 0:
                parser.error('--reweight-alpha must be finite and nonnegative')
            if args.output == ROOT / 'results/debug_caption_master_answer/online':
                args.output = ROOT / 'results/debug_online_retrieval' / args.mode / args.retrieval / args.online_weighting
            if args.memory_dir == ROOT / 'results/debug_caption_master_answer/offline/memory':
                # Component mode defaults to the complete selected domain memory.
                args.memory_dir = args.question_source.parent
                if args.mode == 'caption_visual_master':
                    args.memory_dir = args.memory_dir / 'memory'
        elif args.whitening_lambda is not None or args.reweight_alpha is not None:
            parser.error('--whitening-lambda and --reweight-alpha require --online-weighting z_score or positive_variance')
    if args.phase == 'online' and args.simplemem_backbone == 'gpt-5':
        if args.output_root == OUTPUT:
            parser.error('GPT-5 SimpleMem requires an explicit fresh --output-root')
        from scripts.evaluation.simplemem_gpt5_backbone import pipeline as run_gpt5_simplemem
        return run_gpt5_simplemem(args)
    if args.phase == 'online' and args.reader_backend == 'api':
        if args.mode != 'caption_visual_master' or args.dataset != 'memeye':
            parser.error('--reader-backend api currently requires caption_visual_master on MemEye')
        if args.question_ids or args.task or args.limit_questions:
            parser.error('API reader ablation freezes all 371 questions; omit question filters')
        if args.api_workers < 1:
            parser.error('--api-workers must be positive')
        if args.api_smoke and args.stage not in ('run', 'all'):
            parser.error('--api-smoke requires --stage run or all')
        if args.api_model in ('gpt-5.4', 'gpt-5.4-2026-03-05'):
            args.api_model = 'gpt-5.4-2026-03-05'
        elif args.api_model.startswith('gpt-5'):
            parser.error('The current GPT-5 reader profile supports only gpt-5.4-2026-03-05 with reasoning none')
        if args.output == ROOT / 'results/debug_caption_master_answer/online':
            model_directory = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in args.api_model)
            if args.api_model == 'gpt-5.4-2026-03-05':
                model_directory += '_none'
            if model_directory in ('', '.', '..'):
                parser.error('Invalid API model name')
            args.output = ROOT / 'results/memeye_reader_ablation_20261021' / model_directory
    if args.dataset == 'mem_gallery':
        method = args.memory if args.phase == 'offline' else args.mode
        if method != 'caption_visual_master':
            parser.error('--dataset mem_gallery requires --memory caption_visual_master')
        if args.output == ROOT / 'results/debug_caption_master_answer' / args.phase:
            args.output = ROOT / 'results/mem_gallery_master_answer_20261021'
        if args.phase == 'offline' and (args.image_files or args.task or args.limit or args.from_start):
            parser.error('Mem-Gallery builds the full fixed dataset; omit Home image filters and task/limit/from-start')
        if args.phase == 'online' and (args.question_ids or args.task or args.limit_questions):
            parser.error('Mem-Gallery evaluates all 1711 questions; omit Home question filters')
    else:
        if args.phase == 'offline' and args.image_files is None:
            args.image_files = [*(f'D7_IMG_{i:03}.jpg' for i in range(1, 7)), 'D13_IMG_001.jpg', 'D13_IMG_002.jpg']
        if args.phase == 'online' and args.question_ids is None and args.reader_backend == 'local' and args.online_weighting is None:
            args.question_ids = ['Q8', 'Q10', 'Q27']
    if args.phase == 'offline':
        if args.graph_root is None:
            args.graph_root = ROOT / 'results/visual_embedding_graph'
        if args.limit < 0 or min(args.max_new_tokens, args.caption_max_new_tokens) < 1:
            parser.error('offline limits must be nonnegative and token limits positive')
        if args.extraction_max_new_tokens < 1:
            parser.error('--extraction-max-new-tokens must be positive')
        if args.memory == 'caption_visual_master' and (args.task or args.limit or args.from_start):
            parser.error('caption_visual_master rebuilds the Home examples selected by --image-files; omit --task/--limit/--from-start')
        offline(args)
    else:
        if args.simplemem_semantic_top_k < 1:
            parser.error('--simplemem-semantic-top-k must be positive')
        if args.simplemem_reflection and (args.mode != 'simplemem' or args.simplemem_backend != 'official_text'):
            parser.error('--simplemem-reflection requires --mode simplemem --simplemem-backend official_text')
        if args.simplemem_preserve_empty and (args.mode != 'simplemem' or args.simplemem_context_limit is None):
            parser.error('--simplemem-preserve-empty requires --mode simplemem --simplemem-context-limit')
        if args.simplemem_context_limit is not None and (args.mode != 'simplemem' or args.simplemem_context_limit < 1):
            parser.error('--simplemem-context-limit requires SimpleMem and a positive count')
        if args.a_mem_context_limit is not None and (args.mode != 'a_mem' or args.a_mem_context_limit < 1):
            parser.error('--a-mem-context-limit requires A-Mem and a positive count')
        if args.query_image_captions and (args.mode not in MODES or args.mode == 'fc_t'):
            parser.error('--query-image-captions requires one picture-memory retrieval mode')
        if (args.mode == 'a_mem' or args.mode not in ('fc_t', 'reflexion', *TEXT_MEMORY_METHODS)) and args.top_k is None:
            args.top_k = 5
        if args.top_k is not None and args.top_k < 1 or args.limit_questions < 0:
            parser.error('--top-k must be positive and --limit-questions nonnegative')
        if args.mode != 'a_mem' and args.mode in ('fc_t', 'reflexion', *TEXT_MEMORY_METHODS) and (args.top_k is not None):
            parser.error('omit --top-k: fc_t/reflexion have no retrieval; text cores other than A-Mem use their pinned method defaults')
        if args.context_budget < 1:
            parser.error('--context-budget must be positive')
        if min(args.reflection_max_new_tokens, args.reflexion_batch_size) < 1:
            parser.error('Reflexion limits must be positive')
        if min(args.memory_max_new_tokens, args.memory_workers, args.memory_batch_size) < 1:
            parser.error('memory limits must be positive')
        if args.online_weighting is not None:
            from scripts.memory import online_retrieval
            online_retrieval.online(args) # natural retrieval with optional z-score reweighting; no reader is called
            return
        if args.mode == 'caption_visual_master':
            if args.reader_backend == 'api':
                if args.top_k != 5:
                    parser.error('MemEye API reader ablation requires the frozen Top5')
                if args.api_model == 'gpt-5.4-2026-03-05':
                    from scripts.evaluation import memeye_gpt5_reader
                    memeye_gpt5_reader.pipeline(args)
                else:
                    from scripts.evaluation import memeye_api_reader
                    memeye_api_reader.pipeline(args)
                return
            if args.dataset == 'mem_gallery':
                if args.top_k != 5:
                    parser.error('Mem-Gallery paired evaluation currently requires --top-k 5')
                from scripts.memory import mem_gallery_master_answer
                mem_gallery_master_answer.online(args)
                return
            if args.stage != 'prepare':
                parser.error('caption_visual_master currently supports --stage prepare (retrieval only); no reader is called')
            if args.task or args.limit_questions:
                parser.error('caption_visual_master uses --question-source and --question-ids to select Home examples')
            from scripts.memory import caption_visual_master
            caption_visual_master.online(args)
            return
        if args.mode == 'clip_image':
            from scripts.memory import clip_image_memory
            clip_image_memory.pipeline(args)
            return
        if args.mode == 'clip_image_dialogue':
            from scripts.memory import clip_image_dialogue
            clip_image_dialogue.pipeline(args)
            return
        if args.mode in TEXT_MEMORY_METHODS:
            from scripts.memory import text_memory_baselines
            for stage in ('prepare', 'run', 'analyze'):
                if args.stage in (stage, 'all'):
                    getattr(text_memory_baselines, stage)(args)
            return
        if args.mode == 'reflexion':
            from scripts.memory import reflexion_memory
            for stage in ('prepare', 'run', 'analyze'):
                if args.stage in (stage, 'all'):
                    getattr(reflexion_memory, stage)(args)
            return
        if args.stage in ('prepare', 'all'):
            prepare_online(args)
        if args.stage in ('run', 'all'):
            run_online(args)
        if args.stage in ('analyze', 'all'):
            analyze_online(args)

if __name__ == '__main__':
    main()
