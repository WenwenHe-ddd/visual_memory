"""Prepare caption-free MemEye inputs for replacing every generative backbone.

Historical Qwen captions, query-caption expansions, graph assignments, rankings,
and answers never enter the new stage inputs. MCQ options and assistant-authored
candidate templates are preserved; evaluation labels live in a separate file.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'data/independent_visual_inputs'
CANDIDATES = ROOT / 'reports/all_master_answer_query_20261021'
VISUAL = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
REVISION = 'f139f89d08bf0f66986153b5ff010cac24ba383d'


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index(rows, key='id'):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f'Duplicate source key: {key}')
    return result


def sanitize_memory(row):
    """Never propagate old caption, retrieval text, or per-image vectors."""
    return {'id':row['memory_id'], 'image_id':row['image_id'],
            'task':row['memory_id'].split('/', 1)[0], 'dialogue':row['dialogue'],
            'timestamp':row['timestamp'], 'insertion_index':row['insertion_index'],
            'image_sha256':row['sha256'], 'image_path':row['source_path']}


def sanitize_question(row, candidate):
    require(all(candidate[key] == row[key] for key in ('id','question','options')),
            f'Candidate question/options mismatch: {row["id"]}')
    statements = candidate['candidate_statements']
    require(set(statements) == set('ABCD') and all(isinstance(s,str) and s.strip() for s in statements.values()),
            f'Invalid frozen candidate statements: {row["id"]}')
    images = [{'path':str(Path(path).resolve()), 'sha256':sha(path)} for path in row['query_image_paths']]
    return {'id':row['id'], 'task':row['task'], 'question':row['question'], 'options':row['options'],
            'query_images':images, 'candidate_statements':{key:statements[key] for key in 'ABCD'}}


def image_jobs(memories, questions, prompt):
    """Deduplicate by image content across history and query, not memory ID."""
    jobs = {}

    def add(identity, path, role, reference):
        require(sha(path) == identity, f'Image content differs: {path}')
        row = jobs.setdefault(identity, {'id':identity, 'sha256':identity, 'path':path,
                                         'source_paths':[], 'history_memory_ids':[], 'query_question_ids':[]})
        if path not in row['source_paths']:
            row['source_paths'].append(path)
        row[role].append(reference)

    for row in memories:
        add(row['image_sha256'], row['image_path'], 'history_memory_ids', row['id'])
    for row in questions:
        for image in row['query_images']:
            add(image['sha256'], image['path'], 'query_question_ids', row['id'])
    requests = [{'id':row['id'], 'user_text':prompt,
                 'query_images':[{'path':row['path'], 'sha256':row['sha256']}]} for row in jobs.values()]
    return list(jobs.values()), requests


def commit(output, artifacts):
    for name, content in artifacts.items():
        path = output / name
        require(not path.exists() or path.read_bytes() == content, f'Refusing to overwrite changed input: {path}')
    if (output / 'protocol.json').exists():
        require(all((output / name).exists() for name in artifacts), 'Frozen input artifact is missing')
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = output / name
        if not path.exists():
            path.write_bytes(content)


def verify_download(path, dataset, tree):
    relative = path.relative_to(dataset).as_posix()
    metadata_path = dataset / '.cache/huggingface/download' / (relative + '.metadata')
    metadata = metadata_path.read_text().splitlines()
    payload = path.read_bytes()
    item = tree[relative]
    actual = (hashlib.sha256(payload).hexdigest() if 'lfs_sha256' in item else
              hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest())
    expected = item.get('lfs_sha256', item['blob_id'])
    require(actual == expected == metadata[1] and metadata[0] == REVISION and len(payload) == item['size'],
            f'Public benchmark download differs: {path}')
    return metadata_path


def prepare_inputs(output: Path, model: str) -> dict:
    output = Path(output).resolve()
    require(isinstance(model,str) and bool(model.strip()), 'An explicit generative model is required')
    source_paths = [FROZEN/name for name in ('protocol.json','qa_manifest.jsonl','arrivals.jsonl')]
    source_paths += [CANDIDATES/name for name in ('candidates.jsonl','candidate_generation_protocol.json',
                                                 'sanitized_candidate_inputs.json')]
    caption_protocol_path = ROOT / 'results/natural_observation/protocol.json'
    reader_protocol_path = CANDIDATES / 'reader/protocol.json'
    source_paths += [caption_protocol_path, reader_protocol_path, Path(__file__).resolve()]
    frozen_protocol = read(FROZEN/'protocol.json')
    for name in ('qa_manifest.jsonl','arrivals.jsonl'):
        require(sha(FROZEN/name) == frozen_protocol['export_hashes'][name], f'Frozen source changed: {name}')
    arrivals, original_questions = lines(FROZEN/'arrivals.jsonl'), lines(FROZEN/'qa_manifest.jsonl')
    require(len(index(arrivals,'memory_id')) == len(arrivals) == 438, 'Expected all 438 memory leaves')
    require(len(index(original_questions)) == len(original_questions) == 371, 'Expected all 371 questions')
    candidate_protocol = read(CANDIDATES/'candidate_generation_protocol.json')
    require(sha(CANDIDATES/'candidates.jsonl') == candidate_protocol['candidates_sha256'], 'Candidate file hash mismatch')
    require(sha(CANDIDATES/'sanitized_candidate_inputs.json')
            == candidate_protocol['source_sha256']['sanitized_candidate_inputs.json'], 'Candidate input hash mismatch')
    require(candidate_protocol['gold_memory_caption_or_image_inputs_used'] is False
            and candidate_protocol['selection_or_revision_using_gold_or_retrieval_scores'] is False,
            'Frozen candidate generation provenance differs')
    candidates = index(lines(CANDIDATES/'candidates.jsonl'))
    sanitized_candidates = index(read(CANDIDATES/'sanitized_candidate_inputs.json'))
    require(set(candidates) == set(sanitized_candidates) == {q['id'] for q in original_questions},
            'Frozen candidate question coverage differs')
    for q in original_questions:
        require(sanitized_candidates[q['id']] == {key:q[key] for key in ('id','question','options')},
                f'Sanitized candidate source differs: {q["id"]}')
    memories = [sanitize_memory(row) for row in arrivals]
    questions = [sanitize_question(row,candidates[row['id']]) for row in original_questions]
    evaluation = [{key:row[key] for key in ('id','gold_choice','gold_ids','x','y')} for row in original_questions]
    caption_protocol, reader_protocol = read(caption_protocol_path), read(reader_protocol_path)
    prompt = caption_protocol['caption_prompt']
    images, requests = image_jobs(memories,questions,prompt)
    domains = sorted({row['task'] for row in questions})
    require(len(domains) == 8, 'Expected all eight domains')
    dataset = ROOT / 'data/MemEye'
    tree_path = dataset / '.cache/huggingface/trees' / f'{REVISION}.json'
    tree = read(tree_path)['files']
    source_paths += [tree_path,dataset/'README.md']
    source_paths.append(verify_download(dataset/'README.md',dataset,tree))
    raw_mcq_dialogue_differences = []
    basis_by_domain, models = {}, None
    for domain in domains:
        open_path = dataset / 'data/dialog' / f'{domain}_Open.json'
        mcq_path = dataset / 'data/dialog' / f'{domain}.json'
        for path in (open_path,mcq_path):
            source_paths += [path,verify_download(path,dataset,tree)]
        open_doc, mcq_doc = read(open_path),read(mcq_path)
        open_rounds = [(session,turn) for session in open_doc['multi_session_dialogues']
                       for turn in session['dialogues'] if turn.get('input_image')]
        actual = [row for row in memories if row['task'] == domain]
        require(len(actual) == len(open_rounds), f'History round count differs: {domain}')
        mcq_rounds = {turn['round']:turn for session in mcq_doc['multi_session_dialogues'] for turn in session['dialogues']}
        for position,(row,(session,turn)) in enumerate(zip(actual,open_rounds),1):
            require(len(turn['input_image']) == 1, 'Expected one image per frozen memory round')
            mid = f'{domain}/{turn["round"]}'
            require(row['id'] == mid and row['image_id'] == mid+'#image1' and row['insertion_index'] == position,
                    f'History insertion order differs: {mid}')
            require(row['dialogue'] == f'User: {turn["user"]}\nAssistant: {turn["assistant"]}'
                    and row['timestamp'] == session['date'], f'Public Open history differs: {mid}')
            require(Path(row['image_path']).resolve() == (dataset/'data/image'/turn['input_image'][0]).resolve(),
                    f'History image reference differs: {mid}')
            other = mcq_rounds[turn['round']]
            if row['dialogue'] != f'User: {other["user"]}\nAssistant: {other["assistant"]}':
                raw_mcq_dialogue_differences.append(mid)
        raw_questions = index(mcq_doc['human-annotated QAs'],'question_id')
        for q in [row for row in original_questions if row['task'] == domain]:
            original = raw_questions[q['id'].split('/',1)[1]]
            matching = [rotation for rotation in original['options']
                        if {k:rotation[k] for k in 'ABCD'} == q['options']]
            require(original['question'] == q['question'] and len(matching) == 1
                    and matching[0]['answer'] == q['gold_choice'], f'Original MCQ fields differ: {q["id"]}')
            require(list(dict.fromkeys(f'{domain}/{clue}' for clue in original['clue'])) == q['gold_ids'],
                    f'Original clue labels differ: {q["id"]}')
            raw_paths = original.get('question_image',[])
            if isinstance(raw_paths,str):
                raw_paths = [raw_paths]
            require([str((dataset/'data/image'/p).resolve()) for p in raw_paths] == q['query_image_paths'],
                    f'Original query image references differ: {q["id"]}')
        basis = VISUAL/domain/'memory/basis.npy'
        encoder_protocol_path = VISUAL/domain/'protocol.json'
        source_paths += [basis,encoder_protocol_path]
        reference = read(encoder_protocol_path)
        selected = {key:reference[key] for key in ('image_encoder','text_encoder','query_prefix')}
        if models is None:
            models = selected
        else:
            require(selected == models, f'CLIP/BGE models differ across domains: {domain}')
        basis_by_domain[domain] = {'path':str(basis), 'sha256':sha(basis),
                                   'protocol_path':str(encoder_protocol_path), 'protocol_sha256':sha(encoder_protocol_path)}
    unique_paths = sorted({path for image in images for path in image['source_paths']})
    for name in unique_paths:
        path = Path(name)
        source_paths += [path,verify_download(path,dataset,tree)]
    history_hashes = {row['image_sha256'] for row in memories}
    query_hashes = {image['sha256'] for row in questions for image in row['query_images']}
    counts = {'questions':371,'domains':8,'memory_leaves':438,'history_unique_images':len(history_hashes),
              'query_image_questions':sum(bool(row['query_images']) for row in questions),
              'query_image_references':sum(len(row['query_images']) for row in questions),
              'query_unique_images':len(query_hashes),'history_query_overlap_images':len(history_hashes & query_hashes),
              'unique_caption_requests':len(requests),'candidate_statements':sum(len(q['candidate_statements']) for q in questions),
              'old_qwen_captions_reused':0,'old_qwen_query_captions_reused':0,'old_reader_answers_reused':0,
              'old_retrieval_rankings_reused':0,'image_source_paths':len(unique_paths)}
    require((counts['history_unique_images'],counts['query_image_questions'],counts['query_unique_images'],
             counts['history_query_overlap_images'],counts['unique_caption_requests'],counts['candidate_statements'])
            == (433,8,6,2,437,1484), 'Image or candidate coverage differs')
    source_hashes = {str(path.resolve()):sha(path) for path in dict.fromkeys(source_paths)}
    encode = lambda obj:(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    jsonl = lambda rows:''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows).encode()
    artifacts = {'source_memories.jsonl':jsonl(memories),'source_questions.jsonl':jsonl(questions),
                 'evaluation.jsonl':jsonl(evaluation),'images.jsonl':jsonl(images),'api_requests_caption.jsonl':jsonl(requests)}
    protocol = {
        'schema':'memeye_full_backbone_inputs_v1','model':model,'dataset':{'repo':'MemEyeBench/MemEye','revision':REVISION},
        'counts':counts,'caption_prompt':prompt,'caption_prompt_source':str(caption_protocol_path),
        'caption_max_new_tokens':caption_protocol['caption_max_new_tokens'],
        'caption_input_policy':'One original image plus the fixed factual caption instruction; no questions, options, dialogue, or gold.',
        'caption_cache_policy':'One fresh caption per unique image SHA shared between history and query images; scoped to the selected model.',
        'historical_caption_provenance_limit':'The exact generation chain of all old history captions is partly retired; '
            'this run replaces them rather than claiming the old complete generation recipe is reproducible.',
        'reader_system_prompt':reader_protocol['reader_settings']['reader_system_prompt'],
        'reader_system_prompt_source':str(reader_protocol_path),
        'history_source_policy':'Exact public Open-version dialogue/timestamp/order, preserving the prior experiment.',
        'public_open_history_matches':438,'public_mcq_history_text_differences':raw_mcq_dialogue_differences,
        'public_provenance_correction':'Two Home dialogues differ between the public Open and MCQ dataset variants. '
            'They exactly match the downloaded Open data; earlier describing them as local edits was inaccurate.',
        'questions_source_policy':'Exact public MCQ question, original frozen option permutation and original query image bytes.',
        'candidate_policy':'Existing assistant-authored question-and-options hypotheses retained; these were not Qwen generations.',
        'candidate_provenance':candidate_protocol,
        'old_qwen_outputs_used':False,'gold_separated':True,'graph_and_retrieval_must_be_rebuilt':True,
        'per_image_visual_vectors_in_inputs':False,'source_images_copied':False,
        **models,'basis_by_domain':basis_by_domain,
        'source_sha256':source_hashes,
        'artifact_sha256':{name:hashlib.sha256(content).hexdigest() for name,content in artifacts.items()},
    }
    artifacts['protocol.json'] = encode(protocol)
    require(all(sha(path) == expected for path,expected in source_hashes.items()), 'Source changed during preparation')
    commit(output,artifacts)
    return protocol
