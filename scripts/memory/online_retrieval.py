"""Memory objects own query construction/grouping; optional weighting only scores.

Both objects retain the frozen full-scan/195-of-371 dialogue + caption texts.
Online embeddings/PCA are transient. Only scalar diagnostics are saved.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.memory.caption_visual_master import (
    configure_device, load_text_encoder, load_clip_text_encoder, encode_text,
    choose_master_answers, normalize, read_json, write_json, sha,
)
from scripts.memory.online_zscore import ZScoreWeighting
from scripts.memory.online_positive_variance import PositiveVarianceWeighting
from scripts.memory.memory_store import read_manifest, leaf_truncations
from scripts.memory.result_method_annotations import online_method_annotations


def make_weighting(name, ridge, alpha):
    """Select an independent score component; unknown names must not disable it."""
    if name == 'none':
        return None
    if name == 'z_score':
        return ZScoreWeighting(ridge, alpha)
    if name == 'positive_variance':
        return PositiveVarianceWeighting(ridge, alpha)
    raise ValueError(f'Unknown online weighting: {name}')


@dataclass(frozen=True)
class RetrievalGroup:
    id: str
    indices: tuple[int, ...]
    query_index: int
    details: dict


class NaturalCaptionMemory:
    # 【184/371 全量基线 · 1. 候选记忆结构】
    # 当前 domain 的全部图片记忆；每条为该轮 User/Assistant 对话 + 完整 Qwen caption。
    # 8 个 domain 合计 438 条，独立检索；Home 有 89 条。对话位于记忆侧，不拼进 query。
    # natural 不需要 master；全量基线配置为 full_scan + online_weighting=none。
    name = 'natural'
    default_retrieval = 'full_scan'
    group_policy = 'all candidate memories in the current domain form one group'

    def __init__(self, leaves):
        self.leaves = leaves
        self.allowed_leaf_truncations = []
        self.texts = [leaf['dialogue'] + '\nMemory description: ' + leaf['caption'] for leaf in leaves]
        ids = [leaf['image_id'] for leaf in leaves]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Memory must contain nonempty, unique image IDs.')
        for leaf, text in zip(leaves, self.texts):
            if leaf.get('retrieval_text', text) != text:
                raise ValueError('Memory text differs from the dialogue + full caption baseline.')

    def plan(self, question):
        return FullScanRetrieval().plan(question, self)

    def group_definitions(self):
        return [('all_memory', tuple(range(len(self.leaves))))]


class CaptionVisualMasterMemory(NaturalCaptionMemory):
    # 【195/371 多路径方案 · 1. 候选记忆结构】
    # 同 domain 的图片叶子按 caption embedding 经 Ward/silhouette 离线分簇。
    # master 保存成员及聚合视觉向量；叶子保留原始单轮对话 + 完整 Qwen caption。
    # 历史全量为 8 个 domain、438 条叶子、88 个 master；Home 为 89 条叶子、25 个 master。
    # 对应配置：caption_visual_master + multi_path + online_weighting=none。
    name = 'caption_visual_master'
    default_retrieval = 'multi_path'
    group_policy = 'each saved master and all its leaves form one group'

    def __init__(self, leaves, masters, centers, candidates, clip_encoder, tokenizer, clip_limit):
        super().__init__(leaves)
        self.masters, self.centers, self.candidates = masters, centers, candidates
        self.clip_encoder, self.tokenizer, self.clip_limit = clip_encoder, tokenizer, clip_limit
        self.indices = [tuple(i for i, leaf in enumerate(leaves) if leaf['master_id'] == m['id'])
                        for m in masters]
        if len({m['id'] for m in masters}) != len(masters):
            raise ValueError('Duplicate master IDs.')
        for master, indices in zip(masters, self.indices):
            members = [leaves[i]['image_id'] for i in indices]
            if not members or len(members) != master['n'] or set(members) != set(master['member_ids']):
                raise ValueError('Invalid master membership.')
        if sorted(i for indices in self.indices for i in indices) != list(range(len(leaves))):
            raise ValueError('Masters must partition the complete candidate memory.')

    def plan(self, question):
        return MultiPathRetrieval().plan(question, self)

    def group_definitions(self):
        return [(m['id'], indices) for m, indices in zip(self.masters, self.indices)]


class FullScanRetrieval:
    name = 'full_scan'
    query_policy = 'unchanged retrieval_query; no option or candidate statement expansion'

    def plan(self, question, memory):
        # 【184/371 全量基线 · 2. 查询内容】
        # 原问题；若题目带图，retrieval_query 已附该题图片的缓存 caption。
        # 不追加选项、候选答案陈述或历史对话；查询编码前再添加 BGE query_prefix。
        return [question['retrieval_query']], [
            RetrievalGroup(key, indices, 0, {}) for key, indices in memory.group_definitions()]


class MultiPathRetrieval:
    name = 'multi_path'
    query_policy = 'retrieval_query + selected complete candidate statement, as in 195/371'

    def plan(self, question, memory):
        if not isinstance(memory, CaptionVisualMasterMemory):
            raise ValueError('multi_path requires a memory with visual masters.')
        row = memory.candidates[question['id']]
        if any(row[key] != question[key] for key in ('question', 'options')):
            raise ValueError('Candidate statements belong to a different question/options arrangement.')
        statements = row['candidate_statements']
        if set(statements) != set('ABCD') or any(not isinstance(s, str) or not s.strip() for s in statements.values()):
            raise ValueError('Expected four nonempty complete candidate statements.')
        # 【195/371 多路径方案 · 4. 编码方式：CLIP 候选匹配】
        # CLIP ViT-B/32：完整候选陈述的 512 维文本向量，与已保存的 master 视觉均值比较。
        # 两者均 L2 归一化，按余弦最大选择 A/B/C/D；同分取字母在前者，文本上限 77 token。
        # 这里只编码陈述；不会重新编码历史图片，也不把叶子的二维坐标用于检索打分。
        clip_scores, choices = choose_master_answers(
            memory.centers, statements, memory.clip_encoder, memory.tokenizer, memory.clip_limit)
        # 【195/371 多路径方案 · 2. 查询内容】
        # 使用仅由问题和选项构造并冻结的四条完整陈述；每个 master 独立选择其中一句。
        # q_m = 原 retrieval_query（含已有题图 caption）+ "\nCandidate answer: " + 完整陈述。
        # 不是直接拼原始选项短语；陈述不包含 gold，且只用于检索，不进入 reader 上下文。
        # BREAKPOINT: exact 195/371 statement expansion, independent of weighting.
        queries = [question['retrieval_query'] + '\nCandidate answer: ' + statements[key] for key in 'ABCD']
        groups = [RetrievalGroup(master['id'], indices, int(choice), {
            'choice': 'ABCD'[choice], 'selected_statement': statements['ABCD'[choice]],
            'clip_statement_scores': {key: float(clip_scores[i, j]) for j, key in enumerate('ABCD')},
        }) for i, (master, indices, choice) in enumerate(zip(memory.masters, memory.indices, choices))]
        return queries, groups


def encode_memory_text(memory, encoder):
    if not memory.allowed_leaf_truncations:
        return encode_text(encoder, memory.texts)
    lengths = [len(ids) for ids in encoder.tokenizer(memory.texts, truncation=False)['input_ids']]
    actual = [{'id': leaf['image_id'], 'original_tokens': length, 'retained_limit': encoder.max_seq_length,
               'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
              for leaf, text, length in zip(memory.leaves, memory.texts, lengths) if length > encoder.max_seq_length]
    if actual != memory.allowed_leaf_truncations:
        raise ValueError('Leaf truncation differs from the recorded baseline encoding policy.')
    return normalize(encoder.encode(memory.texts, batch_size=16, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False, prompt=''))


def retrieve_question(question, memory, text_encoder, query_prefix, top_k, weighting=None, retrieval=None):
    """Memory, retrieval strategy and optional weighting are separate objects."""
    retrieval = retrieval or (FullScanRetrieval() if memory.default_retrieval == 'full_scan' else MultiPathRetrieval())
    queries, groups = retrieval.plan(question, memory)
    # 【184/371 全量基线 · 4. 编码方式】
    # 使用 memory 元数据固定的 BAAI/bge-small-en-v1.5：384 维，查询/记忆均 L2 归一化。
    # 查询前缀为 "Represent this sentence for searching relevant passages: "；记忆不加前缀。
    # 记忆编码文本为 dialogue + "\nMemory description: " + 完整 caption，batch_size=16。
    # 最大长度 512；历史 Card 的一条 577-token 记忆按已记录策略截断，详见 encode_memory_text。
    # 历史实验复用冻结向量；此在线入口现场编码，已逐题核对原始 Top-5 相同。
    # 【195/371 多路径方案 · 4. 编码方式：BGE 叶子匹配】
    # 同一 BGE-small-en-v1.5 编码各 master 的扩展查询，以及未改写的「对话 + 完整 caption」。
    # 384 维、L2 归一化、查询前缀、长度策略均与 184/371 相同；CLIP 选择不替代 BGE 评分。
    query_vectors = encode_text(text_encoder, queries, query_prefix)
    # Fresh online encoding; these vectors and each group's PCA die with this call.
    leaf_vectors = encode_memory_text(memory, text_encoder)
    raw, weighted = np.empty(len(memory.leaves)), np.empty(len(memory.leaves))
    leaves, group_rows = [None] * len(memory.leaves), []
    for group in groups:
        indices = np.asarray(group.indices, dtype=int)
        query, vectors = query_vectors[group.query_index], leaf_vectors[indices]
        if weighting is None:
            # 【184/371 全量基线 · 3. 检索方式：评分】
            # full_scan 给同 domain 全部候选使用同一个 query，计算归一化向量点积（余弦）。
            # 基线不做 CLIP/master 路由、白化或 z-score；下方 weighting 分支是独立可选扩展。
            # 【195/371 多路径方案 · 3. 检索方式：簇内匹配】
            # multi_path 中每条叶子只用所属 master 选出的 q_m 计算原始 BGE 余弦。
            # 所有 master 都参与，没有先筛少数 master；195/371 原方案不含白化/z-score。
            values = vectors @ query
            result = {'raw': values, 'weighted': values, 'correction': np.zeros(len(values))}
        else:
            result = weighting.apply(query, vectors)
        raw[indices], weighted[indices] = result['raw'], result['weighted']
        group_rows.append({'group_id': group.id, 'size': len(indices),
            'query': queries[group.query_index], **group.details,
            **{key: result[key] for key in ('raw_std', 'white_std', 'positive_variance_std', 'raw_status', 'z_status') if key in result}})
        for j, i in enumerate(indices):
            leaf = memory.leaves[i]
            leaves[i] = {'memory_id': leaf['memory_id'], 'image_id': leaf['image_id'], 'group_id': group.id,
                **{key + '_score': float(result[key][j]) for key in ('raw', 'weighted')},
                'correction': float(result['correction'][j])}
            if 'white' in result:
                leaves[i].update(white_score=float(result['white'][j]), white_z=float(result['z'][j]))
            elif 'positive_variance' in result:
                leaves[i].update(positive_variance_score=float(result['positive_variance'][j]),
                                 positive_variance_z=float(result['z'][j]))
    # 【184/371 全量基线 · 3. 检索方式：选取】
    # 所有候选统一按余弦分数降序取 Top-5；完全同分时保留源记忆顺序。
    # weighting=None 时 raw 与 weighted 相同；开启赋权才按修正后的匹配分数排序。
    # 【195/371 多路径方案 · 3. 检索方式：全局合并】
    # 合并当前 domain 所有 master 下的叶子原始匹配分数，统一取 Top-5，同分按源顺序。
    # CLIP 分数只负责选择陈述，不与 BGE 分数融合；不同 master 可使用不同 q_m。
    # BREAKPOINT: globally rank original cosine plus correction, not z alone.
    orders = {'raw': np.argsort(-raw, kind='stable'), 'weighted': np.argsort(-weighted, kind='stable')}
    # Evaluation labels are only accessed after query selection, scoring and ranking.
    gold = set(question.get('gold_ids', []))
    conditions = []
    for name, order in orders.items():
        scores = raw if name == 'raw' else weighted
        # 【184/371 全量基线 · 5. 回答方式】
        # 按检索排名提供 5 条原始「对话 + 完整 caption」，再拼原问题和 A/B/C/D 选项；
        # 带图问题另传原图给 Qwen3-VL-8B-Instruct。候选答案扩展文本不进入回答上下文。
        # 历史 reader：BF16、seed=42、do_sample=False、max_new_tokens=16，只输出大写选项字母。
        # 实际回答入口：scripts/analysis/evaluate_all_master_answer_reader.py 的 prepare()/run()。
        # 本函数只交付 reader_text，不调用回答模型；184/371 是历史最终问答准确率。
        # 【195/371 多路径方案 · 5. 回答方式】
        # 沿用上面的 Qwen reader 与问题/选项/原图输入；变化是按多路径排名选出的 Top-5 记忆。
        # 完整候选陈述只参与检索，不传给 reader。历史 195/371 是最终 QA accuracy。
        # 当前函数仍仅输出检索结果；开启 z-score 后的 QA 分数必须另行测量。
        top = [{'rank': rank + 1, 'memory_id': leaves[i]['memory_id'], 'image_id': leaves[i]['image_id'],
                'group_id': leaves[i]['group_id'], 'score': float(scores[i]),
                'is_gold': leaves[i]['memory_id'] in gold,
                'dialogue': memory.leaves[i]['dialogue'], 'caption': memory.leaves[i]['caption'],
                'reader_text': memory.texts[i]} for rank, i in enumerate(order[:top_k])]
        conditions.append({'name': name, 'top5': top, 'top5_memory_ids': [t['memory_id'] for t in top],
                           'gold_hits_top5': sum(t['is_gold'] for t in top)})
    return {'question_id': question['id'], 'question': question['question'],
            'options': question.get('options'), 'query_image_paths': question.get('query_image_paths', []),
            'retrieval_query': question['retrieval_query'], 'groups': group_rows,
            'leaf_scores': leaves, 'conditions': conditions,
            'evaluation': {key: question[key] for key in ('gold_ids', 'gold_choice') if key in question},
            'reader_called': False}


def select_questions(source, requested):
    by_id = {q['id']: q for q in source}
    if len(by_id) != len(source):
        raise ValueError('Duplicate source question IDs.')
    if not requested:
        return list(source)
    result = []
    for key in requested:
        matches = [q for qid, q in by_id.items() if qid == key or ('/' not in key and qid.rsplit('/', 1)[-1] == key)]
        if len(matches) != 1:
            raise ValueError(f'Question ID is missing or ambiguous: {key}')
        result.append(matches[0])
    if len({q['id'] for q in result}) != len(result):
        raise ValueError('Duplicate --question-ids entries.')
    return result


def load_memory(args, questions):
    """Natural needs only text memory, never a master graph or a CLIP encoder."""
    state = args.memory_dir.resolve()
    manifest = read_manifest(state) if (state / 'memory.json').exists() else None
    if manifest and args.mode != manifest['offline_method']:
        raise ValueError('Requested memory type differs from stored offline_method.')
    stored_paths = {key: state / ref['path'] for key, ref in manifest['files'].items()} if manifest else {}
    if args.mode == 'natural':
        if manifest:
            paths = [state / 'memory.json', *stored_paths.values()]
            memory = NaturalCaptionMemory(read_json(stored_paths['leaves']))
            metadata = manifest['encoding']
        else:
            paths = [state / 'inputs.json', state / 'protocol.json']
            memory = NaturalCaptionMemory(read_json(paths[0])['leaves'])
            metadata = read_json(paths[1])
        devices = {}
    elif args.mode == 'caption_visual_master':
        if manifest:
            paths = [state / 'memory.json', *stored_paths.values()]
            graph, metadata = read_json(stored_paths['graph']), manifest['encoding']
            sums_path = stored_paths['visual_sums']
        else:
            paths = [state / 'graph.json', state / 'master_sums.npy', state.parent / 'protocol.json']
            graph, metadata = read_json(paths[0]), read_json(paths[2])
            sums_path = paths[1]
        if graph['method'] != 'caption_ward_silhouette_visual':
            raise ValueError('Expected a caption_visual_master memory graph.')
        for name, expected in metadata.get('memory_sha256', {}).items():
            if sha(state / name) != expected:
                raise ValueError(f'Offline memory changed: {name}')
        if (state / 'basis.npy').exists():
            paths.append(state / 'basis.npy')
        nodes, masters = graph['image_nodes'], graph['masters']
        leaves = [{**n, 'image_id': n['id'], 'memory_id': n['id'].rsplit('#image', 1)[0]} for n in nodes]
        if manifest:
            stored_leaves = read_json(stored_paths['leaves'])
            if [{k: leaf[k] for k in ('image_id', 'memory_id', 'dialogue', 'caption')} for leaf in leaves] != stored_leaves:
                raise ValueError('Graph leaves and stored text memory differ.')
        sums = np.load(sums_path, allow_pickle=False)
        for m in masters:
            ref = m['visual_mean_ref']
            if ref['path'] != 'master_sums.npy' or not 0 <= ref['row'] < len(sums) or abs(ref['scale'] - 1 / m['n']) > 1e-15:
                raise ValueError('Invalid saved visual master mean reference.')
        centers = normalize([sums[m['visual_mean_ref']['row']] * m['visual_mean_ref']['scale'] for m in masters])
        candidates, clip, tokenizer, limit, devices = {}, None, None, None, {}
        if getattr(args, 'retrieval', None) != 'full_scan':
            candidate_path = args.candidates.resolve()
            paths.append(candidate_path)
            rows = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
            candidates = {row['id']: row for row in rows}
            if len(candidates) != len(rows) or any(q['id'] not in candidates for q in questions):
                raise ValueError('Candidate statements have duplicate IDs or omit selected questions.')
            clip, tokenizer, limit = load_clip_text_encoder(metadata['image_encoder'], args.device)
            devices = {'clip_text_encoder': str(next(clip.parameters()).device)}
        memory = CaptionVisualMasterMemory(leaves, masters, centers, candidates, clip, tokenizer, limit)
    else:
        raise ValueError('Online weighting supports natural and caption_visual_master memories.')
    domains = {leaf['image_id'].split('/', 1)[0] for leaf in memory.leaves}
    if len(domains) != 1 or any(q['id'].split('/', 1)[0] not in domains for q in questions):
        raise ValueError('Questions and memory must belong to the same domain.')
    if manifest and (manifest['domain'] != next(iter(domains)) or manifest['leaf_count'] != len(memory.leaves)):
        raise ValueError('Memory manifest domain/count differs from its leaves.')
    memory.allowed_leaf_truncations = metadata.get('allowed_leaf_truncations', leaf_truncations(metadata, memory.leaves))
    return memory, metadata, paths, devices


def online(args):
    state, output = args.memory_dir.resolve(), args.output.resolve()
    if output.is_relative_to(state):
        raise ValueError('Keep retrieval output outside the read-only memory directory.')
    source = read_json(args.question_source)
    questions = select_questions(source['questions'], args.question_ids)
    if not questions:
        raise ValueError('No questions selected.')
    configure_device(args)
    memory, metadata, paths, devices = load_memory(args, questions)
    if 'leaves' in source:
        source_leaves = {leaf['image_id']: leaf for leaf in source['leaves']}
        for leaf in memory.leaves:
            if any(source_leaves[leaf['image_id']][key] != leaf[key] for key in ('caption', 'dialogue')):
                raise ValueError('Question-source memory text differs from the loaded memory.')
    paths += [args.question_source.resolve(), Path(__file__), Path(__file__).with_name('online_zscore.py'),
              Path(__file__).with_name('result_method_annotations.py'),
              Path(__file__).with_name('memory_store.py'),
              Path(__file__).with_name('ridge_whitening.py'), Path(__file__).with_name('caption_visual_master.py'),
              Path(__file__).parents[1] / 'visual_memory_pipeline.py']
    if args.online_weighting == 'positive_variance':
        paths.append(Path(__file__).with_name('online_positive_variance.py'))
    hashes = {str(p): sha(p) for p in paths}
    text_encoder = load_text_encoder(metadata['text_encoder'], args.device)
    devices['text_encoder'] = str(text_encoder.device)
    weighting = make_weighting(args.online_weighting, args.whitening_lambda, args.reweight_alpha)
    strategy = getattr(args, 'retrieval', None) or memory.default_retrieval # full_scan or multi_path
    retrieval = FullScanRetrieval() if strategy == 'full_scan' else MultiPathRetrieval()
    results = []
    for q in questions:
        result = retrieve_question(q, memory, text_encoder, metadata['query_prefix'], args.top_k, weighting, retrieval)
        results.append(result)
        print(json.dumps({'question': q['id'], 'memory': memory.name, 'weighting': args.online_weighting,
                          'top5': result['conditions'][1]['top5_memory_ids']}, ensure_ascii=False), flush=True)
    for path, expected in hashes.items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Input changed during retrieval: {path}')
    output.mkdir(parents=True, exist_ok=True)
    report = {'purpose': 'online_retrieval; no QA evaluation', 'memory': memory.name,
        'offline_method': memory.name, 'memory_path': str(state), 'retrieval': retrieval.name,
        'online_weighting': args.online_weighting, 'query_policy': retrieval.query_policy,
        'group_policy': memory.group_policy, 'leaf_text': 'dialogue + newline + Memory description: + full caption',
        'ridge_lambda': weighting.ridge if weighting else None, 'alpha': weighting.alpha if weighting else None,
        'formula': ('raw + alpha * std_pop(raw in group) * tanh(positive_variance_z)'
                    if args.online_weighting == 'positive_variance' else
                    'raw + alpha * std_pop(raw in group) * tanh(white_z)' if weighting else 'raw'),
        'memory_dir': str(state), 'images': len(memory.leaves), 'top_k': args.top_k,
        'questions': results, 'source_sha256': hashes, 'memory_unchanged': True,
        'reader_called': False, 'vectors_persisted': False, 'execution_devices': devices,
        'allowed_leaf_truncations': memory.allowed_leaf_truncations}
    if args.online_weighting == 'positive_variance':
        report['intermediate_score_formula'] = 'raw + sum_j [v_j/(v_j+lambda)] * (q dot u_j) * (e_i dot u_j)'
        report['svd_term_sign'] = '+'
    # 五项方法说明随结果保存；不参与查询、检索、排序或回答。
    report['method_annotations'] = online_method_annotations(report, metadata)
    write_json(output / 'retrieval_results.json', report)
    print(f'Online retrieval complete: {output / "retrieval_results.json"}', flush=True)
    return report
