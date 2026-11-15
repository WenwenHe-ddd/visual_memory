"""Human-readable method metadata for retrieval artifacts; never used in scoring."""


def online_method_annotations(report, encoding):
    multi = report['retrieval'] == 'multi_path'
    weighted = report['online_weighting'] == 'z_score'
    positive = report['online_weighting'] == 'positive_variance'
    master = report['offline_method'] == 'caption_visual_master'
    return {
        'schema_version': 1,
        'scope': '本文件当前运行；raw 为未赋权结果，weighted 为所选赋权设置的结果。',
        '1. 候选记忆结构': {
            '说明': ('当前 domain 的图片记忆按 caption embedding 经 Ward/silhouette 离线分簇；每个 master 保存成员和聚合视觉向量，叶子保存原始单轮对话＋完整 Qwen caption。'
                     if master else '当前 domain 的全部图片记忆；每条是对应轮次的 User/Assistant 对话＋完整 Qwen caption。对话在记忆侧。'),
            'memory_path': report['memory_path'], 'offline_method': report['offline_method'],
            'candidate_count': report['images'],
            'text_format': 'dialogue + "\\nMemory description: " + caption',
            ('positive_variance_group' if positive else 'z_score_group'):
                '每个已有 master 的全部叶子' if master else '同 domain 全部候选组成一个簇',
        },
        '2. 查询内容': {
            '说明': ('原问题（带图时附已有 query-image caption）＋本 master 视觉匹配选中的完整候选答案陈述。'
                     if multi else '原问题；带图时附该题图片已有的 Qwen caption。不追加选项、候选答案陈述或历史对话。'),
            'query_prefix': encoding['query_prefix'],
            'candidate_statements_used': multi,
            **({'candidate_construction': '仅由问题和 A/B/C/D 选项构造并冻结的四条完整陈述；不是原始选项短语，不使用 gold。',
                'expanded_query_format': 'retrieval_query + "\\nCandidate answer: " + selected_complete_statement',
                'candidate_input': 'online --candidates；属于查询输入，不属于离线 memory'} if multi else {}),
        },
        '3. 检索方式': {
            '说明': ('每个 master 用 CLIP 视觉中心选择完整候选陈述，扩展查询后匹配本簇叶子，再合并全局 Top-k。'
                     if multi else '同一个 query 与同 domain 全部候选计算 BGE 余弦相似度，降序取 Top-k。'),
            'retrieval': report['retrieval'], 'top_k': report['top_k'],
            'tie_break': '完全同分时保留源记忆顺序', 'online_weighting': report['online_weighting'],
            'final_score': report['formula'],
            'weighting_note': ('组内 SVD 高方差方向正向增强：h=raw+sum[v/(v+lambda)](q·u)(e·u)；'
                               '对 h 标准化，再以 +alpha*std(raw)*tanh(z_h) 修正原始分数。独立于 z_score，不是白化或 -tanh。'
                               if positive else '组内白化分数标准化后提供修正；仍保留原始文本匹配分数。'
                               if weighted else '不做白化或 z-score；直接使用原始余弦分数。'),
            **({'intermediate_score': report.get('intermediate_score_formula'), 'svd_term_sign': '+'} if positive else {}),
            'lambda': report['ridge_lambda'], 'alpha': report['alpha'],
            **({'master_coverage': '当前 domain 全部 master，覆盖全部叶子；不先裁剪 master，也不分配每簇 Top-k 配额。',
                'clip_score_role': '仅选择每个 master 的候选陈述；不与 BGE 叶子分数融合。',
                'historical_unweighted_condition': 'master_answer_query（195/371；该分数仅属于历史未赋权 QA 实验）'} if multi else {}),
        },
        '4. 编码方式': {
            '说明': 'BGE 编码查询和对话＋完整 caption，二者均 L2 归一化；仅查询加上述前缀。在线生成向量，用完即弃。',
            'model': encoding['text_encoder']['model'],
            'revision': encoding['text_encoder']['revision'],
            'dimension': encoding['text_encoder']['dimension'], 'batch_size': 16,
            'max_seq_length': encoding['text_encoder']['max_seq_length'],
            'allowed_leaf_truncations': report['allowed_leaf_truncations'],
            **({'clip_candidate_encoding': {
                'model': encoding['image_encoder']['model_name'],
                'revision': encoding['image_encoder']['revision'],
                'dimension': encoding['image_encoder']['embedding_dim'], 'token_limit': 77,
                'text_input': '四条完整候选陈述',
                'visual_input': '已保存 master 成员视觉向量的均值，再 L2 归一化',
                'normalization': '候选文本向量和 master 视觉均值均 L2 归一化',
                'selection': '最大余弦；同分按 A/B/C/D 顺序',
                'old_images_reencoded': False, 'leaf_2d_coordinates_used': False,
            }} if multi else {}),
        },
        '5. 回答方式': {
            '说明': '本文件仅完成检索，提供按排名排列的原始对话＋完整 caption（reader_text）；没有调用回答模型。',
            'reader_called': False, 'measured_qa_accuracy': None,
            'reference_reader_only': {
                '说明': '对应历史 QA 方案把 Top-5 记忆、原问题和 A/B/C/D 选项交给 Qwen3-VL-8B-Instruct；带图题另传原图，只输出选项字母。',
                'dtype': 'bfloat16', 'seed': 42, 'do_sample': False, 'max_new_tokens': 16,
                'candidate_statements_in_reader': False,
                'note': '这是历史回答方式说明，不是本次执行或准确率。',
            },
        },
    }
