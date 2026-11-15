"""Measure GPT-5 MemEye memory artifacts using the previous Qwen storage policy.

Exports historical GPT-5 captions/graphs into the same online memory schema;
does not run a model, recompute clustering, or modify any experiment source.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/memory_storage_gpt5_memeye_20261102'
FULL = ROOT / 'results/memeye_gpt5_base_seed42_full_20261024'
TEXT = ROOT / 'results/memeye_gpt5_seed42_text_methods_20261025'
SIMPLE = ROOT / 'results/simplemem_official_gpt5_memeye_k10_rrf5_reflection_20261029'
MIB = 1048576


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = {q['id']: q for q in lines(ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl')}
    assert len(canonical) == 371
    components, results, vectors, tracked = [], [], [], {}
    runtime_manifest = read(TEXT/'runtime_artifacts_manifest.json')

    def track(p):
        tracked[str(p.resolve())] = sha(p)
        return tracked[str(p.resolve())]

    def add(method, category, p):
        if p.is_relative_to(TEXT):
            assert sha(p) == runtime_manifest[str(p.relative_to(TEXT))]
        components.append(dict(method=method, category=category, bytes=p.stat().st_size,
                               path=str(p.resolve()), sha256=track(p)))

    def accuracy(p, expected, condition=None):
        track(p)
        records = [r for r in lines(p) if condition is None or r.get('condition') == condition]
        assert len(records) == len({r['id'] for r in records}) == 371
        assert {r['id'] for r in records} == set(canonical)
        for r in records:
            q = canonical[r['id']]
            if 'gold_choice' in r:
                assert r['gold_choice'] == q['gold_choice']
            if 'options' in r:
                assert r['options'] == q['options'] and r['question'] == q['question']
            assert r.get('returned_model') == 'gpt-5-2025-08-07'
        actual = sum(r['predicted_choice'] == canonical[r['id']]['gold_choice'] for r in records)
        assert actual == expected, (p, actual, expected)
        return actual

    def row(method, correct, recall, count, status, note, source):
        files = [c for c in components if c['method'] == method]
        assert len(files) == len({c['path'] for c in files})
        size = sum(c['bytes'] for c in files)
        results.append(dict(method=method, correct=correct, questions=371, accuracy=correct/371,
            recall_at5_micro=recall, storage_bytes=size, storage_mib=size/MIB, memory_count=count,
            storage_status=status, note=note, metric_source=str(source)))

    def vector(method, count, dim, itemsize, persisted, note):
        vectors.append(dict(method=method, count=count, dimension=dim, itemsize=itemsize,
            numerical_bytes=count*dim*itemsize, numerical_mib=count*dim*itemsize/MIB,
            persisted=persisted, note=note))

    # Verify graph leaf content against actual GPT-5 caption output and original dialogue.
    captions = {r['id']:r['caption'] for r in lines(FULL / 'captions.jsonl')}
    memories = {r['image_id']:r for r in lines(FULL / 'inputs/source_memories.jsonl')}
    metadata = read(FULL / 'inputs/protocol.json')
    encoding_audits = {a['task']:a['leaf'] for a in read(FULL/'retrieval_protocol.json')['encoding_audits']}
    evaluation = lines(FULL / 'inputs/evaluation.jsonl')
    assert {r['id'] for r in evaluation} == set(canonical)
    assert all(r['gold_choice'] == canonical[r['id']]['gold_choice'] for r in evaluation)
    summary = read(FULL / 'summary.json')
    assert summary['requested_model'] == 'gpt-5'
    for p in (FULL/'captions.jsonl', FULL/'inputs/source_memories.jsonl', FULL/'inputs/protocol.json',
              FULL/'inputs/evaluation.jsonl', FULL/'summary.json'):
        track(p)
    leaf_count = master_count = 0
    stores = {}
    for domain in sorted((FULL / 'domains').iterdir()):
        if not domain.is_dir():
            continue
        graph_path = domain/'memory/graph.json'
        sums_path = domain/'memory/master_sums.npy'
        graph = read(graph_path)
        build = read(domain/'build_protocol.json')
        for p in (graph_path, sums_path):
            assert track(p) == build['artifact_sha256'][str(p)]
        leaves = []
        for node in graph['image_nodes']:
            original = memories[node['id']]
            assert node['dialogue'] == original['dialogue']
            assert node['caption'] == captions[original['image_sha256']]
            leaves.append(dict(image_id=node['id'], memory_id=original['id'],
                               dialogue=node['dialogue'], caption=node['caption']))
        leaf_count += len(leaves)
        master_count += len(graph['masters'])
        sums = np.load(sums_path)
        assert sums.shape == (len(graph['masters']), 512) and sums.dtype == np.float64
        for kind in ('natural', 'caption_visual_master'):
            dest = OUT/'memory_stores'/domain.name/kind
            write(dest/'leaves.json', leaves)
            files = {'leaves': 'leaves.json'}
            encoding = {k:metadata[k] for k in ('text_encoder','query_prefix')}
            encoding['allowed_leaf_truncations'] = [dict(id=leaf['image_id'],
                original_tokens=record['original_tokens'], retained_limit=record['retained_limit'],
                text_sha256=hashlib.sha256((leaf['dialogue']+'\nMemory description: '+leaf['caption']).encode()).hexdigest())
                for record in encoding_audits[domain.name]['affected_inputs']
                for leaf in leaves if record['id'] in (leaf['memory_id'],leaf['image_id'])]
            sources = {str(p):track(p) for p in (FULL/'inputs/protocol.json', FULL/'inputs/source_memories.jsonl', FULL/'captions.jsonl')}
            if kind == 'caption_visual_master':
                for key, p in (('graph',graph_path), ('visual_sums',sums_path)):
                    shutil.copyfile(p, dest/p.name)
                    files[key] = p.name
                    sources[str(p)] = track(p)
                encoding['image_encoder'] = metadata['image_encoder']
            write(dest/'memory.json', dict(schema_version=1, offline_method=kind, domain=domain.name,
                leaf_count=len(leaves), leaf_text_format='dialogue + newline + Memory description: + full caption',
                files={k:dict(path=name,sha256=sha(dest/name)) for k,name in files.items()},
                encoding=encoding, source_sha256=sources,
                export_policy='existing GPT-5 offline memory; no new clustering; no QA data included'))
            stores.setdefault(kind, []).append(dest)
    assert leaf_count == len(memories) == 438 and len(stores['natural']) == 8
    spec = [('Semantic RAG (baseline)','baseline_question_only','natural'),
            ('ours_1','master_answer_query','caption_visual_master'),
            ('ours_2 (z-score)','baseline_question_only_zscore','natural'),
            ('ours_1 + ours_2','master_answer_query_zscore','caption_visual_master')]
    for method, condition, kind in spec:
        for dest in stores[kind]:
            for p in sorted(dest.iterdir()):
                add(method, '持久化向量/索引' if p.suffix == '.npy' else '文本/结构/配置', p)
        score = summary['overall'][condition]
        correct = accuracy(FULL/'reader/responses.jsonl',score['correct'],condition)
        note = '438条GPT-5完整caption＋原对话；按此前Qwen在线包格式导出，BGE叶子向量在线重建，未计入主列。'
        if kind != 'natural':
            note += f'{master_count}个master；graph/leaves重复文本按实际文件计入；basis.npy仅供离线更新，按此前口径不计。'
            vector(method,master_count,512,8,True,'master_sums.npy中已计入；数值载荷不含NPY头')
        if 'zscore' in condition:
            note += 'z-score在线计算无额外落盘状态；准确率对应λ=0.01、α=1原始完整运行，其他参数不改变本库容量。'
        row(method,correct,score['gold_hits_top5']/1041,438,'同格式历史记忆导出',note,FULL/'summary.json')
        vector(method,438,384,8,False,'未落盘BGE叶子矩阵；不是峰值RAM')

    for method,key,correct in [('Generative Agents (Top10)','gen_agents',177),('A-Mem (context 5)','a_mem',199)]:
        directory = TEXT/key
        protocol = read(directory/'protocol.json')
        assert protocol['model'] == 'gpt-5'
        count = 0
        for task in protocol['tasks']:
            p = directory/task/'final_memory.json'
            add(method,'文本/结构/配置',p)
            count += len(read(p))
        correct = accuracy(directory/'qa_responses.jsonl',correct)
        recall = None
        if key == 'a_mem':
            p = directory/'source_recall_at_5.json'
            track(p)
            recall = read(p)['micro']
        row(method,correct,recall,count,'已保存快照；完整索引未落盘',
            'GPT-5生成的最终记忆快照；检索向量未导出；GA重要性/访问时间检索张量也未导出。',directory/'summary.json')
        vector(method,count,384,4,False,'未落盘检索向量估算；不加进Storage')

    method = 'MemoryOS'
    directory = TEXT/'memoryos'
    protocol = read(directory/'protocol.json')
    shapes = Counter()
    runtime_file_count = 0
    def count_embeddings(value, key=''):
        if isinstance(value,dict):
            for k,v in value.items(): count_embeddings(v,k)
        elif isinstance(value,list):
            if 'embedding' in key.lower() and value and all(isinstance(v,(float,int)) for v in value):
                shapes[len(value)] += 1
            else:
                for v in value: count_embeddings(v)
    for task in protocol['tasks']:
        files = sorted((directory/task/'runtime').rglob('*.json'))
        assert {p.name for p in files} <= {'short_term.json','mid_term.json','long_term_user.json','long_term_assistant.json'}
        expected_paths = {TEXT/p for p in runtime_manifest if p.startswith(f'memoryos/{task}/runtime/')}
        assert set(files) == expected_paths
        runtime_file_count += len(files)
        for p in files:
            add(method,'文本及JSON内嵌向量',p)
            count_embeddings(read(p))
    row(method,accuracy(directory/'qa_responses.jsonl',204),None,None,'原生持久化分层记忆',
        f'{runtime_file_count}个分层JSON；Brand任务原运行仅保存短期/中期，其余任务各4个文件，已核对完成清单；包含十进制embedding数组。',directory/'summary.json')
    for dim,count in shapes.items(): vector(method,count,dim,4,True,'JSON内已包含；float32数值载荷仅供参考，不重复加总')

    method = 'SimpleMem (official, context 5)'
    directory = SIMPLE/'top5/simplemem_official_text_context5'
    protocol = read(directory/'protocol.json')
    count = 0
    for task in protocol['tasks']:
        runtime = directory/task/'runtime'
        for p in sorted((runtime/'lancedb').rglob('*')):
            if p.is_file(): add(method,'数据库/向量/全文索引',p)
        add(method,'文本/结构/配置',runtime/'resolved_config.json')
        count += len(read(directory/task/'final_memory.json'))
    recall_path = SIMPLE/'source_attribution_20261029/recall_summary.json'
    track(recall_path)
    row(method,accuracy(directory/'qa_responses.jsonl',191),read(recall_path)['recall_at5_micro'],count,
        '原生数据库（未压实）','官方SimpleMem，最终5条；计数据库、FTS和保留版本；排除重复final_memory；非196/371旧OmniMemory变体。Recall为来源归属估计。',directory/'summary.json')
    vector(method,count,384,4,True,'已计入数据库；不重复加总')

    method = 'CLIP → 原图＋历史对话'
    previous = ROOT/'reports/memory_storage_20261101'
    with (previous/'storage_files.csv').open(encoding='utf-8-sig') as f:
        original_files = [r for r in csv.DictReader(f) if r['method'] == method]
    for r in original_files:
        p = Path(r['path'])
        assert p.stat().st_size == int(r['bytes']) and sha(p) == r['sha256']
        add(method,r['category'],p)
    directory = ROOT/'results/visual_memory_pipeline/top5/clip_image_dialogue_api/gpt-5'
    manifest = lines(directory/'qa_manifest.jsonl')
    hits = sum(len(set(q['top_memory_ids']) & set(q['gold_ids'])) for q in manifest)
    row(method,accuracy(directory/'qa_responses.jsonl',207),hits/1041,433,
        '复用相同历史原图记忆','与Qwen使用相同433张去重原图、438轮对话、CLIP向量；只换回答模型，容量不变。',directory/'summary.json')
    vector(method,433,512,8,True,'纯历史图片矩阵，不含371条问题向量')

    order = [s[0] for s in spec]+['Generative Agents (Top10)','MemoryOS','SimpleMem (official, context 5)','A-Mem (context 5)',method]
    results.sort(key=lambda r:order.index(r['method']))
    notes = [
        'MemEye，8个任务的全历史库；不是Mem-Gallery，不是单次Top5上下文。MiB=bytes/1048576。',
        '沿用此前Qwen表口径：当前持久化核心文件的逻辑字节；不是RAM/显存，也不是统一格式的算法理论存储。',
        '排除模型权重、代码、问题/选项/答案、问题专属图片及向量、日志、推理缓存、审计和重复导出。',
        'GPT-5自然/图记忆从已完成实验导出到memory_stores；图和caption不重建，原实验只读；与Qwen使用相同4字段叶子及memory.json结构。新增导出配置的路径/哈希元数据也计入，不能解释为原运行目录总大小。',
        'BGE叶子向量在线重建不计入主列；GA/A-Mem只有已保存快照，不含完整检索索引。未落盘向量数值载荷单独列出，不代表实测RAM。',
        'ours图包沿用Qwen统计口径计graph.json、leaves.json、master_sums.npy及memory.json；basis.npy供离线更新，在线评分不需要，未计入。保留重复叶子文本的序列化开销。',
        'z-score参数不改变已保存库；本表准确率用于标识实验，对应GPT-5完整运行的λ=0.01、α=1，不代表其他参数的成绩。',
        'GA Top10及MemoryOS动态上下文不填Recall@5；SimpleMem官方来源归属Recall是估计。',
        'GPT-5 caption/图实验与GPT-5文本方法使用各自已完成运行生成的caption，不假设两轮caption字节相同。',
        'CLIP原图记忆与此前Qwen完全相同，GPT-5仅替换reader；当前正确数207/371。',
    ]
    for r in results:
        assert r['storage_bytes'] == sum(c['bytes'] for c in components if c['method']==r['method'])
    assert results[0]['storage_bytes'] == results[2]['storage_bytes']
    assert results[1]['storage_bytes'] == results[3]['storage_bytes']
    for name,data in [('storage_summary.csv',results),('storage_files.csv',components),('vector_payloads.csv',vectors)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(OUT/'storage_summary.json',dict(rows=results,notes=notes,vector_payloads=vectors,master_count=master_count))
    write(OUT/'storage_audit.json',dict(source_sha256=tracked,files=len(components),questions=371,
        caption_dialogue_leaf_equality_verified=True,leaf_count=438,master_count=master_count,
        source_inputs_unchanged=all(sha(Path(p))==v for p,v in tracked.items())))

    book = Workbook()
    sheet = book.active;sheet.title='GPT-5 Storage对比'
    sheet.append(['方法','正确/总题数','Accuracy','Recall@5 micro','Storage (MiB)','Storage (bytes)','记忆数','统计状态','说明','结果来源'])
    for r in results:
        sheet.append([r['method'],f"{r['correct']}/371",r['accuracy'],r['recall_at5_micro'],r['storage_mib'],r['storage_bytes'],r['memory_count'],r['storage_status'],r['note'],r['metric_source']])
        for col in (3,4):sheet.cell(sheet.max_row,col).number_format='0.00%'
        sheet.cell(sheet.max_row,5).number_format='0.000000'
    detail=book.create_sheet('逐文件明细');detail.append(list(components[0]));
    for r in components:detail.append(list(r.values()))
    v=book.create_sheet('向量数值载荷');v.append(list(vectors[0]));
    for r in vectors:v.append(list(r.values()))
    n=book.create_sheet('统计口径');n.append(['说明'])
    for note in notes:n.append([note])
    for s in book:
        s.freeze_panes='A2';s.auto_filter.ref=s.dimensions
        for cell in s[1]:cell.fill=PatternFill('solid',fgColor='17365D');cell.font=Font(color='FFFFFF',bold=True)
        for col in range(1,s.max_column+1):s.column_dimensions[get_column_letter(col)].width=28
        for r in s.iter_rows(min_row=2):
            for cell in r:cell.alignment=Alignment(vertical='top',wrap_text=True)
    sheet.column_dimensions['I'].width=95;sheet.column_dimensions['J'].width=70
    detail.column_dimensions['D'].width=100;n.column_dimensions['A'].width=150
    for i in range(2,sheet.max_row+1):sheet.row_dimensions[i].height=65
    path=OUT/'MemEye_GPT5_Storage对比_20261102.xlsx';book.save(path)
    check=load_workbook(path,data_only=True)
    for i,r in enumerate(results,2):
        assert check.worksheets[0].cell(i,6).value==r['storage_bytes']
        assert abs(check.worksheets[0].cell(i,5).value-r['storage_mib'])<1e-10
    print(json.dumps([dict(method=r['method'],storage_mib=r['storage_mib'],correct=r['correct'],count=r['memory_count']) for r in results],ensure_ascii=False,indent=2))
    print(path)


if __name__ == '__main__':
    main()
