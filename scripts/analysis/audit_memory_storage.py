"""Measure query-free persisted memory artifacts for the nine MemEye comparison rows.

Never sum result directories: exclude questions, answers, logs, model weights and
replay caches. CLIP's mixed evaluation cache is exported to a history-only index.
GA/A-Mem exports are explicitly snapshots, not complete saved retrieval engines.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024 ** 2


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/memory_storage_20261101')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    canonical_path = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
    canonical = {q['id']: q for q in lines(canonical_path)}
    assert len(canonical) == 371
    tracked, file_cache, components, rows, vector_rows = {}, {}, [], [], []

    def file_info(path):
        path = path.resolve()
        if str(path) not in file_cache:
            data = path.read_bytes()
            file_cache[str(path)] = (len(data), hashlib.sha256(data).hexdigest())
        size, sha = file_cache[str(path)]
        tracked[str(path)] = sha
        return size, sha

    def add_file(method, category, path, note=''):
        size, sha = file_info(path)
        components.append({'method': method, 'category': category, 'bytes': size,
                           'path': str(path.resolve()), 'sha256': sha, 'note': note})

    def accuracy(path, condition=None):
        file_info(path)
        found = []
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if condition is None or record.get('condition') == condition:
                    found.append(record)
        ids = [r.get('question_id', r.get('id')) for r in found]
        assert len(ids) == len(set(ids)) == len(canonical) and set(ids) == set(canonical)
        manifest = None
        if any('gold_choice' not in r for r in found):
            manifest_path = path.parent / 'qa_manifest.jsonl'
            file_info(manifest_path)
            manifest = {r['id']: r for r in lines(manifest_path)}
        for mid, r in zip(ids, found):
            if 'gold_choice' in r:
                assert r['gold_choice'] == canonical[mid]['gold_choice']
            else:
                q = manifest[mid]
                assert q['gold_choice'] == canonical[mid]['gold_choice']
                assert q['options'] == canonical[mid]['options'] and q['question'] == canonical[mid]['question']
                assert r['payload_sha256'] == q['payload_sha256']
            if 'options' in r:
                assert r['options'] == canonical[mid]['options']
        return sum(r['predicted_choice'] == canonical[mid]['gold_choice'] for mid, r in zip(ids, found))

    def row(name, correct, recall, status, note, source, context='5'):
        total = sum(c['bytes'] for c in components if c['method'] == name)
        rows.append({'method': name, 'correct': correct, 'questions': 371,
                     'accuracy': correct / 371, 'recall_at5_micro': recall,
                     'storage_bytes': total, 'storage_mib': total / MIB,
                     'storage_status': status, 'context_count': context,
                     'note': note, 'metric_source': str(source.resolve())})

    def vector(name, count, dimension, itemsize, state, note):
        vector_rows.append({'method': name, 'vectors': count, 'dimension': dimension,
            'bytes_per_element': itemsize, 'logical_bytes': count * dimension * itemsize,
            'logical_mib': count * dimension * itemsize / MIB, 'persistence': state, 'note': note})

    stores = ROOT / 'results/memory_stores_20261023'
    joint = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
    joint_summary = read(joint / 'summary.json')
    file_info(joint / 'summary.json')
    spec = [
        ('Semantic RAG (baseline)', 'baseline_question_only', 'natural', 184),
        ('ours_1', 'master_answer_query', 'caption_visual_master', 195),
        ('ours_2 (λ=0.03, α=1)', 'baseline_question_only__l0p03_a1', 'natural', 185),
        ('ours_1 + ours_2 (λ=0.03, α=1)', 'master_answer_query__l0p03_a1', 'caption_visual_master', 199),
    ]
    from scripts.memory.memory_store import read_manifest
    for name, condition, kind, expected in spec:
        leaves_count, masters = 0, 0
        for directory in sorted(stores.glob('*/' + kind)):
            manifest = read_manifest(directory)
            leaves_count += manifest['leaf_count']
            add_file(name, '文本/结构/配置', directory / 'memory.json')
            for key, value in manifest['files'].items():
                path = directory / value['path']
                category = '持久化向量/索引' if key == 'visual_sums' else '文本/结构/配置'
                note = 'graph.json 与 leaves.json 重复保存的叶子文本按当前文件实计' if key == 'leaves' and kind != 'natural' else ''
                add_file(name, category, path, note)
                if key == 'visual_sums':
                    a = np.load(path)
                    assert a.ndim == 2 and a.shape[1] == 512 and a.dtype == np.float64
                    masters += len(a)
        assert leaves_count == 438
        actual = accuracy(joint / 'reader/responses.jsonl', condition)
        assert actual == expected == joint_summary['overall'][condition]['correct']
        hits = joint_summary['overall'][condition]['gold_hits_top5']
        note = '438 条对话+完整 caption；当前持久化包不含 BGE 叶子向量，查询时重建。'
        if kind != 'natural':
            assert masters == 88
            note += '88 个 master；512 维 float64 聚合视觉向量；保留现有重复叶子文本。'
        if 'l0p03' in condition:
            note += 'z-score/PCA 在线临时计算，无额外持久化记忆。'
        row(name, actual, hits / 1041, '原生持久化记忆包', note, joint / 'summary.json')
        vector(name, 438, 384, 8, '未落盘；不计入 Storage', '当前在线代码的 BGE 叶子矩阵；8 个任务合计，不是峰值 RAM。')
        if masters:
            vector(name, masters, 512, 8, '已计入 Storage', 'master_sums.npy；逻辑数值字节不包含 NPY 文件头。')

    text_root = ROOT / 'results/visual_memory_pipeline'
    for name, subdir, expected in [('Generative Agents (Top10)', 'gen_agents', 174),
                                    ('A-Mem (context 5)', 'top5/a_mem_context5', 195)]:
        directory = text_root / subdir
        protocol = read(directory / 'protocol.json')
        file_info(directory / 'protocol.json')
        count = 0
        for task in protocol['tasks']:
            path = directory / task / 'final_memory.json'
            count += len(read(path))
            add_file(name, '文本/结构/配置', path, '只计主快照一次；排除 memory_source 镜像和 replay 缓存')
        actual = accuracy(directory / 'qa_responses.jsonl')
        assert actual == expected
        is_ga = subdir == 'gen_agents'
        note = f'{count} 条最终记忆快照；原始对话/caption 已在文本中。检索向量未导出，不能把此列视为完整检索引擎存储。'
        if is_ga:
            note += '重要性及最近访问评分张量也未导出；Top10 的 recall 不填入 Recall@5。'
        recall_value = None
        if not is_ga:
            recall_path = ROOT / 'reports/text_memory_recall_20261025/recall_summary.csv'
            file_info(recall_path)
            with recall_path.open() as handle:
                record = next(r for r in csv.DictReader(handle) if r['method'] == 'A-Mem (context 5)')
            assert int(record['qa_correct']) == actual and int(record['gold_hits']) == 360
            recall_value = float(record['recall_micro'])
        row(name, actual, recall_value, '已保存快照；完整索引未落盘',
            note, directory / 'summary.json', '10' if is_ga else '5')
        vector(name, count, 384, 4, '未落盘；不计入 Storage',
               '按 MiniLM 384 维 float32 估算 dense matrix，不含 Python 对象/检索元数据。')
        if is_ga:
            vector(name + ' 时间/重要性张量', count, 2, 4, '未落盘；不计入 Storage',
                   '按两个 float32 向量估算，不包含反思控制器等额外运行状态。')

    name = 'MemoryOS'
    directory = text_root / 'memoryos'
    protocol = read(directory / 'protocol.json')
    file_info(directory / 'protocol.json')
    embedding_shapes = Counter()
    def count_embeddings(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items():
                count_embeddings(v, k)
        elif isinstance(value, list):
            if 'embedding' in key.lower() and value and all(isinstance(v, (int, float)) for v in value):
                embedding_shapes[len(value)] += 1
            else:
                for v in value:
                    count_embeddings(v)
    for task in protocol['tasks']:
        files = sorted((directory / task / 'runtime').rglob('*.json'))
        assert {f.name for f in files} == {'short_term.json', 'mid_term.json', 'long_term_user.json', 'long_term_assistant.json'}
        for path in files:
            add_file(name, '文本及JSON内嵌向量', path)
            count_embeddings(read(path))
    actual = accuracy(directory / 'qa_responses.jsonl')
    assert actual == 177
    row(name, actual, None, '原生持久化分层记忆',
        '短期/中期/用户长期/助手长期 JSON；embedding 以十进制 JSON 数组保存，已包含在 Storage 中。上下文不是严格 Top5。',
        directory / 'summary.json', '动态页数+长期知识')
    for dim, count in embedding_shapes.items():
        vector(name, count, dim, 4, '已在JSON中计入 Storage',
               '仅示意转成 float32 后的数值载荷；实际 JSON 比此数大，不额外相加。')

    name = 'SimpleMem (final context 5)'
    simple_root = ROOT / 'results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028'
    directory = simple_root / 'top5/simplemem_official_text_context5'
    protocol = read(directory / 'protocol.json')
    file_info(directory / 'protocol.json')
    entries = 0
    for task in protocol['tasks']:
        runtime = directory / task / 'runtime'
        for path in sorted((runtime / 'lancedb').rglob('*')):
            if path.is_file():
                category = '数据库/全文索引' if '_indices' in path.parts else '数据库/向量/元数据'
                add_file(name, category, path)
        add_file(name, '文本/结构/配置', runtime / 'resolved_config.json')
        entries += len(read(directory / task / 'final_memory.json'))
    actual = accuracy(directory / 'qa_responses.jsonl')
    recall_path = simple_root / 'source_attribution_20261028/recall_summary.json'
    recall = read(recall_path)
    file_info(recall_path)
    assert actual == 186 == recall['qa_correct'] and entries == 1143
    row(name, actual, recall['recall_at5_micro'], '原生数据库（未压实）',
        '截图对应的官方 SimpleMem：检索 K10 + 反思 + RRF 后限制5条；完整库1143条压缩记忆。计 LanceDB/FTS 及保留的数据库版本；不重复计 final_memory 导出。Recall 为事后保守来源归属估计。',
        directory / 'summary.json')
    vector(name, entries, 384, 4, '已计入数据库 Storage', 'dense embedding 数值载荷；数据库文件另含文本和索引开销。')

    name = 'CLIP → 原图＋历史对话'
    clip = text_root / 'top5/clip_image'
    dialogue = text_root / 'top5/clip_image_dialogue'
    candidates = lines(clip / 'image_candidates.jsonl')
    file_info(clip / 'image_candidates.jsonl')
    file_info(clip / 'clip_embeddings.npz')
    archive = np.load(clip / 'clip_embeddings.npz')
    vectors = archive['image_vectors']
    assert vectors.shape == (433, 512) and vectors.dtype == np.float64
    assert archive['image_ids'].tolist() == [c['image_id'] for c in candidates]
    from scripts.memory.clip_image_dialogue import source_dialogues
    dialogues, sources = source_dialogues(ROOT / 'results/natural_observation', {c['task'] for c in candidates})
    assert len(dialogues) == 438
    for path in sources:
        file_info(path)
    export = out / 'clip_image_dialogue_memory'
    export.mkdir(exist_ok=True)
    shutil.copyfile(clip / 'image_candidates.jsonl', export / 'image_candidates.jsonl')
    np.save(export / 'image_vectors.npy', vectors)
    with (export / 'dialogues.jsonl').open('w') as handle:
        for mid, data in dialogues.items():
            handle.write(json.dumps({'memory_id': mid, **data}, ensure_ascii=False, separators=(',', ':')) + '\n')
    write(export / 'memory.json', {'method': 'clip_image_dialogue', 'image_count': len(candidates),
        'dialogue_rounds': len(dialogues), 'encoder': 'CLIP ViT-B/32', 'vector_dtype': str(vectors.dtype),
        'image_index': 'image_candidates.jsonl', 'image_vectors': 'image_vectors.npy',
        'dialogues': 'dialogues.jsonl', 'image_policy': 'referenced original files; count each content hash once',
        'export_policy': 'history only; excludes query vectors, query images, QA and predictions'})
    for path in sorted(export.iterdir()):
        add_file(name, '持久化向量/索引' if path.suffix == '.npy' else '文本/结构/配置', path,
                 '从已完成实验导出的纯历史记忆文件，未重新编码或生成内容')
    seen = set()
    for candidate in candidates:
        if candidate['image_id'] in seen:
            continue
        path = Path(candidate['path'])
        _, sha = file_info(path)
        assert sha == candidate['image_id']
        seen.add(sha)
        add_file(name, '历史原图', path, '以内容哈希去重；题目图片不额外计入历史库')
    assert len(seen) == 433
    actual = accuracy(dialogue / 'qa_responses.jsonl')
    assert actual == 193
    clip_manifest = lines(clip / 'qa_manifest.jsonl')
    hits = sum(len(set(q['top_memory_ids']) & set(q['gold_ids'])) for q in clip_manifest)
    assert hits == 373
    row(name, actual, hits / 1041, '纯历史记忆导出＋引用原图',
        '433 张去重原图 + 438 轮原文对话 + float64 CLIP 图片向量；图片不复制但完整计其文件字节。检索仍为问题文本→CLIP，回答加入对应对话。',
        dialogue / 'summary.json', '5张图+对应对话')
    vector(name, 433, 512, 8, '已计入 Storage', '只计历史图片矩阵；排除 mixed NPZ 中的371条问题向量。')

    order = [s[0] for s in spec] + ['Generative Agents (Top10)', 'MemoryOS',
        'SimpleMem (final context 5)', 'A-Mem (context 5)', name]
    rows.sort(key=lambda r: order.index(r['method']))
    baseline_bytes = rows[0]['storage_bytes']
    for r in rows:
        r['storage_ratio_to_caption_baseline'] = r['storage_bytes'] / baseline_bytes
        subset = [c for c in components if c['method'] == r['method']]
        assert len(subset) == len({c['path'] for c in subset})
        assert sum(c['bytes'] for c in subset) == r['storage_bytes']
    assert rows[0]['storage_bytes'] == rows[2]['storage_bytes']
    assert rows[1]['storage_bytes'] == rows[3]['storage_bytes']
    notes = [
        '范围：8个任务的完整记忆库，371道题；不是仅取回Top5的上下文大小。',
        '主列 Storage 是未压缩文件的逻辑字节数 stat/读取长度，MiB=bytes/1048576；不是磁盘块分配量，也不是RAM/显存峰值。',
        '排除模型权重、代码、问题/选项/答案、题目专属图片、问题向量、LLM日志、推理缓存、重复结果副本、构建审计与来源注释。历史库本身已有的图片仍计入。',
        'Semantic RAG/ours 使用当前 memory_stores_20261023 原生持久化包；BGE叶子向量在线重建。不把冻结评测 features.npz 当作必要记忆存储。',
        'ours_1 与 ours_1+ours_2 的 graph.json/leaves.json 重复叶子文本仍按实际文件计入；这是当前序列化开销，不是算法所需最小容量。',
        'z-score 的PCA/白化状态在线生成后释放，因此 ours_2 不增加持久化存储；不代表计算成本或运行内存为零。',
        'GA/A-Mem 主列只反映已保存 final_memory 快照，不是完整持久检索引擎大小；dense向量未落盘。GA重要性/最近访问评分状态也未导出。不能据此宣称算法完整存储更低。',
        '未落盘向量页按节点数×维度×dtype估算数值载荷；不是实测RAM，也不包含全部Python对象、PCA工作区、模型或运行峰值。',
        'MemoryOS 原生JSON包含十进制embedding数组，因此文件体积比相同float32矩阵大；本次未改变其格式。',
        'SimpleMem 为截图186/371版本，不是早先198/371的OmniMemory变体。数据库按当前全部文件计入，包括保留的版本/FTS文件；未压实。final_memory是重复导出，排除。',
        'SimpleMem Recall@5=350/1041 为模型辅助、人工复核的保守来源归属估计，不等同原生记录的来源ID。',
        'GA使用Top10；MemoryOS为动态页数加长期知识，故主表Recall@5保持空值。',
        'CLIP导出只拆分已有历史图片向量/索引，并从固定源文件取原文对话；不运行模型。原图只引用不复制，容量仍全额计入。',
        '不同方法的记忆粒度不同：caption/ours为438个含图轮次；GA/A-Mem还写入纯文本轮次；SimpleMem完整库为1143条，来源归属审计的620条不是全库大小。Storage是当前实现的文件测量，不是相同格式的理论压缩率。',
    ]
    write(out / 'storage_summary.json', {'rows': rows, 'notes': notes, 'vector_payloads': vector_rows})
    write(out / 'storage_audit.json', {'files': len(components), 'source_sha256': tracked,
        'all_nine_accuracies_recomputed': True, 'all_component_sums_checked': True,
        'historical_image_hashes_verified': 433, 'clip_dialogue_rounds': 438,
        'clip_images_copied': False, 'original_experiments_modified': False})
    for filename, data in [('storage_summary.csv', rows), ('storage_files.csv', components), ('vector_payloads.csv', vector_rows)]:
        with (out / filename).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Storage对比'
    sheet.append(['MemEye-EM · Qwen3-VL-8B · 371题 · Storage 对比'])
    sheet.merge_cells('A1:K1')
    sheet.append(['Storage：持久化核心文件；GA/A-Mem为已保存快照，详见口径。MiB=2²⁰ bytes。'])
    sheet.merge_cells('A2:K2')
    sheet.append([])
    sheet.append(['方法', '正确/总题数', 'Accuracy', 'Recall@5 (micro)', 'Storage (MiB)', 'Storage (bytes)',
                  '相对caption基线', '计数范围', '统计状态', '备注', '指标来源'])
    for r in rows:
        sheet.append([r['method'], f"{r['correct']}/371", r['accuracy'], r['recall_at5_micro'],
            r['storage_mib'], r['storage_bytes'], r['storage_ratio_to_caption_baseline'],
            r['context_count'], r['storage_status'], r['note'], r['metric_source']])
    for i in range(5, 14):
        for col in ('C', 'D'):
            sheet[f'{col}{i}'].number_format = '0.00%'
        sheet[f'E{i}'].number_format = '0.000'
        sheet[f'F{i}'].number_format = '#,##0'
        sheet[f'G{i}'].number_format = '0.00"×"'
        sheet.row_dimensions[i].height = 66
    widths = [38, 15, 14, 19, 20, 19, 21, 25, 30, 85, 70]

    detail = workbook.create_sheet('容量组成')
    categories = sorted({c['category'] for c in components})
    detail.append(['方法', *[c + ' (bytes)' for c in categories], '合计 (bytes)'])
    for r in rows:
        counts = Counter()
        for c in components:
            if c['method'] == r['method']:
                counts[c['category']] += c['bytes']
        detail.append([r['method'], *[counts[c] for c in categories], r['storage_bytes']])
    vectors_sheet = workbook.create_sheet('向量数值载荷')
    vectors_sheet.append(['方法', '向量数', '维度', '每元素字节', '数值载荷bytes', '数值载荷MiB', '持久化情况', '说明'])
    for r in vector_rows:
        vectors_sheet.append(list(r.values()))
    files_sheet = workbook.create_sheet('逐文件明细')
    files_sheet.append(['方法', '类别', '字节数', '文件路径', 'SHA256', '备注'])
    for c in components:
        files_sheet.append(list(c.values()))
    methodology = workbook.create_sheet('统计口径')
    methodology.append(['编号', '说明'])
    for i, note in enumerate(notes, 1):
        methodology.append([i, note])
        methodology.row_dimensions[i + 1].height = 48

    for ws in workbook:
        header = 4 if ws == sheet else 1
        ws.freeze_panes = f'B{header + 1}'
        ws.auto_filter.ref = f'A{header}:{get_column_letter(ws.max_column)}{ws.max_row}'
        ws.sheet_view.showGridLines = False
        for cell in ws[header]:
            cell.fill = PatternFill('solid', fgColor='183A56')
            cell.font = Font(name='Calibri', color='FFFFFF', bold=True, size=11)
        for row_cells in ws.iter_rows(min_row=header + 1):
            for cell in row_cells:
                cell.font = Font(name='Calibri', size=11)
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if cell.row % 2:
                    cell.fill = PatternFill('solid', fgColor='F1F5F9')
        for col in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(col)].width = 24 if col > 1 else 38
    for i, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(i)].width = width
    sheet['A1'].font = Font(name='Calibri', bold=True, size=16, color='183A56')
    sheet.row_dimensions[1].height = 28
    for cell in sheet[13]:
        cell.fill = PatternFill('solid', fgColor='DDF2E7')
        cell.font = Font(name='Calibri', bold=True)
    files_sheet.column_dimensions['D'].width = 110
    files_sheet.column_dimensions['E'].width = 68
    files_sheet.column_dimensions['F'].width = 65
    methodology.column_dimensions['A'].width = 10
    methodology.column_dimensions['B'].width = 140
    vectors_sheet.column_dimensions['H'].width = 90
    filename = out / 'MemEye_Accuracy_Recall_Storage对比_20261101.xlsx'
    workbook.save(filename)
    verified = load_workbook(filename, read_only=True, data_only=True)
    for i, r in enumerate(rows, 5):
        assert verified['Storage对比'].cell(i, 6).value == r['storage_bytes']
        assert abs(verified['Storage对比'].cell(i, 3).value - r['accuracy']) < 1e-14
    assert verified['Storage对比'].max_row == 13
    print(json.dumps({'workbook': str(filename), 'rows': rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
