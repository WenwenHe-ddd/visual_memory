"""Post-generation coverage, context, provenance and official-score audit."""
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation.mem_gallery_qwen_text_methods import (
    OUTPUT, SOURCE, METHODS, NAMES, read, rows, sha, obj, dump, write_rows, verify, native, gallery)


def summarize(items):
    gold = [r for r in items if r['gold_count']]
    total_gold = sum(r['gold_count'] for r in gold)
    return dict(questions=len(items), correct=sum(int(r['em']) for r in items),
        **{key: sum(r[key] for r in items) / len(items) for key in ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')},
        gold_questions=len(gold), gold_links=total_gold,
        source_recall_micro=sum(r['gold_hits'] for r in gold) / total_gold if total_gold else None,
        source_recall_macro=sum(r['gold_hits'] / r['gold_count'] for r in gold) / len(gold) if gold else None,
        source_hit_rate=sum(r['gold_hits'] > 0 for r in gold) / len(gold) if gold else None,
        gold_hits=sum(r['gold_hits'] for r in gold),
        context_count_distribution=dict(sorted(Counter(r['context_count'] for r in items).items())),
        empty_answers=sum(not r['raw_response'] for r in items),
        answer_token_limit_records=sum(r['hit_token_limit'] for r in items),
        unmapped_context_blocks=sum(r['unmapped_context_blocks'] for r in items))


def audit_method(method):
    root = OUTPUT / method
    p = verify(OUTPUT, method)
    qlist = rows(OUTPUT / 'inputs/questions.jsonl')
    questions = {q['id']: q for q in qlist}
    gold = {q['id']: q for q in rows(OUTPUT / 'inputs/gold.jsonl')}
    memories = {m['id']: m for m in rows(OUTPUT / 'inputs/memories.jsonl')}
    observations = read(OUTPUT / 'observations.json')
    assert len(qlist) == len(questions) == len(gold) == 1711
    assert sum(map(len, observations.values())) == 3962
    assert set(questions) == set(gold)
    artifacts = {}

    def tracked(path):
        artifacts[str(path)] = sha(path)
        return read(path)

    def tracked_rows(path):
        artifacts[str(path)] = sha(path)
        return rows(path)

    notes = {}
    helper = native.helpers(method)
    for task, original in observations.items():
        state = tracked(root / task / 'complete.json')
        assert state['experiment_signature'] == p['experiment_signature']
        assert state['retrieval_sha256'] == sha(root / task / 'retrieval.jsonl')
        stats = tracked(root / task / 'memory_stats.json')
        assert stats['stored_rounds'] == len(original)
        for o in original:
            m = memories[task + '/' + o['id']]
            assert m['dialogue'] == 'User: ' + o['user'] + '\nAssistant: ' + o['assistant']
            assert m['timestamp'] == o['timestamp']
            assert o['raw']['image_caption'] == ([m['caption']] if m['is_visual'] else [])
        if method in ('gen_agents', 'a_mem'):
            notes[task] = tracked(root / task / 'final_memory.json')
            if method == 'a_mem':
                assert set(notes[task]) == {o['id'] for o in original}
                for o in original:
                    n = notes[task][o['id']]
                    assert n['content'] == helper['build_a_mem_note_text'](o, o['speaker'], 'assistant')
                    assert n['timestamp'] == o['timestamp']
            else:
                assert {n['dialogue_id'] for n in notes[task] if n.get('source') is False} == {o['id'] for o in original}
                source_notes = {n['dialogue_id']: n for n in notes[task] if n.get('source') is False}
                for o in original:
                    assert source_notes[o['id']]['text'] == helper['_build_round_text'](o, o['speaker'], 'assistant')
    if method == 'simplemem':
        from scripts.evaluation.simplemem_context_ablation import formatter
        simple_render = formatter()
    osmaps = {}
    if method == 'memoryos':
        for task, original in observations.items():
            mapping = defaultdict(list)
            for o in original:
                key = (helper['_round_user_with_captions'](o).strip(), o['assistant'].strip(), o['timestamp'].strip())
                mapping[key].append(task + '/' + o['id'])
            osmaps[task] = mapping
    retrieval = tracked_rows(root / 'retrieval.jsonl')
    answers = tracked_rows(root / 'reader/responses.jsonl')
    manifest = tracked_rows(root / 'reader/manifest.jsonl')
    complete = tracked(root / 'reader/complete.json')
    assert complete['responses_sha256'] == sha(root / 'reader/responses.jsonl')
    assert complete['manifest_sha256'] == sha(root / 'reader/manifest.jsonl')
    rmap, amap, mmap = ({r['id']: r for r in values} for values in (retrieval, answers, manifest))
    assert len(rmap) == len(amap) == len(mmap) == len(retrieval) == len(answers) == len(manifest) == 1711
    assert set(rmap) == set(amap) == set(mmap) == set(questions)
    scored = []
    for q in qlist:
        rid, task = q['id'], q['task']
        r, answer, payload = rmap[rid], amap[rid], mmap[rid]
        assert r['experiment_signature'] == answer['experiment_signature'] == p['experiment_signature']
        assert r['retrieval_query'] == q['retrieval_query']
        assert payload['system'] == gallery.READER_SYSTEM
        assert payload['query_image_paths'] == q['query_image_paths']
        suffix = ('\n\nPlease answer the question about the conversation using the retrieved memories. '
            'Return only the answer content, concisely but completely. For dates or times, '
            'provide a specific date or time whenever possible.\n\nQuestion:\n' + q['question'])
        constraint = gallery.TASK_INSTRUCTIONS.get(q['point'].upper())
        if constraint:
            suffix += '\n\n' + constraint
        assert payload['user_text'] == 'The retrieved memory contents are as follows:\n\n' + r['retrieved_context'] + suffix
        assert obj({k: v for k, v in payload.items() if k != 'payload_sha256'}) == answer['payload_sha256'] == payload['payload_sha256']
        detail, context = r['retrieval_detail'], r['retrieved_context']
        source_ids, unmapped, extras = [], 0, 0
        if method == 'a_mem':
            ids = detail['context_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            chosen = [notes[task][i] for i in ids]
            expected = ''.join('talk start time:' + n['timestamp'] + 'memory content: ' + n['content']
                + 'memory context: ' + n['context'] + 'memory keywords: ' + str(n['keywords'])
                + 'memory tags: ' + str(n['tags']) + '\n' for n in chosen)
            assert expected == context
            source_ids = [task + '/' + i for i in ids]
            count = len(ids)
        elif method == 'simplemem':
            ids = detail['context_memory_ids']
            assert len(ids) == len(set(ids)) <= 5
            assert ids == list(dict.fromkeys(detail['original_item_ids']))[:5]
            assert ids == [i['id'] for i in detail['items']]
            assert simple_render(detail)[0] == context
            for item in detail['items']:
                rounds = [t.removeprefix('round_id:') for t in item['tags'] if t.startswith('round_id:')]
                assert len(rounds) == 1
                source_ids.append(task + '/' + rounds[0])
            count = len(ids)
        elif method == 'gen_agents':
            boundaries = list(re.finditer(r'(?m)^\[(\d+)\] ', context))
            count = len(boundaries)
            assert count <= 10
            for i, boundary in enumerate(boundaries):
                block = context[boundary.end():boundaries[i + 1].start() if i + 1 < count else len(context)].strip()
                candidates = [n for n in notes[task] if n['text'].strip() == block]
                if len(candidates) == 1:
                    n = candidates[0]
                    if n.get('source') is False:
                        source_ids.append(task + '/' + n['dialogue_id'])
                    else:
                        extras += 1
                elif candidates and all(n.get('source') is not False for n in candidates):
                    extras += 1  # Identical repeated reflections; no original-source credit.
                else:
                    unmapped += 1  # Never credit partially rendered or ambiguous sources.
        else:
            assert helper['_format_retrieval_context'](detail) == context
            count = len(detail['retrieved_pages'])
            extras = sum(len(detail[k]) for k in ('retrieved_user_knowledge', 'retrieved_assistant_knowledge'))
            for page in detail['retrieved_pages']:
                key = (page['user_input'].strip(), page['agent_response'].strip(), page['timestamp'].strip())
                candidates = osmaps[task].get(key, [])
                if len(candidates) == 1:
                    source_ids.extend(candidates)
                else:
                    unmapped += 1
        assert all(mid in memories and memories[mid]['task'] == task for mid in source_ids)
        target = gold[rid]
        gold_ids = set(target['gold_ids'])
        scored.append(dict(id=rid, task=task, point=q['point'], answer=target['answer'],
            raw_response=answer['raw_response'], hit_token_limit=answer['hit_token_limit'],
            **gallery.score_answer(answer['raw_response'], target['answer']),
            gold_count=len(gold_ids), gold_hits=len(gold_ids & set(source_ids)),
            source_ids=sorted(set(source_ids)), gold_ids=sorted(gold_ids),
            context_count=count, extra_reflection_or_knowledge_entries=extras, unmapped_context_blocks=unmapped))
    calls, limits, invalid, empty = {}, 0, 0, 0
    for path in sorted((root / 'calls').glob('*.json')):
        call = read(path); request, out = call['request'], call['output']
        assert obj(request) == path.stem
        assert request['experiment_signature'] == p['experiment_signature']
        assert request['seed'] == 42 and request['kind'] == 'memory'
        assert all(isinstance(m['content'], str) for m in request['messages'])
        limits += bool(out['hit_token_limit'])
        invalid += out.get('structured_output_valid') is False
        empty += not out['raw_response'].strip()
        calls[path.name] = sha(path)
    events = tracked_rows(root / 'call_events.jsonl')
    assert all(e['key'] + '.json' in calls and e['kind'] == 'memory' for e in events)
    log_path = OUTPUT / (method + '.log')
    warning_lines = [line for line in log_path.read_text(errors='replace').splitlines()
        if re.search(r'^(?:ERROR\b|WARNING\b|\[(?:ERROR|WARNING)\]|Error\b|Failed\b|Traceback\b|JSON parsing error|(?:ShortTermMemory|LongTermMemory|MidTermMemory): (?:Error|An unexpected error))| - (?:ERROR|WARNING) - ', line)
        and 'Qwen3VLVideoProcessorInitKwargs' not in line]
    dump(root / 'diagnostic_warning_log.json', dict(lines=warning_lines,
        note='This includes native parser/fallback warnings and prior orchestration/audit exceptions; these lines are not a count of failed memory calls.'))
    format_retries = []
    if (root / 'format_continuation_protocol.json').exists():
        continuation = tracked(root / 'format_continuation_protocol.json')
        assert continuation['parent_experiment_signature'] == p['experiment_signature']
        for name, expected in continuation['source_sha256'].items():
            assert sha(name) == expected
            artifacts[name] = expected
        assert tracked(root / 'recovery.json')['status'] == 'complete'
        format_retries = tracked_rows(root / 'format_retry_events.jsonl')
        assert all(1 <= e['attempt'] <= 3 and e['expected'] == 2 for e in format_retries)
    elif (root / 'format_retry_events.jsonl').exists():
        assert p.get('reflection_format_policy')
        format_retries = tracked_rows(root / 'format_retry_events.jsonl')
        assert all(1 <= e['attempt'] <= 3 and e['expected'] == 2 for e in format_retries)
    dump(root / 'calls_manifest.json', calls)
    result = dict(method=method, name=NAMES[method], status='complete',
        overall=summarize(scored),
        by_domain={t: summarize([s for s in scored if s['task'] == t]) for t in sorted(observations)},
        by_question_type={t: summarize([s for s in scored if s['point'] == t]) for t in sorted({s['point'] for s in scored})},
        memory_calls=dict(unique=len(calls), token_limit=limits, schema_invalid=invalid, empty=empty,
                          reflection_format_retry_attempts=len(format_retries), diagnostic_warning_lines=len(warning_lines)),
        source_recall_policy='Identifiable original rounds in actual final context; no inferred credit for reflections/knowledge or ambiguous/truncated blocks. Empty gold sets excluded.',
        strict_context_at_most_five=method in ('simplemem', 'a_mem'), annotations=p['annotations'])
    write_rows(root / 'scored_questions.jsonl', scored)
    dump(root / 'summary.json', result)
    dump(root / 'external_audit.json', dict(status='passed', questions=1711, scenarios=20, memory_rounds=3962,
        final_context_checked=True, reader_payload_checked=True, gold_separate=True,
        memory_calls=result['memory_calls'], annotations=p['annotations'], artifacts_sha256=artifacts))
    print(json.dumps(dict(method=method, **result['overall'], memory_calls=result['memory_calls'])), flush=True)
    return result


def export():
    summaries = {method: read(OUTPUT / method / 'summary.json') for method in METHODS}
    component_audits = {}
    for method in METHODS:
        verify(OUTPUT, method)
        checked = read(OUTPUT / method / 'external_audit.json')
        assert checked['status'] == 'passed' and checked['questions'] == 1711
        for name, expected in checked['artifacts_sha256'].items():
            assert sha(name) == expected, name
        environment = read(OUTPUT / method / 'generation_environment.json')
        assert environment['device'].startswith('cuda:')
        component_audits[method] = dict(status='passed', questions=1711, memory_rounds=3962,
            audit_sha256=sha(OUTPUT / method / 'external_audit.json'), device=environment['device'])
    for source, item in read(OUTPUT / 'source_snapshot/manifest.json').items():
        assert sha(item['path']) == item['sha256'], source
    for name, expected in read(OUTPUT / 'inputs/protocol.json')['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    current_model = gallery.model_provenance()
    assert current_model == read(OUTPUT / METHODS[0] / 'protocol.json')['model']
    assert current_model == read(SOURCE / 'reader/protocol.json')['contract']['model']
    previous = read(SOURCE / 'reader/summary.json')['overall']
    pairs = read(SOURCE / 'reader/paired_results.json')
    assert len(pairs) == 1711
    reference = {}
    for method in gallery.CONDITIONS:
        correct = sum(int(gallery.score_answer(r[method]['raw_response'], r['answer'])['em']) for r in pairs)
        assert correct / 1711 == previous[method]['em']
        reference[method] = dict(correct=correct, questions=1711, em=correct / 1711, f1=previous[method]['f1'],
            recall_micro=previous[method]['recall_at_5_micro'], recall_macro=previous[method]['recall_at_5_macro_nonempty'])
    assert reference['baseline_question_only']['correct'] == 778 and reference['master_answer_query']['correct'] == 798
    records = []
    for method, r in reference.items():
        records.append(dict(method=method, source='previous matched experiment', correct=r['correct'],
            questions=1711, em=r['em'], f1=r['f1'], source_recall_micro=r['recall_micro'],
            source_recall_macro=r['recall_macro'], context_policy='Top-5 original dialogue rounds'))
    for method, result in summaries.items():
        r = result['overall']
        records.append(dict(method=NAMES[method], source='new experiment',
            **{k: r[k] for k in ('correct', 'questions', 'em', 'f1', 'source_recall_micro', 'source_recall_macro')},
            context_policy='At most 5 native units' if result['strict_context_at_most_five'] else 'Native K=10 / multi-tier; not Recall@5'))
    with (OUTPUT / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    scored = {m: {r['id']: r for r in rows(OUTPUT / m / 'scored_questions.jsonl')} for m in METHODS}
    with (OUTPUT / 'paired_questions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['id', 'task', 'point', 'answer'] + [m + suffix for m in METHODS for suffix in ('_answer', '_em', '_f1', '_gold_hits')]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for q in rows(OUTPUT / 'inputs/questions.jsonl'):
            row = {k: scored[METHODS[0]][q['id']][k] for k in fields[:4]}
            for method in METHODS:
                s = scored[method][q['id']]
                row.update({method + '_answer': s['raw_response'], method + '_em': s['em'],
                    method + '_f1': s['f1'], method + '_gold_hits': s['gold_hits']})
            writer.writerow(row)
    table = ''.join('<tr><td>' + html.escape(r['method']) + '</td><td>' + str(r['correct']) + '/1711</td>'
        + ''.join('<td>' + f'{r[k]:.4%}' + '</td>' for k in ('em', 'f1', 'source_recall_micro', 'source_recall_macro'))
        + '<td>' + html.escape(r['context_policy']) + '</td></tr>' for r in records)
    warnings = ''.join('<li>' + html.escape(NAMES[m]) + ': ' + html.escape(str(s['memory_calls'])) + '</li>' for m, s in summaries.items())
    (OUTPUT / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Mem-Gallery Qwen memory methods</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;color:#17212b}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f8}p{line-height:1.7}</style>'
        '<h1>Mem-Gallery · Qwen3-VL-8B-Instruct</h1><p>20 个场景、3,962 轮历史对话、1,711 道开放问答；seed=42。历史 caption 与此前基线完全相同，检索使用官方题图 caption，回答使用原始题图。EM、F1 沿用官方确定性评分。</p>'
        '<table><tr><th>方法</th><th>EM 正确数</th><th>EM</th><th>F1</th><th>来源召回 micro</th><th>来源召回 macro</th><th>上下文限制</th></tr>' + table + '</table>'
        '<p>前两行复用已核验的历史结果，其余四行重新构建记忆并回答。来源召回只计实际上下文中可追溯的原始对话轮，不推断摘要中的事实覆盖。GA 和 MemoryOS 不标记为 Recall@5；空 clue 的 184 题参与 EM/F1，但不参与召回率分母。</p>'
        '<p>四种记忆方法保留原生 MiniLM 编码配置（384 维，默认最多 256 token），回答器统一为同一 Qwen。完整配置见各方法 protocol.json 与 encoding_metadata.json。</p>'
        '<p>实现固定为实验启动时保存的 MemEye 原生方法适配器。共享工作目录后续加入的其他 SimpleMem 后端未混入本轮；续跑与核验使用已校验的代码快照，保留完整版本与运行记录。</p>'
        '<p>中间记忆调用审计（保留原生解析与回退行为）：</p><ul>' + warnings + '</ul>'
        '<p><a href="metrics.csv">指标 CSV</a> · <a href="paired_questions.csv">逐题对比 CSV</a> · <a href="final_results.json">完整结果 JSON</a></p></html>')
    dump(OUTPUT / 'final_results.json', dict(status='complete', references=reference, methods=summaries))
    runtime_files = {}
    for method in METHODS:
        for folder in (OUTPUT / method).glob('*/runtime'):
            for path in folder.rglob('*'):
                if path.is_file():
                    runtime_files[str(path)] = sha(path)
    dump(OUTPUT / 'runtime_artifacts_manifest.json', runtime_files)
    dump(OUTPUT / 'external_audit.json', dict(status='passed', methods=component_audits,
        shared_input_rounds=3962, shared_questions=1711, scenarios=20,
        baseline_references_recomputed=True, snapshots_verified=True,
        runtime_files_hashed=len(runtime_files), original_query_images_verified=True,
        source_comment_equivalence_recorded=(OUTPUT / 'source_comment_equivalence.json').exists(),
        frozen_snapshot_runtime_recorded=(OUTPUT / 'snapshot_runtime.json').exists()))
    status = read(OUTPUT / 'execution_status.json')
    status.update(status='complete', active_method=None)
    for method in METHODS:
        status['methods'][method]['final_status'] = 'audited_complete'
    dump(OUTPUT / 'execution_status.json', status)
    paths = [OUTPUT / n for n in ('metrics.csv', 'paired_questions.csv', 'report.html', 'final_results.json', 'observations.json', 'matched_configuration_audit.json', 'caption_reader_model_match.json', 'matched_reference_scoring.json', 'encoding_metadata.json', 'external_audit.json', 'runtime_artifacts_manifest.json')]
    for method in METHODS:
        paths += [OUTPUT / method / n for n in ('protocol.json', 'summary.json', 'external_audit.json', 'calls_manifest.json',
                  'scored_questions.jsonl', 'retrieval.jsonl', 'reader/responses.jsonl', 'reader/manifest.json', 'reader/complete.json')]
    paths += list((OUTPUT / 'inputs').glob('*')) + list((OUTPUT / 'source_snapshot').glob('*'))
    paths += list((OUTPUT / 'run_attempts').glob('*.json'))
    paths += [Path(__file__), ROOT / 'scripts/evaluation/run_mem_gallery_text_methods.py',
              ROOT / 'scripts/tests/test_mem_gallery_text_methods.py',
              ROOT / 'scripts/evaluation/run_mem_gallery_text_methods_v2.py']
    if (OUTPUT / 'metadata_adaptation.json').exists():
        paths.append(OUTPUT / 'metadata_adaptation.json')
    if (OUTPUT / 'source_comment_equivalence.json').exists():
        paths += [OUTPUT / 'source_comment_equivalence.json', ROOT / 'scripts/evaluation/mem_gallery_frozen_runtime.py']
    if (OUTPUT / 'snapshot_runtime.json').exists():
        paths += [OUTPUT / 'snapshot_runtime.json', ROOT / 'scripts/evaluation/mem_gallery_snapshot_runtime.py']
        paths += list((OUTPUT / 'snapshot_verification').glob('*.json'))
    paths += [p for p in (OUTPUT / 'gen_agents').glob('*') if p.is_file() and p.name in (
        'format_continuation_protocol.json', 'format_retry_events.jsonl', 'recovery.json', 'run_errors.json')]
    dump(OUTPUT / 'completion_manifest.json', dict(status='complete', artifacts_sha256={str(p): sha(p) for p in paths if p.is_file()}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    if not args.export_only:
        for method in ([args.method] if args.method else METHODS):
            audit_method(method)
    if not args.method:
        export()
