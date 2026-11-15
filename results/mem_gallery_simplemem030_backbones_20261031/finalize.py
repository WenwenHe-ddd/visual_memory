"""Seal the completed GPT-5 run and the revalidated Qwen comparison."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import html
import hashlib
import json
from pathlib import Path
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.vendor')]
from scripts.evaluation import mem_gallery_gpt5_simplemem030 as run


def verify_hashes(hashes):
    def one(item):
        name, digest = item
        assert run.sha(name) == digest, name
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, hashes.items()))


def main():
    gpt = run.OUTPUT
    protocol = run.verify(gpt)
    done = run.read(gpt/'completion_manifest.json')
    assert done['status'] == 'complete'
    verify_hashes(done['artifacts_sha256'])
    gaudit = run.read(gpt/'audit.json')
    assert gaudit['status'] == 'passed' and gaudit['independent_rrf_questions'] == 1711
    memory_responses = run.rows(gpt/'memory/api_responses.jsonl')
    wire_keys = [r['output']['request_sha256'] for r in memory_responses]
    assert len(wire_keys) == len(set(wire_keys)), 'Duplicate memory requests require a cache provenance audit'
    recovery_note = []
    recovery_artifacts = {}
    if (gpt/'json_quote_recovery_amendment.json').exists():
        from scripts.evaluation import recover_gallery_gpt5_simplemem030 as recovery
        proof = run.read(gpt/'json_quote_recovery_amendment.json')
        completed = run.read(gpt/'json_quote_recovery_complete.json')
        assert completed['status'] == 'complete' and completed['original_call_unchanged']
        assert proof['source_sha256'] == run.sha(recovery.__file__)
        assert completed['amendment_sha256'] == run.sha(gpt/'json_quote_recovery_amendment.json')
        original = gpt/'memory/calls'/(recovery.CALL+'.json')
        assert run.sha(original) == proof['original_call_sha256']
        fixed = recovery.repair(run.read(original)['output']['raw_response'])
        assert hashlib.sha256(fixed.encode()).hexdigest() == proof['corrected_response_sha256']
        repairs = list((gpt/'memory').glob('*/runtime/json_quote_repair.json'))
        assert len(repairs) == 1
        repaired = run.read(repairs[0])
        assert repaired['corrected_response'] == fixed and repaired['replacements'] == 1
        notes = {e['entry_id']:e for e in run.read(repairs[0].parent.parent/'final_memory.json')}
        expected = json.loads(fixed)
        assert len(repaired['entry_ids']) == len(expected) == 10
        for identity, entry in zip(repaired['entry_ids'], expected):
            assert all(notes[identity][k] == v for k, v in entry.items())
        for task, digest in proof['retained_complete_tasks'].items():
            assert run.sha(gpt/'memory'/task/'complete.json') == digest
        recovery_note = ['GPT-5: one hash-bound repair of a single-quoted keyword to valid JSON; all field values unchanged.']
        for name in ('json_quote_recovery_amendment.json','json_quote_recovery_complete.json','initial_memory_errors.json'):
            recovery_artifacts[str(gpt/name)] = run.sha(gpt/name)
        if 'parallel_recovery_sha256' in proof:
            assert run.sha(gpt/'parallel_format_recovery.json') == proof['parallel_recovery_sha256']
            assert run.sha(gpt/'parallel_api_merge.json') == proof['api_merge_sha256']
            parallel = run.read(gpt/'parallel_format_recovery.json')
            assert parallel['status'] == 'complete' and parallel['concurrency_bound'] <= protocol['task_workers']
            assert parallel['source_sha256'] == run.sha(ROOT/'scripts/evaluation/recover_gallery_gpt5_simplemem030_parallel.py')
            for name, digest in parallel['stream_sha256'].items():
                assert run.sha(name) == digest
                recovery_artifacts[name] = digest
            for name, entry in run.read(gpt/'parallel_api_merge.json').items():
                data = Path(name).read_bytes()
                assert run.sha(name) == entry['merged_sha256']
                assert hashlib.sha256(data[:entry['prefix_bytes']]).hexdigest() == entry['prefix_sha256']
                assert data[entry['prefix_bytes']:] == Path(entry['source']).read_bytes()
            for name in ('parallel_format_recovery.json','parallel_api_merge.json'):
                recovery_artifacts[str(gpt/name)] = run.sha(gpt/name)
    q = run.read(OUTPUT/'qwen_revalidation.json')
    assert q['status'] == 'passed' and q['independent_rrf_questions'] == 1711
    verify_hashes(q['artifacts_sha256'])
    ds = ROOT/'results/mem_gallery_deepseek_text_methods_20261031/simplemem'
    ddone = run.read(ds/'completion_manifest.json')
    assert ddone['status'] == 'complete'
    for name in ('final_results.json', 'protocol.json', 'audit.json'):
        assert run.sha(ds/name) == ddone['artifacts_sha256'][str(ds/name)]
    gresult, dresult = run.read(gpt/'final_results.json'), run.read(ds/'final_results.json')
    methods = [dict(backbone='Qwen3-VL-8B-Instruct', kind='existing run revalidated', **q['overall']),
               dict(backbone=protocol['expected_model'], kind='new full run', **gresult['overall']),
               dict(backbone='deepseek-v4.1-flash', kind='previous completed reference', **dresult['overall'])]
    assert all(m['questions'] == 1711 and m['context_counts'] == {'5': 1711} for m in methods)
    run.dump(OUTPUT/'comparison.json', dict(status='complete', dataset='Mem-Gallery', method='SimpleMem 0.3.0',
        questions=1711, methods=methods, semantic_k=10, keyword_k=5, structured_k=5,
        planning=True, reflection=True, max_reflection_rounds=2, fusion='two-stage channel RRF60',
        final_context=5, source_recall=None,
        source_recall_note='Exact fact-to-source attribution unavailable; construction-window membership is not source recall.',
        differences=['History captions use the corresponding backbone; official supplied query captions are fixed.',
                     'Same pinned MiniLM embedding weights; Qwen control places the encoder on GPU, GPT-5 uses CPU.',
                     'Qwen greedy reader cap256 and local pixel preprocessing; GPT-5 minimal reasoning cap2048 and original image bytes.',
                     'Native temperatures recorded for Qwen/DeepSeek; GPT-5 omits temperature, seed42/minimal reasoning.',
                     *recovery_note,
                     *q['caveats'],
                     'DeepSeek history captions include two provider rejections; oversized images use documented same-resolution JPEG compression.']))
    with (OUTPUT/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['backbone','kind','correct','questions','em','f1','bleu_1'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(methods)
    table = ''.join('<tr><td>'+html.escape(m['backbone'])+'</td><td>'+str(m['correct'])+'/1711</td>'+
                    ''.join(f'<td>{100*m[k]:.2f}%</td>' for k in ('em','f1','bleu_1'))+'</tr>' for m in methods)
    (OUTPUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SimpleMem 0.3.0 backbone comparison</title>'
        '<style>body{font:16px system-ui;margin:36px;max-width:1100px}table{border-collapse:collapse}th,td{padding:12px;border:1px solid #ccc}</style>'
        '<h1>Mem-Gallery · SimpleMem 0.3.0</h1><p>1711 questions. Semantic K10, keyword K5, structured K5; planning and reflection (up to two rounds); two-stage channel RRF60; five whole entries per question.</p>'
        '<table><tr><th>Backbone</th><th>Correct</th><th>EM</th><th>F1</th><th>BLEU-1</th></tr>'+table+'</table>'
        '<p>GPT-5 is a new full run. Qwen is the completed run independently revalidated for all 1711 questions, actual contexts, RRF rankings, recovery records and scores. DeepSeek is the previous completed reference.</p>'
        '<p>Qwen reader: greedy, cap256, no final truncations; one historical extraction retry and one lossless location-field repair. GPT-5: minimal reasoning, cap2048 including reasoning, temperature omitted. History captions use each backbone; supplied query captions and task prompts are fixed.</p>'
        '<p>'+html.escape(' '.join(recovery_note))+'</p>'
        '<p>These are matched retrieval settings, with model-specific generation and image preprocessing. The pinned MiniLM weights are the same; the Qwen control uses GPU embeddings and GPT-5 uses CPU embeddings. DeepSeek also has two rejected history captions and documented compression for oversized images.</p>'
        '<p>Exact source Recall@5 is unavailable: a construction window is not exact provenance for a generated memory fact.</p>'
        '<p><a href="comparison.csv">Comparison CSV</a> · <a href="../mem_gallery_gpt5_simplemem030_20261031/scored_questions.csv">GPT-5 per-question CSV</a></p>')
    post = dict(status='passed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        gpt5_artifacts_checked=len(done['artifacts_sha256']), qwen_artifacts_checked=len(q['artifacts_sha256']),
        original_inputs_and_code_snapshots_verified=True, final_questions_each=1711,
        duplicate_memory_wire_requests=0,
        gpt5_format_recovery_verified=bool(recovery_note),
        finalizer_sha256=run.sha(__file__), gpt5_experiment_signature=protocol['experiment_signature'])
    run.dump(OUTPUT/'postrun_verification.json', post)
    run.dump(OUTPUT/'completion_manifest.json', dict(status='complete',
        artifacts_sha256={**recovery_artifacts, **{str(OUTPUT/n):run.sha(OUTPUT/n) for n in ('comparison.json','comparison.csv','report.html','qwen_revalidation.json','postrun_verification.json','finalize.py')}},
        gpt5_completion_manifest_sha256=run.sha(gpt/'completion_manifest.json')))
    print(json.dumps(methods), flush=True)


if __name__ == '__main__':
    main()
