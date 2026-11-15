"""Post-run verification and Qwen text/Omni comparison; never starts inference."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'results/mem_gallery_qwen_omni_simplemem030_v2_20261104'
DEFAULT_OUT = ROOT / 'results/mem_gallery_qwen_omni_vs_text_20261104'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix('.tmp')
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    target.replace(path)


def verify_and_compare(run, output):
    import jsonschema
    manifest = read(run / 'completion_manifest.json')
    assert manifest['status'] == 'complete'
    for path, expected in manifest['artifact_sha256'].items():
        assert sha(path) == expected, path
    protocol = read(run / 'protocol.json')
    for path, expected in protocol['source_sha256'].items():
        assert sha(path) == expected, path
    guard = read(run / 'text_control_guard.json')
    for path, expected in guard.items():
        assert sha(path) == expected, path
    rejected = set()
    retries = run / 'memory/transport_recoveries.jsonl'
    for recovery in rows(retries) if retries.exists() else []:
        rejected.update(recovery['attempt_keys'][:-1])
    schema_invalid, schema_valid = set(), 0
    for path in (run / 'memory/calls').glob('*.json'):
        record = read(path)
        payload = record['request']
        assert hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == path.stem
        fmt = payload.get('response_format')
        if fmt and 'json_schema' in fmt:
            try:
                jsonschema.validate(json.loads(record['output']['raw_response']), fmt['json_schema']['schema'])
                schema_valid += 1
            except (ValueError, jsonschema.ValidationError):
                schema_invalid.add(path.stem)
                assert path.stem in rejected, path
    # Independently verify returned history pixels match native cold storage,
    # not only the adapter's hash copied into the reader manifest.
    maus = {}
    for task in protocol['tasks']:
        for mau in read(run / 'memory' / task / 'runtime/final_memory.json'):
            maus[mau['id']] = mau
    image_evidence = 0
    for retrieval in rows(run / 'retrieval.jsonl'):
        for image in retrieval['retrieval_detail']['history_images']:
            native_pointer = Path(maus[image['mau_id']]['raw_pointer'])
            assert native_pointer.is_relative_to(run)
            assert sha(native_pointer) == sha(image['path']) == image['sha256']
            image_evidence += 1
    current = read(run / 'final_results.json')
    control_root = Path(protocol['control_root'])
    assert str(control_root / 'final_results.json') in guard
    control = read(control_root / 'final_results.json')
    assert read(run / 'audit.json')['status'] == read(control_root / 'audit.json')['status'] == 'passed'
    assert current['overall']['questions'] == control['overall']['questions'] == 1711
    scored = rows(run / 'scored_questions.jsonl')
    assert len(scored) == len({r['id'] for r in scored}) == 1711
    for metric in ('em', 'f1', 'bleu_1', 'bleu_2', 'bleu'):
        assert abs(sum(r[metric] for r in scored) / 1711 - current['overall'][metric]) < 1e-12
    table = []
    for label, result in [('SimpleMem 0.3.0 text + caption', control), ('Omni-SimpleMem 0.3.0 visual RRF5', current)]:
        stats = result['overall']
        table.append(dict(method=label, questions=1711, correct=stats['correct'],
                          em_percent=100 * stats['em'], f1_percent=100 * stats['f1'],
                          bleu1_percent=100 * stats['bleu_1']))
    comparison = dict(status='complete', backbone='Qwen3-VL-8B-Instruct', results=table,
        correct_delta=current['overall']['correct'] - control['overall']['correct'],
        em_percentage_point_delta=100 * (current['overall']['em'] - control['overall']['em']),
        omni_recall_at_5=current['recall_at_5'], differences=protocol['differences_from_text'],
        note='Separate adapted backends, not a single-variable ablation or reproduction of official paper scores.',
        source_results_sha256={str(p): sha(p) for p in (run / 'final_results.json', control_root / 'final_results.json')})
    verification = dict(status='passed', run_manifest_artifacts=len(manifest['artifact_sha256']),
        input_source_hashes=len(protocol['source_sha256']), text_control_files=len(guard),
        independently_validated_schema_outputs=schema_valid, rejected_schema_outputs=len(schema_invalid),
        expanded_images_matched_to_native_storage=image_evidence)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'comparison.json', comparison)
    dump(output / 'postrun_verification.json', verification)
    with (output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    body = '<!doctype html><meta charset="utf-8"><title>Qwen Omni / text comparison</title>'
    body += '<style>body{font:17px system-ui;max-width:1000px;margin:40px auto}td,th{padding:12px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
    body += '<h1>Qwen3-VL-8B: Omni-SimpleMem / text</h1><table><tr><th>Method</th><th>Correct</th><th>EM</th><th>F1</th><th>BLEU-1</th></tr>'
    for row in table:
        body += '<tr><td>' + html.escape(row['method']) + f"</td><td>{row['correct']}/1711</td><td>{row['em_percent']:.2f}%</td><td>{row['f1_percent']:.2f}%</td><td>{row['bleu1_percent']:.2f}%</td></tr>"
    body += '</table><p>' + html.escape(comparison['note']) + '</p><ul>'
    body += ''.join('<li>' + html.escape(x) + '</li>' for x in comparison['differences']) + '</ul>'
    (output / 'report.html').write_text(body)
    dump(output / 'completion_manifest.json', dict(status='complete',
        artifact_sha256={str(path): sha(path) for path in output.iterdir() if path.is_file()
                        and path.name not in ('completion_manifest.json', 'run.log', 'execution.json')}))
    dump(output / 'execution.json', dict(status='complete'))
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / 'execution.json', dict(status='waiting_for_inference', run=str(args.run)))
    while args.wait and not (args.run / 'completion_manifest.json').exists():
        state = read(args.run / 'execution.json') if (args.run / 'execution.json').exists() else {}
        if state.get('status') == 'failed':
            dump(args.output / 'execution.json', dict(status='inference_failed', run=str(args.run)))
            raise RuntimeError('Inference failed; no comparison was fabricated')
        time.sleep(30)
    try:
        verify_and_compare(args.run, args.output)
    except BaseException as exc:
        dump(args.output / 'execution.json', dict(status='verification_failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
