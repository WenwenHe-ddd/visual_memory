"""Matched lambda/alpha experiment using unchanged signed retrieval/reader code."""
import argparse
import csv
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_zscore as base
from scripts.evaluation import mem_gallery_zscore_ablation as runner
from scripts.analysis import audit_mem_gallery_zscore as audit

DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_qwen_zscore_l0p03_a0p3_20261028'
REFERENCES = [base.DEFAULT_OUTPUT, runner.DEFAULT_OUTPUT]


def prepare(output, device, ridge, alpha):
    if output.resolve() in {p.resolve() for p in [base.SOURCE, *REFERENCES]}:
        raise ValueError('Use a separate output directory for each parameter pair')
    for ref in REFERENCES:
        base.verify_hashes(base.read(ref / 'completion_manifest.json')['artifacts_sha256'])
    p = runner.prepare(output, device, ridge, alpha)
    tracked = [Path(__file__).resolve()]
    tracked += [ref / name for ref in REFERENCES for name in
                ('completion_manifest.json', 'comparison.json', 'reader/paired_results.json')]
    for path in tracked:
        expected = base.sha(path)
        if str(path) in p['source_sha256']:
            assert p['source_sha256'][str(path)] == expected
        else:
            if (output / 'reader/protocol.json').exists():
                raise ValueError('Cannot add sources after reader preparation')
            p['source_sha256'][str(path)] = expected
    base.dump(output / 'retrieval_protocol.json', p)
    base.dump(output / 'run_invocation.json', {
        'entrypoint': str(Path(__file__).resolve()), 'output': str(output),
        'device': device, 'lambda': ridge, 'alpha': alpha, 'seed': 42,
        'requested_conditions': list(base.CONDITIONS),
        'policy': 'Only lambda/alpha change; captions, graph, hypotheses, queries, global Top-5 merge and reader configuration are preserved.'})
    return p


def finish(output, ridge, alpha):
    runner.check_parameters(output, ridge, alpha)
    current = base.analyze(output)
    checked = audit.audit_final(output)
    previous = [(path, base.read(path / 'comparison.json')) for path in REFERENCES]
    rows = []
    for method in base.BASE_CONDITIONS:
        plain = next(r for r in current['results'] if r['method'] == method)
        selected = [('unweighted_historical', None, None, plain)]
        for _, result in previous:
            item = next(r for r in result['results'] if r['method'] == method+'+zscore')
            selected.append(('previous_zscore', result['lambda'], result['alpha'], item))
        selected.append(('requested_zscore', ridge, alpha,
                         next(r for r in current['results'] if r['method'] == method+'+zscore')))
        for label, lmbda, strength, item in selected:
            rows.append({'method': method, 'condition': label, 'lambda': lmbda, 'alpha': strength,
                         **{k:item[k] for k in ('correct', 'questions', 'em', 'f1', 'recall_at_5_micro',
                           'recall_at_5_macro', 'gold_hits', 'gold_links', 'empty_answers',
                           'token_limit_records', 'result_source')}})
    pairs = base.read(output / 'reader/paired_results.json')
    changes = {}
    for path, result in previous:
        old = {p['id']:p for p in base.read(path / 'reader/paired_results.json')}
        key = f'lambda={result["lambda"]},alpha={result["alpha"]}'
        changes[key] = {}
        for method in base.CONDITIONS:
            c = dict(em_gained=0, em_lost=0, same_ordered_top5=0,
                     same_payload_em_gained=0, same_payload_em_lost=0, changed_top5_sets=0)
            for p in pairs:
                a, b = old[p['id']][method], p[method]
                c['em_gained'] += b['em'] > a['em']
                c['em_lost'] += b['em'] < a['em']
                c['same_ordered_top5'] += a['top5_memory_ids'] == b['top5_memory_ids']
                c['changed_top5_sets'] += set(a['top5_memory_ids']) != set(b['top5_memory_ids'])
                if a['payload_sha256'] == b['payload_sha256']:
                    c['same_payload_em_gained'] += b['em'] > a['em']
                    c['same_payload_em_lost'] += b['em'] < a['em']
            changes[key][method] = c
    result = {'status': 'complete', 'model': 'Qwen3-VL-8B-Instruct', 'seed': 42,
              'questions': 1711, 'requested_lambda': ridge, 'requested_alpha': alpha,
              'results': rows, 'paired_changes_vs_previous': changes,
              'comparison_policy': 'Unweighted and previous parameter settings are frozen historical results; requested settings receive fresh inference. Same-payload answer variation is reported separately.',
              'independent_audit_passed': checked['passed']}
    base.dump(output / 'parameter_comparison.json', result)
    with (output / 'parameter_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = ''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{r["lambda"] if r["lambda"] is not None else "none"}</td><td>{r["alpha"] if r["alpha"] is not None else "none"}</td><td>{r["correct"]}/1711</td><td>{100*r["em"]:.4f}%</td><td>{100*r["f1"]:.4f}%</td><td>{100*r["recall_at_5_micro"]:.4f}%</td></tr>' for r in rows)
    (output / 'parameter_comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Mem-Gallery parameter ablation</title><style>body{font:16px system-ui;margin:36px;max-width:1300px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}</style><h1>Mem-Gallery · Qwen3-VL-8B parameter ablation</h1>'
        f'<p>Requested lambda={ridge}, alpha={alpha}, seed=42. All 1711 questions; original captions, graph, queries and reader configuration.</p><table><tr><th>Method</th><th>Lambda</th><th>Alpha</th><th>Correct</th><th>EM</th><th>F1</th><th>Recall@5 micro</th></tr>'+table+
        '<p>Recall uses 4109 gold links in 1527 questions. The 184 empty-clue questions remain in EM/F1. Previous settings retain historical answers; this setting uses fresh inference.</p><p><a href="parameter_comparison.csv">CSV</a> · <a href="parameter_comparison.json">Paired changes</a> · <a href="final_audit.json">Independent audit</a> · <a href="reader/paired_results.csv">Per-question results</a></p>')
    paths = [output / name for name in ('retrieval_protocol.json', 'retrieval_audit.json', 'final_audit.json',
             'run_invocation.json', 'comparison.json', 'metrics.csv', 'report.html', 'reader/protocol.json',
             'reader/summary.json', 'reader/validation_output.json', 'reader/paired_results.json',
             'reader/paired_results.csv', 'parameter_comparison.json', 'parameter_comparison.csv',
             'parameter_comparison.html')]
    paths += [Path(__file__).resolve(), Path(runner.__file__).resolve(), Path(audit.__file__).resolve()]
    base.dump(output / 'completion_manifest.json', {'status': 'complete', 'questions_per_condition': 1711,
        'conditions': 2, 'lambda': ridge, 'alpha': alpha, 'seed': 42, 'independent_audit_passed': True,
        'artifacts_sha256': {str(path):base.sha(path) for path in paths}})
    print({'stage': 'parameter_ablation_complete', 'results': rows, 'paired_changes': changes}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'reader', 'analyze', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--whitening-lambda', type=float, default=.03)
    parser.add_argument('--reweight-alpha', type=float, default=.3)
    args = parser.parse_args(); output = args.output.resolve()
    if args.stage in ('prepare', 'all'):
        prepare(output, args.device, args.whitening_lambda, args.reweight_alpha)
    if args.stage in ('reader', 'all'):
        runner.reader(output, args.device, args.whitening_lambda, args.reweight_alpha)
    if args.stage in ('analyze', 'all'):
        finish(output, args.whitening_lambda, args.reweight_alpha)


if __name__ == '__main__':
    main()
