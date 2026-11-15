"""Run a requested lambda/alpha ablation without editing signed prior sources."""
import argparse
import csv
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_zscore as base
from scripts.analysis import audit_mem_gallery_zscore as audit

DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_qwen_zscore_l0p01_a1_20261027'
REFERENCE = base.DEFAULT_OUTPUT


def prepare(output, device, ridge, alpha):
    if output.resolve() in (REFERENCE.resolve(), base.SOURCE.resolve()):
        raise ValueError('Ablations require a separate output directory')
    p = base.prepare(output, device, ridge, alpha)
    policy = f'User-requested Mem-Gallery ablation: lambda={ridge}, alpha={alpha}; all other retrieval and reader settings preserved.'
    if p.get('ablation_reference') != str(REFERENCE):
        if (output / 'reader/protocol.json').exists():
            raise ValueError('Cannot change ablation metadata after reader preparation')
        # The existing runner has a historical .03 parameter-selection note.
        # Correct only the new run's metadata before audit/inference; scoring
        # already used the actual ridge/alpha arguments passed above.
        p['parameter_policy'] = policy
        p['ablation_reference'] = str(REFERENCE)
        p['source_sha256'][str(Path(__file__).resolve())] = base.sha(__file__)
        p['source_sha256'][str(Path(audit.__file__).resolve())] = base.sha(audit.__file__)
        for name in ('completion_manifest.json', 'comparison.json', 'reader/protocol.json'):
            path = REFERENCE / name
            p['source_sha256'][str(path)] = base.sha(path)
        base.dump(output / 'retrieval_protocol.json', p)
    assert p['parameter_policy'] == policy
    assert p['lambda'] == ridge and p['alpha'] == alpha
    result = audit.audit_retrieval(output)
    print({'stage': 'retrieval_audited', **result}, flush=True)
    return p


def check_parameters(output, ridge, alpha):
    p = base.verify(output)
    assert p['lambda'] == ridge and p['alpha'] == alpha
    assert p['ablation_reference'] == str(REFERENCE)
    return p


def reader(output, device, ridge, alpha):
    check_parameters(output, ridge, alpha)
    model = base.reader.model_provenance()
    prior = base.read(REFERENCE / 'reader/protocol.json')['contract']['model']
    assert model == prior
    print({'stage': 'reader_preflight', 'model_hashes_match_previous_run': True,
           'lambda': ridge, 'alpha': alpha, 'seed': 42, 'device': device}, flush=True)
    return base.run_reader(output, device)


def finish(output, ridge, alpha):
    check_parameters(output, ridge, alpha)
    current = base.analyze(output)
    final_audit = audit.audit_final(output)
    previous = base.read(REFERENCE / 'comparison.json')
    assert previous['lambda'] == .03 and previous['alpha'] == 1.
    rows = []
    for method in base.BASE_CONDITIONS:
        plain = next(r for r in current['results'] if r['method'] == method)
        old = next(r for r in previous['results'] if r['method'] == method+'+zscore')
        new = next(r for r in current['results'] if r['method'] == method+'+zscore')
        for name, r, lmbda, a in [('unweighted_historical', plain, None, None),
                                  ('previous_zscore', old, previous['lambda'], previous['alpha']),
                                  ('requested_zscore', new, ridge, alpha)]:
            rows.append({'method': method, 'condition': name, 'lambda': lmbda, 'alpha': a,
                         **{k:r[k] for k in ('correct', 'questions', 'em', 'f1', 'recall_at_5_micro',
                                            'recall_at_5_macro', 'gold_hits', 'gold_links',
                                            'empty_answers', 'token_limit_records', 'result_source')}})
    prev_pairs = {r['id']:r for r in base.read(REFERENCE / 'reader/paired_results.json')}
    pairs = base.read(output / 'reader/paired_results.json')
    changes = {}
    for method in base.CONDITIONS:
        changes[method] = {'em_gained': 0, 'em_lost': 0, 'same_ordered_top5': 0,
                           'same_payload_em_gained': 0, 'same_payload_em_lost': 0,
                           'changed_top5_sets': 0}
        for pair in pairs:
            a, b = prev_pairs[pair['id']][method], pair[method]
            c = changes[method]
            c['em_gained'] += b['em'] > a['em']
            c['em_lost'] += b['em'] < a['em']
            c['same_ordered_top5'] += a['top5_memory_ids'] == b['top5_memory_ids']
            c['changed_top5_sets'] += set(a['top5_memory_ids']) != set(b['top5_memory_ids'])
            if a['payload_sha256'] == b['payload_sha256']:
                c['same_payload_em_gained'] += b['em'] > a['em']
                c['same_payload_em_lost'] += b['em'] < a['em']
    result = {'status': 'complete', 'model': 'Qwen3-VL-8B-Instruct', 'seed': 42,
              'questions': 1711, 'results': rows, 'paired_change_vs_lambda0p03': changes,
              'comparison_policy': 'Original no-z answers and lambda=.03 answers remain historical; all requested weighted answers receive fresh inference. Same-payload GPU answer variation is recorded separately.',
              'independent_audit_passed': final_audit['passed']}
    base.dump(output / 'lambda_comparison.json', result)
    with (output / 'lambda_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = ''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{r["lambda"] if r["lambda"] is not None else "none"}</td><td>{r["correct"]}/1711</td><td>{100*r["em"]:.4f}%</td><td>{100*r["f1"]:.4f}%</td><td>{100*r["recall_at_5_micro"]:.4f}%</td></tr>' for r in rows)
    (output / 'lambda_comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Mem-Gallery lambda ablation</title><style>body{font:16px system-ui;margin:36px;max-width:1200px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}</style><h1>Mem-Gallery Qwen3-VL-8B · lambda ablation</h1><p>Alpha=1, seed=42. Full 1711-question evaluation. Same captions, master graphs, queries and reader settings.</p><table><tr><th>Method</th><th>Lambda</th><th>Correct</th><th>EM</th><th>F1</th><th>Recall@5 micro</th></tr>'+table+'<p>Recall uses 4109 gold links in 1527 questions. All 184 empty-clue questions remain in answer scoring. Original unweighted and lambda=.03 results are frozen historical runs. New results use fresh inference.</p><p><a href="lambda_comparison.csv">CSV</a> · <a href="lambda_comparison.json">Comparison and paired changes</a> · <a href="final_audit.json">Independent audit</a> · <a href="reader/paired_results.csv">Per-question results</a></p>')
    files = [output / n for n in ('retrieval_protocol.json', 'retrieval_audit.json', 'final_audit.json',
             'comparison.json', 'metrics.csv', 'report.html', 'reader/protocol.json', 'reader/summary.json',
             'reader/validation_output.json', 'reader/paired_results.json', 'reader/paired_results.csv',
             'lambda_comparison.json', 'lambda_comparison.csv', 'lambda_comparison.html')]
    files.extend([Path(__file__).resolve(), Path(audit.__file__).resolve()])
    base.dump(output / 'completion_manifest.json', {'status': 'complete', 'questions_per_condition': 1711,
              'conditions': 2, 'lambda': ridge, 'alpha': alpha, 'seed': 42, 'independent_audit_passed': True,
              'artifacts_sha256': {str(f):base.sha(f) for f in files}})
    print({'stage': 'ablation_complete', 'results': rows, 'paired_changes': changes}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'reader', 'analyze', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--whitening-lambda', type=float, default=.01)
    parser.add_argument('--reweight-alpha', type=float, default=1.)
    args = parser.parse_args(); output = args.output.resolve()
    if args.stage in ('prepare', 'all'):
        prepare(output, args.device, args.whitening_lambda, args.reweight_alpha)
    if args.stage in ('reader', 'all'):
        reader(output, args.device, args.whitening_lambda, args.reweight_alpha)
    if args.stage in ('analyze', 'all'):
        finish(output, args.whitening_lambda, args.reweight_alpha)


if __name__ == '__main__':
    main()
