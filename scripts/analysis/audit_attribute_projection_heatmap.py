"""Independently recompute controlled-edit projections and verify deliverables."""
from pathlib import Path
import csv
from decimal import Decimal
import hashlib
import json
import os
import re
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/gpt5_attribute_projection_20261113'
PREVIOUS = ROOT / 'reports/gpt5_variance_observation_20261108'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def csvrows(name):
    with (OUT / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    plan = read(OUT / 'plan.json')
    summary = read(OUT / 'numeric_summary.json')
    assert summary['plan_sha256'] == sha(OUT / 'plan.json')
    assert plan['stimuli_sha256'] == sha(OUT / 'stimuli.jsonl')
    for p, h in plan['source_sha256'].items():
        assert sha(p) == h, p
    members = read(PREVIOUS / 'largest_members.json')
    mm = {m['id']: m for m in members}
    stimuli = [json.loads(s) for s in (OUT / 'stimuli.jsonl').read_text().splitlines()]
    assert len(stimuli) == len({s['id'] for s in stimuli}) == 943
    fields = plan['attributes'][:5]
    eligible = []
    for m in members:
        if all(len(re.findall(re.escape(f) + r'[^,.:%0-9]{0,20}(\d+(?:\.\d+)?)', m['caption'], re.I)) == 1 for f in fields):
            eligible.append(m['id'])
    assert eligible == plan['eligible_ids'] and len(eligible) == 41
    for s in stimuli:
        m = mm[s['memory_id']]
        assert s['original_text'] == m['retrieval_text']
        start, end = s['span_start'], s['span_end']
        assert s['old'] == s['original_text'][start:end] and s['old'] != s['new']
        assert s['edited_text'] == s['original_text'][:start] + s['new'] + s['original_text'][end:]
        if s['attribute'] in fields:
            step = int(s['variant'].split('_')[1])
            unit = Decimal('.1') if s['attribute'] == 'Evening Drift' else Decimal(1)
            assert Decimal(s['new']) - Decimal(s['old']) == step * unit
            assert step in [-3, -1, 1, 3]
            assert s['edited_text'].split('\nMemory description: ')[0] == m['dialogue']
            for f in fields:
                pattern = re.escape(f) + r'[^,.:%0-9]{0,20}(\d+(?:\.\d+)?)'
                old = re.findall(pattern, s['original_text'], re.I)
                new = re.findall(pattern, s['edited_text'], re.I)
                assert len(old) == len(new) == 1
                assert (old != new) == (f == s['attribute'])
        elif s['attribute'] == 'Assistant acknowledgment':
            assert s['edited_text'].split('\nMemory description: ')[1] == m['caption']
            assert s['new'] == 'Acknowledged.'
        else:
            assert s['edited_text'].split('\nMemory description: ')[0] == m['dialogue']
            for f in fields:
                pattern = re.escape(f) + r'[^,.:%0-9]{0,20}(\d+(?:\.\d+)?)'
                assert re.findall(pattern, s['original_text'], re.I) == re.findall(pattern, s['edited_text'], re.I)
    z = np.load(OUT / 'analysis_vectors.npz')
    x, u, v = z['originals'], z['basis'], z['eigenvalues']
    assert x.shape == (48, 384) and u.shape == (47, 384) and z['edited'].shape == (943, 384)
    for array in [x, z['edited'], z['repeats']]:
        np.testing.assert_allclose(np.linalg.norm(array, axis=1), 1, atol=1e-12, rtol=0)
    centered = x - x.mean(0)
    np.testing.assert_allclose(centered @ u.T @ u, centered, atol=1e-12, rtol=0)
    np.testing.assert_allclose(u @ u.T, np.eye(47), atol=1e-12, rtol=0)
    covariance = centered.T @ centered / 47
    np.testing.assert_allclose(u @ covariance @ u.T, np.diag(v), atol=1e-12, rtol=0)
    old_coords = {r['memory_id']: r['coordinates'] for r in read(PREVIOUS / 'largest_coordinates.json')}
    coord_error = float(abs(centered @ u.T - np.array([old_coords[m['id']] for m in members])).max())
    assert coord_error < 2e-6
    indices = [next(i for i, m in enumerate(members) if m['id'] == s['memory_id']) for s in stimuli]
    np.testing.assert_array_equal(indices, z['original_indices'])
    # Use vector-by-vector projections and an explicit orthogonal residual.
    energy, bands = [], []
    for i, idx in enumerate(indices):
        delta = z['edited'][i] - x[idx]
        den = float(np.dot(delta, delta))
        coefficients = np.array([np.dot(direction, delta) for direction in u])
        e = coefficients**2 / den
        residual = delta - coefficients @ u
        outside = float(np.dot(residual, residual) / den)
        energy.append(e)
        bands.append([e[:10].sum(), e[10:37].sum(), e[37:].sum(), outside])
    energy = np.array(energy); bands = np.array(bands)
    np.testing.assert_allclose(energy, z['energy'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(bands, z['bands'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(bands.sum(1), 1, atol=1e-12, rtol=0)
    rows = csvrows('projection_energy.csv')
    assert [r['id'] for r in rows] == [s['id'] for s in stimuli]
    stored = np.array([[float(r[f'PC{j+1:02d}']) for j in range(47)] for r in rows])
    np.testing.assert_allclose(stored, energy, atol=1e-12, rtol=0)
    bymember = {}
    for attr in plan['attributes']:
        for mid in eligible:
            ix = [i for i, s in enumerate(stimuli) if s['attribute'] == attr and s['memory_id'] == mid]
            assert len(ix) == (4 if attr in fields else 1)
            bymember[mid, attr] = np.r_[energy[ix].mean(0), bands[ix].mean(0)]
    for r in summary['attributes']:
        mean = np.mean([bymember[mid, r['attribute']] for mid in eligible], axis=0)
        stored = [r[f'PC{j+1:02d}'] for j in range(47)] + [r[k] for k in ['high', 'middle', 'low', 'outside']]
        np.testing.assert_allclose(mean, stored, atol=1e-12, rtol=0)
    paired = np.array([np.mean([bymember[mid, a][-4:] for a in fields[:2]], axis=0) -
                       np.mean([bymember[mid, a][-4:] for a in fields[2:]], axis=0) for mid in eligible])
    rng = np.random.default_rng(42)
    boot = paired[rng.integers(0, len(paired), size=(5000, len(paired)))].mean(1)
    for j, k in enumerate(['high', 'middle', 'low', 'outside']):
        result = summary['paired_numeric_comparison'][k]
        np.testing.assert_allclose(result['requested_minus_other'], paired[:, j].mean(), atol=1e-12)
        np.testing.assert_allclose(result['bootstrap95'], np.quantile(boot[:, j], [.025, .975]), atol=1e-12)
    assert sum(a['truncated_inputs'] for a in summary['encoding_audits']) == 0
    workbook = OUT / 'Attribute_projection_heatmap.xlsx'
    wb = load_workbook(workbook)
    assert wb['Edits'].max_row == 944 and wb['AllProjections'].max_row == 944
    assert wb['PerMember'].max_row == 329 and wb['Attributes'].max_row == 9
    assert len(wb['BandHeatmap']._charts) == 1 and len(wb['Figures']._images) == 2
    assert len(wb['BandHeatmap']._charts[0].series) == 4
    for i, r in enumerate(summary['attributes'], 2):
        for j, k in enumerate(['high', 'middle', 'low', 'outside'], 2):
            np.testing.assert_allclose(wb['BandHeatmap'].cell(i, j).value, r[k]*100, atol=1e-10)
    with zipfile.ZipFile(workbook) as archive:
        assert archive.testzip() is None
        assert len([p for p in archive.namelist() if re.fullmatch('xl/charts/chart[0-9]+.xml', p)]) == 1
    doc = (OUT / 'report.html').read_text()
    assert doc.count('<details>') == 943
    for p in ['attribute_pc_heatmap', 'attribute_band_heatmap', 'q16_two_cards_band_heatmap']:
        for ext in ['png', 'pdf']:
            assert (OUT / f'{p}.{ext}').stat().st_size > 1000
    audit = {'status': 'passed', 'original_members': 48, 'matched_members': 41,
             'single_edit_stimuli_verified': 943, 'pca_dimensions': 47,
             'independent_projection_entries': 943*47, 'energy_conservation_max_error': float(abs(bands.sum(1)-1).max()),
             'coordinate_max_error_vs_prior': coord_error,
             'source_hashes_verified': len(plan['source_sha256']), 'unchanged_source_files': True,
             'independent_residual_energy_checked': True, 'all_aggregate_values_and_bootstrap_verified': True,
             'workbook_native_charts': 1, 'workbook_embedded_images': 2, 'workbook_archive_integrity': True,
             'external_api_calls': 0, 'original_memory_modified': False,
             'audit_script': str(Path(__file__)), 'audit_script_sha256': sha(__file__)}
    (OUT / 'independent_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    manifest = {'status': 'complete', 'audit_status': 'passed',
                'output_sha256': {str(p): sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'completion_manifest.json'},
                'source_sha256': plan['source_sha256'], 'audit_script_sha256': sha(__file__),
                'memory_modified': False, 'api_calls': 0}
    (OUT / 'completion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
