"""Controlled caption-attribute probes in a frozen, original BGE PCA basis.

prepare fixes all edits before encoding; run never calls a caption/answer API.
Counterfactual text copies are analysis stimuli, not replacement memory facts.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                   'MPLCONFIGDIR': '/tmp/visualmemory-matplotlib'}.items():
    os.environ.setdefault(key, value)

SOURCE = ROOT / 'results/memeye_gpt5_base_seed42_full_20261024'
PREVIOUS = ROOT / 'reports/gpt5_variance_observation_20261108'
OUT = ROOT / 'reports/gpt5_attribute_projection_20261113'
DOMAIN = 'Personal_Health_Dashboard_Assistant'
MASTER = 'master_000070'
FIELDS = ['Meal Variability', 'Evening Drift', 'Time In Range', 'Morning Response', 'Late Window']
ATTRS = FIELDS + ['Theme description', 'Caption wording', 'Assistant acknowledgment']
QUERY = 'What Meal Variability and Evening Drift values are shown on this dashboard?'
LABELS = ['Meal Variability [requested]', 'Evening Drift [requested]',
          'Time In Range [other]', 'Morning Response [other]', 'Late Window [other]',
          'Theme: dark to light', 'Caption verb paraphrase', 'Assistant acknowledgment']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvwrite(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def field_match(text, field):
    return list(re.finditer(re.escape(field) + r'[^,.:%0-9]{0,20}(\d+(?:\.\d+)?)', text, flags=re.I))


def prepare(out):
    if (out / 'plan.json').exists():
        raise FileExistsError('Prepared inputs already exist; use a fresh directory to change the design.')
    out.mkdir(parents=True, exist_ok=True)
    members = read(PREVIOUS / 'largest_members.json')
    assert len(members) == 48 and {m['master_id'] for m in members} == {MASTER}
    prior = read(PREVIOUS / 'numeric_summary.json')
    # Freeze and verify the actual prior member table before deriving any edits.
    assert sha(PREVIOUS / 'largest_members.json') == prior['output_sha256'][str(PREVIOUS / 'largest_members.json')]
    sources = [PREVIOUS / n for n in ['largest_members.json', 'largest_coordinates.json', 'spectra.csv',
                                    'numeric_summary.json', 'illustrative_rescue.json']]
    sources += [SOURCE / n for n in ['inputs/source_memories.jsonl', 'inputs/source_questions.jsonl',
                                     'inputs/protocol.json', 'captions.jsonl',
                                     f'domains/{DOMAIN}/memory/graph.json']]
    sources += [ROOT / n for n in ['scripts/memory/memeye_backbone_retrieval.py',
                                  'scripts/memory/memeye_backbone_long_text.py',
                                  'scripts/memory/caption_visual_master.py', 'scripts/memory/ridge_whitening.py']]
    sources.append(Path(__file__))
    stimuli, eligible, excluded = [], [], []

    def add(m, attr, start, end, replacement, variant, role):
        before = m['retrieval_text']
        after = before[:start] + replacement + before[end:]
        assert before != after
        stimuli.append({'id': f"{m['id']}/{ATTRS.index(attr):02d}/{variant}",
                        'memory_id': m['id'], 'image_path': m['image_path'],
                        'attribute': attr, 'role': role, 'variant': variant,
                        'span_start': start, 'span_end': end, 'old': before[start:end], 'new': replacement,
                        'original_text': before, 'edited_text': after})

    for m in members:
        text = m['retrieval_text']
        assert text == m['dialogue'] + '\nMemory description: ' + m['caption']
        sources.append(Path(m['image_path']))
        assert sha(m['image_path']) == m['image_sha256']
        matches = {f: field_match(m['caption'], f) for f in FIELDS}
        if any(len(v) != 1 for v in matches.values()):
            excluded.append({'id': m['id'], 'reason': 'Not all five numeric fields occur exactly once',
                             'field_counts': {f: len(v) for f, v in matches.items()}})
            continue
        eligible.append(m['id'])
        offset = len(m['dialogue'] + '\nMemory description: ')
        for field in FIELDS:
            match = matches[field][0]
            old = match.group(1)
            precision = len(old.split('.')[1]) if '.' in old else 0
            unit = Decimal('0.1') if field == 'Evening Drift' else Decimal(1)
            for step in [-3, -1, 1, 3]:
                value = Decimal(old) + unit * step
                assert value > 0 and (field in FIELDS[:2] or value <= 100)
                new = f'{value:.{precision}f}'
                add(m, field, offset + match.start(1), offset + match.end(1), new,
                    f'step_{step:+d}', 'requested_numeric' if field in FIELDS[:2] else 'other_numeric')
        theme = re.search(r'\bdark(?:[-‑ ]themed)?\b', m['caption'], flags=re.I)
        assert theme, m['id']
        old = theme.group()
        new = re.sub('dark', 'light', old, flags=re.I)
        if old[0].isupper():
            new = new[0].upper() + new[1:]
        add(m, ATTRS[5], offset + theme.start(), offset + theme.end(), new, 'light', 'appearance_control')
        verb = re.search(r'\b(shows|displays|presents|shown|displayed|presented)\b', m['caption'])
        assert verb, m['id']
        replacement = {'shows': 'displays', 'displays': 'shows', 'presents': 'shows',
                       'shown': 'displayed', 'displayed': 'shown', 'presented': 'shown'}[verb.group()]
        add(m, ATTRS[6], offset + verb.start(), offset + verb.end(), replacement, 'verb', 'paraphrase_control')
        assistant = re.search(r'(?m)^Assistant: ([^\n]+)$', m['dialogue'])
        assert assistant, m['id']
        add(m, ATTRS[7], assistant.start(1), assistant.end(1), 'Acknowledged.', 'ack', 'dialogue_control')

    assert len(eligible) == 41 and len(stimuli) == 943
    assert len({s['id'] for s in stimuli}) == len(stimuli)
    (out / 'stimuli.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in stimuli))
    csvwrite(out / 'edits.csv', [{k: v for k, v in s.items() if k != 'image_path'} for s in stimuli])
    plan = {
        'status': 'prepared_before_encoding', 'model_caption_source': 'gpt-5-2025-08-07',
        'encoder': 'BAAI/bge-small-en-v1.5', 'domain': DOMAIN, 'master': MASTER,
        'basis': 'Original 48 dialogue+caption unit embeddings, centered SVD, 47 directions; never fit on edited stimuli.',
        'basis_members': 48, 'matched_probe_members': len(eligible), 'eligible_ids': eligible, 'excluded': excluded,
        'stimuli': len(stimuli), 'attributes': ATTRS,
        'selection': 'Continue the prior largest-master observation; include every member with all five numeric fields exactly once, independent of projection outcome.',
        'diagnostic_question': QUERY,
        'question_status': 'Analyst-defined attribute readout question, not an additional MemEye benchmark item or a QA evaluation.',
        'relevance': 'Requested numeric fields are Meal Variability and Evening Drift. Other numeric fields, theme and expression do not change the requested pair of values. This is question-specific, not intrinsic irrelevance.',
        'numeric_edits': 'One scalar per copy, steps -3/-1/+1/+3 in units of 1 point or percentage point, 0.1 hour for Evening Drift; preserve numeric precision and all remaining text.',
        'other_edits': 'First dark-theme mention to light; first shows/displays/presents or shown/displayed/presented verb paraphrase preserving voice; assistant acknowledgment to Acknowledged.',
        'image_policy': 'Original image files are read-only. Changed numeric/theme captions are counterfactual text probes, not true descriptions of those original images.',
        'energy': '(u_j dot (e_edit-e_original))^2 / ||e_edit-e_original||^2 using unwhitened unit embeddings.',
        'outside': '1-sum of the 47 PC energy shares, displayed separately; never renormalize away out-of-span energy.',
        'bands': {'high': [1, 10], 'middle': [11, 37], 'low': [38, 47], 'outside_dimensions': 337},
        'aggregation': 'Average four numeric edits within member first, then average the same 41 members for each attribute. No energy pooling and no per-row rescaling.',
        'primary_comparison': 'Within-member requested-numeric mean minus other-numeric mean; report high/middle/low/outside energy fractions, paired member-bootstrap 95% intervals (5000 resamples, seed 42).',
        'predefined_case_subset': [f'{DOMAIN}/HEALTH_S5:R4', f'{DOMAIN}/HEALTH_S5:R6'],
        'case_subset_reason': 'Two actual D5 cards named in previously inspected Health/Q16, chosen before projections.',
        'limits': ['Text-encoding sensitivity is not image-model sensitivity or task accuracy.',
                   'Changing numbers, theme, wording are not semantically equal interventions.',
                   'Original low-variance directions need not represent visual details.',
                   'The low-dimensional original cluster may not span novel edits.',
                   'Member bootstrap intervals are exploratory within this one cluster, not independent-dataset uncertainty.',
                   'PC numbers may be unstable when eigenvalues are nearly tied; also report equal-size high/low bands.'],
        'api_calls': 0, 'memory_modified': False, 'whitening_applied': False,
        'source_sha256': {str(p): sha(p) for p in sources},
        'stimuli_sha256': sha(out / 'stimuli.jsonl'),
    }
    dump(out / 'plan.json', plan)
    print(json.dumps({'stage': 'prepared', 'members': len(eligible), 'stimuli': len(stimuli), 'output': str(out)}), flush=True)


def run(out, device):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    from scripts.memory.ridge_whitening import fit
    plan = read(out / 'plan.json')
    if (out / 'completion_manifest.json').exists():
        raise FileExistsError('Completed result is immutable; use a fresh output root.')
    for p, h in plan['source_sha256'].items():
        assert sha(p) == h, p
    assert sha(out / 'stimuli.jsonl') == plan['stimuli_sha256']
    stimuli = lines(out / 'stimuli.jsonl')
    args = SimpleNamespace(device=device)
    original.configure_device(args)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    prep = read(SOURCE / 'inputs/protocol.json')
    model = original.load_text_encoder(prep['text_encoder'], args.device)
    memories, _, caption_model = original.attach_captions(lines(SOURCE / 'inputs/source_memories.jsonl'),
        lines(SOURCE / 'inputs/source_questions.jsonl'), lines(SOURCE / 'captions.jsonl'))
    assert caption_model == plan['model_caption_source']
    ms = [m for m in memories if m['task'] == DOMAIN]
    e, orig_audit = encode_complete(model, [m['retrieval_text'] for m in ms], ids=[m['id'] for m in ms], field='original_domain')
    members = read(PREVIOUS / 'largest_members.json')
    id_index = {m['id']: i for i, m in enumerate(ms)}
    x = e[[id_index[m['id']] for m in members]]
    for m in members:
        assert m['retrieval_text'] == ms[id_index[m['id']]]['retrieval_text']
    u, v = fit(x)
    assert len(u) == 47 and (v > 0).all()
    for row in u:
        if row[np.argmax(abs(row))] < 0:
            row *= -1
    coords = (x - x.mean(0)) @ u.T
    oldcoords = {r['memory_id']: r['coordinates'] for r in read(PREVIOUS / 'largest_coordinates.json')}
    coord_error = float(np.max(abs(coords - np.array([oldcoords[m['id']] for m in members]))))
    oldsp = [r for r in csv.DictReader((PREVIOUS / 'spectra.csv').open(encoding='utf-8-sig'))
             if r['domain'] == DOMAIN and r['master_id'] == MASTER]
    spectrum_error = float(np.max(abs(v - np.array([float(r['variance']) for r in oldsp]))))
    assert coord_error < 2e-6 and spectrum_error < 2e-7, (coord_error, spectrum_error)
    print(json.dumps({'stage': 'basis_verified', 'coordinate_max_error': coord_error,
                      'spectrum_max_error': spectrum_error, 'device': args.device}), flush=True)
    edited, edit_audit = encode_complete(model, [s['edited_text'] for s in stimuli],
                                         ids=[s['id'] for s in stimuli], field='edited_copies')
    repeat, repeat_audit = encode_complete(model, [m['retrieval_text'] for m in members],
                                          ids=[m['id'] for m in members], field='identity_reencode')
    repeat_error = float(np.max(np.linalg.norm(repeat - x, axis=1)))
    assert repeat_error < 2e-6
    base = e[[id_index[s['memory_id']] for s in stimuli]]
    delta = edited - base
    total = np.einsum('ij,ij->i', delta, delta)
    assert total.min() > max(1e-12, 100 * repeat_error ** 2)
    coefficients = delta @ u.T
    energies = coefficients ** 2 / total[:, None]
    outside = 1 - energies.sum(1)
    assert outside.min() > -1e-10
    outside = np.maximum(outside, 0)
    bands = np.column_stack([energies[:, :10].sum(1), energies[:, 10:37].sum(1),
                             energies[:, 37:].sum(1), outside])
    np.testing.assert_allclose(bands.sum(1), 1, atol=1e-12)
    assert all(a['truncated_inputs'] == 0 for a in [orig_audit, edit_audit, repeat_audit])
    np.savez_compressed(out / 'analysis_vectors.npz', originals=x, basis=u, eigenvalues=v,
                        original_indices=np.array([next(i for i, m in enumerate(members) if m['id'] == s['memory_id']) for s in stimuli]),
                        edited=edited, repeats=repeat, coefficients=coefficients, energy=energies, bands=bands)
    records = []
    for i, s in enumerate(stimuli):
        records.append({'id': s['id'], 'memory_id': s['memory_id'], 'attribute': s['attribute'],
                        'role': s['role'], 'variant': s['variant'], 'delta_norm': float(np.sqrt(total[i])),
                        **{f'PC{j+1:02d}': float(energies[i, j]) for j in range(47)},
                        **{k: float(bands[i, j]) for j, k in enumerate(['high', 'middle', 'low', 'outside'])}})
    csvwrite(out / 'projection_energy.csv', records)
    per_member = []
    for attr in ATTRS:
        for mid in plan['eligible_ids']:
            ix = [i for i, s in enumerate(stimuli) if s['attribute'] == attr and s['memory_id'] == mid]
            assert len(ix) == (4 if attr in FIELDS else 1)
            per_member.append({'attribute': attr, 'memory_id': mid, 'edits': len(ix),
                               'mean_delta_norm': float(np.sqrt(total[ix]).mean()),
                               **{f'PC{j+1:02d}': float(energies[ix, j].mean()) for j in range(47)},
                               **{k: float(bands[ix, j].mean()) for j, k in enumerate(['high', 'middle', 'low', 'outside'])}})
    csvwrite(out / 'per_member_energy.csv', per_member)
    summary_rows = []
    for attr in ATTRS:
        rs = [r for r in per_member if r['attribute'] == attr]
        keys = ['mean_delta_norm'] + [f'PC{j+1:02d}' for j in range(47)] + ['high', 'middle', 'low', 'outside']
        summary_rows.append({'attribute': attr, 'members': len(rs), 'edits': sum(r['edits'] for r in rs),
                             **{k: float(np.mean([r[k] for r in rs])) for k in keys}})
    csvwrite(out / 'attribute_summary.csv', summary_rows)
    lookup = {(r['memory_id'], r['attribute']): r for r in per_member}
    paired = []
    for mid in plan['eligible_ids']:
        paired.append([np.mean([lookup[mid, a][k] for a in FIELDS[:2]]) -
                       np.mean([lookup[mid, a][k] for a in FIELDS[2:]]) for k in ['high', 'middle', 'low', 'outside']])
    paired = np.array(paired)
    rng = np.random.default_rng(42)
    boot = paired[rng.integers(0, len(paired), size=(5000, len(paired)))].mean(1)
    comparisons = {k: {'requested_minus_other': float(paired[:, i].mean()),
                        'bootstrap95': [float(x0) for x0 in np.quantile(boot[:, i], [.025, .975])]}
                   for i, k in enumerate(['high', 'middle', 'low', 'outside'])}
    natural = []
    for i, m1 in enumerate(members):
        for j in range(i + 1, len(members)):
            m2 = members[j]
            if m1['image_sha256'] == m2['image_sha256'] and m1['caption'] == m2['caption']:
                d = x[j] - x[i]
                en = (u @ d) ** 2 / (d @ d)
                natural.append({'first': m1['id'], 'second': m2['id'],
                                'same_image_and_caption': True, 'first_dialogue': m1['dialogue'],
                                'second_dialogue': m2['dialogue'], 'high': float(en[:10].sum()),
                                'middle': float(en[10:37].sum()), 'low': float(en[37:].sum()),
                                'outside': float(max(0, 1 - en.sum()))})
    summary = {'status': 'numeric_complete', 'plan_sha256': sha(out / 'plan.json'),
               'attributes': summary_rows, 'paired_numeric_comparison': comparisons,
               'same_image_dialogue_controls': natural,
               'coordinate_max_error_vs_previous': coord_error, 'spectrum_max_error_vs_previous': spectrum_error,
               'identity_reencode_max_l2': repeat_error, 'minimum_edit_l2': float(np.sqrt(total.min())),
               'orthogonality_max_error': float(abs(u @ u.T - np.eye(47)).max()),
               'encoding_audits': [orig_audit, edit_audit, repeat_audit],
               'runtime': {'python': sys.version, 'torch': torch.__version__, 'device': args.device,
                           'gpu': torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None},
               'original_variance_shares': {'high': float(v[:10].sum()/v.sum()),
                                           'middle': float(v[10:37].sum()/v.sum()), 'low': float(v[37:].sum()/v.sum())},
               'no_reader_calls': True, 'no_external_api_calls': True, 'memory_modified': False,
               'analysis_vectors': 'Saved only in this separate report for reproducibility, not in the memory database.'}
    for p, h in plan['source_sha256'].items():
        assert sha(p) == h, p
    dump(out / 'numeric_summary.json', summary)
    print(json.dumps({'stage': 'numeric_complete', 'rows': len(records),
                      'bands': [{k: r[k] for k in ['attribute', 'high', 'middle', 'low', 'outside']} for r in summary_rows],
                      'paired_numeric_comparison': comparisons}, ensure_ascii=False), flush=True)


def render(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as ExcelImage
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Font, PatternFill
    from openpyxl.chart import BarChart, Reference
    for f in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 13,
                         'axes.labelweight': 'bold', 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    summary = read(out / 'numeric_summary.json')
    plan = read(out / 'plan.json')
    rows = summary['attributes']
    arr = np.array([[r[f'PC{i+1:02d}'] for i in range(47)] for r in rows]) * 100
    band = np.array([[r[k] for k in ['high', 'middle', 'low', 'outside']] for r in rows]) * 100
    data = np.load(out / 'analysis_vectors.npz')
    fig = plt.figure(figsize=(17, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 5], width_ratios=[47, 5], hspace=.12, wspace=.11)
    spectrum = fig.add_subplot(gs[0, 0])
    v = data['eigenvalues']
    spectrum.bar(np.arange(47), 100*v/v.sum(), color=['#2457A6']*10+['#789BC6']*27+['#C1D1E7']*10)
    spectrum.set_ylabel('Variance\nshare (×100)', fontsize=11)
    spectrum.set_xticks([])
    spectrum.set_xlim(-.5, 46.5)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(arr, cmap='YlOrRd', aspect='auto', vmin=0, vmax=float(arr.max()))
    ax.set_yticks(range(8), LABELS)
    ax.set_xticks([0, 9, 19, 29, 36, 46], [1, 10, 20, 30, 37, 47])
    ax.axvline(9.5, color='#222', linewidth=1)
    ax.axvline(36.5, color='#222', linewidth=1)
    ax.set_xlabel('Original principal component (descending variance)')
    for t in ax.get_yticklabels()[:2]:
        t.set_color('#2457A6'); t.set_weight('bold')
    outside = fig.add_subplot(gs[1, 1])
    outside.imshow(band[:, 3:], aspect='auto', cmap='Blues', vmin=0, vmax=100)
    outside.set_xticks([0], ['Outside\n47 PCs'])
    outside.set_yticks([])
    for i, value in enumerate(band[:, 3]):
        outside.text(0, i, f'{value:.2f}', ha='center', va='center', weight='bold', color='white' if value > 60 else 'black')
    cb = fig.colorbar(im, ax=[spectrum, ax], fraction=.022, pad=.018)
    cb.set_label('Edit energy in one PC (×100)', weight='bold')
    fig.subplots_adjust(left=.24, bottom=.13, right=.96, top=.98)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'attribute_pc_heatmap.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    def band_plot(matrix, name):
        fig, ax = plt.subplots(figsize=(10, 6.1))
        im = ax.imshow(matrix, aspect='auto', cmap='Blues', vmin=0, vmax=100)
        ax.set_yticks(range(8), LABELS)
        ax.set_xticks(range(4), ['High\nPC1–10', 'Middle\nPC11–37', 'Low\nPC38–47', 'Outside\n47 PCs'])
        ax.set_xlabel('Fixed original subspace; energy shares sum to 100 per row')
        for i in range(8):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', weight='bold',
                        color='white' if matrix[i,j] > 60 else '#152234')
        fig.colorbar(im, ax=ax, label='Edit energy share (×100)', fraction=.04, pad=.03)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    band_plot(band, 'attribute_band_heatmap')
    pm = list(csv.DictReader((out / 'per_member_energy.csv').open(encoding='utf-8-sig')))
    q16band = np.array([[np.mean([float(r[k]) for r in pm if r['attribute'] == a and
                                  r['memory_id'] in plan['predefined_case_subset']]) * 100
                         for k in ['high', 'middle', 'low', 'outside']] for a in ATTRS])
    band_plot(q16band, 'q16_two_cards_band_heatmap')

    # Editable data and native Excel chart, with the exact rendered plots included.
    wb = Workbook()
    about = wb.active; about.title = 'Readme'
    for row in [
        ['属性投影观察', 'GPT-5 caption / MemEye / Health master_000070'],
        ['范围', '48 条原记忆拟合 47 PCs；41 条满足五指标条件的记忆；943 个单因素改动'],
        ['诊断问题（非 benchmark 新题）', QUERY],
        ['数值角色', 'Meal Variability、Evening Drift 为所问指标；其他数值相对于该问题为对照'],
        ['公式', 'E_j = (delta_e dot u_j)^2 / ||delta_e||^2；未白化，未做行内重新归一化'],
        ['聚合', '先在同一记忆内平均四个数值步长，再对 41 条记忆平均'],
        ['Outside', '原始簇 47 维子空间以外的能量（剩余 337 维）'],
        ['重要限制', '合成文字干预不是真实原图变化，不改记忆库，不等于 QA/Recall 结果'],
        ['步长', '数值 ±1、±3；Evening Drift 按 ±0.1、±0.3 小时'],
        ['自然对照', 'numeric_summary.json 包含两组同图同 caption、不同对话的实际记忆'],
    ]:
        about.append(row)
    for sheet_name, file in [('Attributes', 'attribute_summary.csv'), ('Edits', 'edits.csv'),
                             ('PerMember', 'per_member_energy.csv'), ('AllProjections', 'projection_energy.csv')]:
        ws = wb.create_sheet(sheet_name)
        rs = list(csv.reader((out / file).open(encoding='utf-8-sig')))
        for ridx, row in enumerate(rs):
            if ridx:
                row = [float(v0) if re.fullmatch(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', v0, re.I) else v0 for v0 in row]
            ws.append(row)
        ws.freeze_panes = 'B2'; ws.auto_filter.ref = ws.dimensions
    ws = wb.create_sheet('BandHeatmap')
    ws.append(['Attribute', 'High PC1-10', 'Middle PC11-37', 'Low PC38-47', 'Outside 47 PCs'])
    for label, vals in zip(LABELS, band):
        ws.append([label, *map(float, vals)])
    ws.conditional_formatting.add('B2:E9', ColorScaleRule(start_type='num', start_value=0, start_color='FFFFFF',
                                                        end_type='num', end_value=100, end_color='2457A6'))
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=5):
        for cell in row: cell.number_format = '0.00'
    chart = BarChart(); chart.type = 'bar'; chart.grouping = 'stacked'; chart.overlap = 100
    chart.add_data(Reference(ws, min_col=2, max_col=5, min_row=1, max_row=9), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=9))
    chart.x_axis.title = 'Edit energy share (x100)'; chart.x_axis.scaling.max = 100
    chart.width = 26; chart.height = 13; ws.add_chart(chart, 'A12')
    plots = wb.create_sheet('Figures')
    for name, anchor, width in [('attribute_pc_heatmap.png', 'A1', 1250), ('attribute_band_heatmap.png', 'A40', 950)]:
        img = ExcelImage(out / name); ratio = width/img.width; img.width = width; img.height *= ratio; plots.add_image(img, anchor)
    for ws in wb:
        ws.column_dimensions['A'].width = 45
        ws.column_dimensions['B'].width = 24 if ws.title != 'Readme' else 115
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF'); cell.fill = PatternFill('solid', fgColor='2457A6')
    wb.save(out / 'Attribute_projection_heatmap.xlsx')

    h = lambda value: html.escape(str(value))
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>属性投影热力图</title>',
            '<style>body{font:16px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#182436}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{padding:16px;background:#eef3f9}details{margin:12px 0;padding:8px;border:1px solid #ddd}mark{background:#ffe7b1}</style>',
            '<h1>固定原簇 PCA 的属性投影热力图</h1>',
            '<p>GPT-5 caption / MemEye / Health / master_000070。原簇 48 条，47 个主成分；41 条记忆参与成对对照，943 个单因素文字改动。使用本地 BGE，外部 API 调用为 0。</p>',
            '<p class="note">这是文字表示的受控观察。数值与主题修改均为合成分析副本，不代表原图实际发生变化，也不写回记忆库。“相关/其他”仅相对于下述诊断问题定义。</p>',
            '<p><b>预先固定的诊断问题：</b>'+h(QUERY)+'。这是用于定义属性角色的读数问题，不是新增 benchmark 评测题。数值变化会改变所问的两个读数；其他指标、主题、措辞不改变这两个读数。</p>',
            '<h2>主成分投影</h2><img src="attribute_pc_heatmap.png"><p>上方柱状图为原簇方差谱。下面每个格子表示 E_j=(Δe·u_j)²/||Δe||²，数值乘 100。所有行共享同一色阶；右侧 Outside 使用独立 0–100 色阶。PCA 只用原始 48 条记忆拟合，改动后的文本不参与拟合，未做白化。</p>',
            '<h2>分段能量</h2><img src="attribute_band_heatmap.png"><p>高、低方差段均含 10 个方向，中间段 27 个方向，Outside 包含余下 337 维。每行四列之和为 100。大 Outside 表明原簇的变化范围不能完整覆盖新改动；不能把它叫做低方差图像细节。</p>',
            '<p>每个数值属性、每条记忆分别做 ±1、±3 个单位的改动；Evening Drift 单位为 0.1 小时，其余为 1。先在记忆内平均，再平均 41 条记忆。主题、caption 动词和助手回复各 1 个改动。不同干预的语义强度不完全可比，数值组之间是主要比较。</p>',
            '<h2>所问数值减其他数值：同记忆配对比较</h2><table><tr><th>方向段</th><th>能量差（×100）</th><th>成员 bootstrap 95% 区间（×100）</th></tr>']
    for k, val in summary['paired_numeric_comparison'].items():
        body.append(f'<tr><td>{h(k)}</td><td>{100*val["requested_minus_other"]:.2f}</td><td>{100*val["bootstrap95"][0]:.2f}, {100*val["bootstrap95"][1]:.2f}</td></tr>')
    body += ['</table><p>区间只描述这个簇内的探索性稳定程度；这些相似页面不是独立数据集，不据此声称跨域统计显著。没有根据结果重新选择属性、成员或主成分。</p>',
             '<h2>预先指定的 Health/Q16 两张 D5 卡片</h2><img src="q16_two_cards_band_heatmap.png"><p>仅 HEALTH_S5:R4 和 HEALTH_S5:R6，两条都来自此前已展示的 Q16。样本量小，仅供定位案例；主结果使用所有 41 条。</p>',
             '<h2>完整改动清单</h2><p>展开可检查每个改动的原文、改后文本及唯一替换片段。</p>']
    stimuli = lines(out / 'stimuli.jsonl')
    for s in stimuli:
        before = s['original_text']; start = s['span_start']; end = s['span_end']
        oldmarked = h(before[:start]) + '<mark>' + h(s['old']) + '</mark>' + h(before[end:])
        newmarked = h(before[:start]) + '<mark>' + h(s['new']) + '</mark>' + h(before[end:])
        body.append('<details><summary>'+h(s['id'])+' · '+h(s['attribute'])+' · '+h(s['old'])+' → '+h(s['new'])+'</summary><p>原文：</p><pre>'+oldmarked+'</pre><p>分析副本：</p><pre>'+newmarked+'</pre></details>')
    body += ['<h2>来源与数据</h2><p><a href="Attribute_projection_heatmap.xlsx">Excel（数据、热力图、可编辑图表）</a> · <a href="plan.json">编码前固定的设计与来源校验</a> · <a href="attribute_summary.csv">属性汇总</a> · <a href="projection_energy.csv">943 个改动的完整能量</a> · <a href="numeric_summary.json">运行及数值核验</a> · <a href="independent_audit.json">独立复核</a></p>',
             '<p>界限：投影能量不含查询方向，不等于目标排序贡献；无需也没有调用回答模型。未检验图像编码器的响应。相同图片、相同 caption、不同对话的自然对照保存在 numeric_summary.json，作为低方差可能包含表达变化的补充。</p></html>']
    (out / 'report.html').write_text('\n'.join(body), encoding='utf-8')
    dump(out / 'render_manifest.json', {'pngs': ['attribute_pc_heatmap.png', 'attribute_band_heatmap.png', 'q16_two_cards_band_heatmap.png'],
                                       'workbook': 'Attribute_projection_heatmap.xlsx', 'stimuli_in_html': len(stimuli)})
    print(json.dumps({'stage': 'rendered', 'output': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'run', 'render'])
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.stage == 'prepare': prepare(a.output.resolve())
    elif a.stage == 'run': run(a.output.resolve(), a.device)
    else: render(a.output.resolve())
