"""Independent numeric, provenance and deliverable audit of the Home/Q10 probes."""
from pathlib import Path
import csv
import hashlib
import json
import os
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/home_q10_query_attribute_projection_20261113'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    plan=read(OUT/'plan.json');summary=read(OUT/'numeric_summary.json');info=read(OUT/'question.json')
    stimuli=read(OUT/'stimuli.json');members=read(OUT/'members.json');z=np.load(OUT/'analysis_vectors.npz')
    assert sha(OUT/'plan.json')==summary['plan_sha256']
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    for p,h in plan['input_sha256'].items():assert sha(OUT/p)==h,p
    assert plan['queries']['raw_query']==info['question']['question']
    assert plan['queries']['frozen_master_query']==info['frozen_master']['augmented_query']
    assert info['frozen_master']['selected_choice']=='B' and info['gold']['gold_choice']=='C'
    assert info['frozen_master']['selected_statement']==info['question']['candidate_statements']['B']
    assert plan['queries']['frozen_master_query']==info['question']['question']+'\nCandidate answer: '+info['question']['candidate_statements']['B']
    assert len(stimuli)==len({s['id'] for s in stimuli})==16
    ti=next(i for i,m in enumerate(members) if m['id']==plan['target_id'])
    di=next(i for i,m in enumerate(members) if m['id']==plan['distractor_id'])
    for s in stimuli:
        assert s['original_text']==members[ti]['retrieval_text']
        expected=s['original_text']
        for r in s['replacements']:
            assert expected.count(r['old'])==1
            expected=expected.replace(r['old'],r['new'],1)
        assert expected==s['edited_text']
        if s['role'] in ['answer_fact','answer_relation','answer_invariant_visual']:
            assert expected.split('\nMemory description: ')[0]==members[ti]['dialogue']
        if s['role']=='episode_locator' or s['id'].startswith('ack_'):
            assert expected.split('\nMemory description: ')[1]==members[ti]['caption']
        if s['id'].startswith('monitor_remove_'):
            assert 'holding a black computer monitor' not in expected
            assert 'with a closed gray laptop and another black chair' in expected
        if s['id'].startswith('monitor_transfer_'):
            assert 'holding a black computer monitor' not in expected
            assert expected.count('black computer monitor')==1
        if s['id'].startswith(('wall_color_','carpet_color_','caption_paraphrase_','ack_')):
            assert 'holding a black computer monitor and mouse pad' in expected
            assert 'with a closed gray laptop and another black chair' in expected
    x=z['originals'];q=z['queries'];u=z['basis'];v=z['eigenvalues']
    assert x.shape==(5,384) and q.shape==(2,384) and u.shape==(4,384)
    for a in [x,q,z['edited'],z['repeated_target']]:np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-12,rtol=0)
    centered=x-x.mean(0)
    np.testing.assert_allclose(centered@u.T@u,centered,atol=1e-12,rtol=0)
    np.testing.assert_allclose(u@u.T,np.eye(4),atol=1e-12,rtol=0)
    np.testing.assert_allclose(u@(centered.T@centered/4)@u.T,np.diag(v),atol=1e-12,rtol=0)
    delta=z['edited']-x[ti]
    c=np.zeros((2,16,4));outside=np.zeros((2,16));total=q@delta.T
    for k in range(2):
        for i in range(16):
            for j in range(4):c[k,i,j]=float(q[k]@u[j])*float(u[j]@delta[i])
            residual=delta[i]-sum((delta[i]@a)*a for a in u)
            outside[k,i]=q[k]@residual
    np.testing.assert_allclose(c,z['contributions'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(outside,z['outside'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(total,z['total'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(c.sum(2)+outside,total,atol=1e-12,rtol=0)
    residual_matrix=delta-(delta@u.T)@u
    np.testing.assert_allclose((residual_matrix**2).sum(1)/(delta**2).sum(1),z['outside_energy'],atol=1e-12)
    # Explicit full matrix verifies the ridge intermediate changes without SVD summation.
    ridge_transform=.03*np.linalg.inv(centered.T@centered/4+.03*np.eye(384))
    white_delta=q@ridge_transform@delta.T
    rows=list(csv.DictReader((OUT/'query_attribute_contributions.csv').open(encoding='utf-8-sig')))
    assert len(rows)==32
    for k,name in enumerate(plan['queries']):
        original_raw=x@q[k];original_white=x@ridge_transform@q[k]
        white_z=(original_white-original_white.mean())/original_white.std()
        weighted=original_raw+original_raw.std()*np.tanh(white_z)
        row=next(r for r in summary['original_pair'] if r['query']==name)
        expected_pair=(q[k]@u.T)*(u@(x[ti]-x[di]))
        np.testing.assert_allclose(expected_pair,z['original_pair'][k],atol=1e-12,rtol=0)
        for field,expected in [('raw_margin',original_raw[ti]-original_raw[di]),
                               ('white_margin',original_white[ti]-original_white[di]),
                               ('zscore_final_margin_in_this_master',weighted[ti]-weighted[di])]:
            np.testing.assert_allclose(row[field],expected,atol=1e-12,rtol=0)
        for i,s in enumerate(stimuli):
            row=next(r for r in rows if r['query']==name and r['id']==s['id'])
            np.testing.assert_allclose([float(row[f'PC{j+1}']) for j in range(4)],c[k,i],atol=1e-12,rtol=0)
            np.testing.assert_allclose(float(row['outside_contribution']),outside[k,i],atol=1e-12)
            np.testing.assert_allclose(float(row['total_score_change']),total[k,i],atol=1e-12)
            np.testing.assert_allclose(float(row['white_score_change_frozen_basis']),white_delta[k,i],atol=1e-12)
    assert summary['source_score_max_error']<2e-6
    assert all(a['truncated_inputs']==0 for a in summary['encoding_audits'])
    workbook=OUT/'Home_Q10_query_attribute_projection.xlsx';wb=load_workbook(workbook)
    assert wb['Edits'].max_row==17 and wb['Contributions'].max_row==33
    assert len(wb['ScoreChangeChart']._charts)==1 and len(wb['ScoreChangeChart']._charts[0].series)==2
    assert len(wb['Figures']._images)==2
    for k,sheet in enumerate(['RawHeatmap','MasterHeatmap']):
        for i in range(16):
            expect=np.r_[c[k,i],outside[k,i],total[k,i]]*100
            np.testing.assert_allclose([wb[sheet].cell(i+2,j+2).value for j in range(6)],expect,atol=1e-10,rtol=0)
    with zipfile.ZipFile(workbook) as f:assert f.testzip() is None
    assert (OUT/'report.html').read_text().count('<details>')==21
    for m in members:
        name=m['id'].split('/')[-1].replace(':','_')+Path(m['image_path']).suffix
        assert sha(OUT/'images'/name)==m['image_sha256']
    for name in ['query_attribute_heatmap','original_pair_contributions','total_score_changes']:
        for ext in ['png','pdf']:assert (OUT/f'{name}.{ext}').stat().st_size>1000
    audit={'status':'passed','original_members':5,'fixed_pcs':4,'counterfactual_texts':16,'real_queries':2,
           'independently_checked_directional_terms':128,'complete_query_decomposition_checked':True,
           'ridge_full_matrix_check':True,'source_score_max_error':summary['source_score_max_error'],
           'source_files_verified':len(plan['source_sha256']),'original_images_verified':5,
           'edits_reconstructed_exactly':True,'workbook_native_charts':1,'workbook_embedded_plots':2,
           'memory_modified':False,'reader_calls':0,'api_calls':0,
           'audit_script_sha256':sha(__file__)}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    manifest={'status':'complete','audit':'passed','source_sha256':plan['source_sha256'],
              'output_sha256':{str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion_manifest.json'},
              'audit_script_sha256':sha(__file__),'render_script_sha256':sha(ROOT/'scripts/analysis/render_home_query_attribute_projection.py'),
              'memory_modified':False,'api_calls':0}
    (OUT/'completion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
