"""Home/Q10 attribute probes with the actual raw and frozen master queries.

All edits are analysis-only counterfactuals of the original target caption or
dialogue. The original cluster and query construction remain frozen.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
                   'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4',
                   'MPLCONFIGDIR':'/tmp/visualmemory-matplotlib'}.items():
    os.environ.setdefault(key, value)
SOURCE = ROOT / 'results/memeye_gpt5_base_seed42_full_20261024'
ZSOURCE = ROOT / 'results/memeye_gpt5_qwen_five_validation_20261024'
OUT = ROOT / 'reports/home_q10_query_attribute_projection_20261113'
DOMAIN = 'Home_Renovation_Interior_Design'
QID = DOMAIN + '/Q10'
TARGET = DOMAIN + '/HOME_S7:R6'
DISTRACTOR = DOMAIN + '/HOME_S13:R2'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def dump(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def table(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def prepare(out):
    if (out/'plan.json').exists(): raise FileExistsError('Design already frozen; choose a fresh output root.')
    out.mkdir(parents=True, exist_ok=True)
    q = next(x for x in lines(SOURCE/'inputs/source_questions.jsonl') if x['id'] == QID)
    gold = next(x for x in lines(SOURCE/'inputs/evaluation.jsonl') if x['id'] == QID)
    previous = next(x for x in lines(SOURCE/'retrieval_results.jsonl') if x['question_id'] == QID)
    caps = {x['id']: x['caption'] for x in lines(SOURCE/'captions.jsonl')}
    ms = [x for x in lines(SOURCE/'inputs/source_memories.jsonl') if x['task'] == DOMAIN]
    graph = read(SOURCE/f'domains/{DOMAIN}/memory/graph.json')
    target = next(x for x in ms if x['id'] == TARGET)
    group = next(x for x in graph['masters'] if target['image_id'] in x['member_ids'])
    master = next(x for x in previous['masters'] if x['master_id'] == group['id'])
    members = []
    for image_id in group['member_ids']:
        m = dict(next(x for x in ms if x['image_id'] == image_id))
        m['caption'] = caps[m['image_sha256']]
        m['retrieval_text'] = m['dialogue'] + '\nMemory description: ' + m['caption']
        members.append(m)
    assert len(members) == 5 and master['selected_choice'] == 'B' and gold['gold_choice'] == 'C'
    assert gold['gold_ids'] == [TARGET] and not q['query_images']
    target = next(x for x in members if x['id'] == TARGET)
    assert previous['conditions'][1]['name'] == 'master_answer_query'
    assert previous['conditions'][1]['top5_memory_ids'][0] == DISTRACTOR
    stimuli = []
    def add(sid, label, role, replacements, semantics):
        before = target['retrieval_text']; after = before
        for old, new in replacements:
            assert after.count(old) == 1, (sid, old)
            after = after.replace(old, new, 1)
        assert after != before
        stimuli.append({'id':sid, 'label':label, 'role':role, 'memory_id':TARGET,
                        'replacements':[{'old':a,'new':b} for a,b in replacements],
                        'semantic_rationale':semantics, 'original_text':before, 'edited_text':after})
    holding = 'holding a black computer monitor and mouse pad'
    right = 'with a closed gray laptop and another black chair'
    for variant, new in [('v1','holding only a mouse pad'), ('v2','holding a mouse pad but no monitor')]:
        add('monitor_remove_'+variant, 'Remove window monitor / '+variant, 'answer_fact', [(holding,new)],
            'Remove the monitor from the window desk while the right desk remains without one. Counterfactual answer becomes Neither desk (B).')
    for variant, new, right_new in [
        ('v1','holding only a mouse pad','with a closed gray laptop, a black computer monitor, and another black chair'),
        ('v2','holding a mouse pad but no monitor','with a black computer monitor beside a closed gray laptop and another black chair')]:
        add('monitor_transfer_'+variant, 'Move monitor to right / '+variant, 'answer_relation', [(holding,new),(right,right_new)],
            'Move the same monitor from the window desk to the right-wall desk. Two coordinated replacements manipulate one ownership relation. Counterfactual answer becomes Right-wall desk (D).')
    phase = 'I tested the setup with both workstations in place.'
    for variant, new in [('v1','Here is the setup after a week of actual use.'),
                          ('v2','I am showing the workstations after using them for a week.')]:
        add('episode_'+variant, 'Change test to later use / '+variant, 'episode_locator', [(phase,new)],
            'Changes the dialogue identifying the requested workspace layout test to a later-use episode. The visible monitor answer stays the same, but memory selection relevance may change.')
    for color in ['white','silver']:
        add('monitor_color_'+color, 'Monitor black to '+color, 'answer_invariant_visual',
            [('black computer monitor',color+' computer monitor')],
            'Changes only monitor color. A monitor remains on the window desk, so the answer stays C. This does not assume zero effect on retrieval identity.')
    for color in ['blue','green']:
        add('wall_color_'+color, 'Walls light to '+color, 'answer_invariant_visual',
            [('light-colored walls',color+' walls')], 'Wall color changes; desk ownership and monitor presence stay fixed. Answer remains C.')
    for color in ['gray','brown']:
        add('carpet_color_'+color, 'Carpet beige to '+color, 'answer_invariant_visual',
            [('beige carpet',color+' carpet')], 'Carpet color changes; desk ownership and monitor presence stay fixed. Answer remains C.')
    for variant, new in [('v1','The picture depicts'),('v2','The photograph shows')]:
        add('caption_paraphrase_'+variant, 'Caption paraphrase / '+variant, 'expression_control',
            [('The image shows',new)], 'Opening paraphrase preserves all described objects, relations, and episode information.')
    ack = "That helps. I'll remember this layout."
    for variant, new in [('v1',"Understood. I'll remember this layout."),('v2',"Thanks. I'll remember this layout.")]:
        add('ack_'+variant, 'Acknowledgment / '+variant, 'expression_control', [(ack,new)],
            'Only the assistant acknowledgment changes; user event and caption remain exact.')
    dump(out/'members.json', members); dump(out/'question.json', {'question':q,'gold':gold,'frozen_master':master})
    dump(out/'stimuli.json', stimuli)
    table(out/'stimuli.csv', [{k:v for k,v in s.items() if k!='replacements'} for s in stimuli])
    files = [SOURCE/p for p in ['inputs/source_questions.jsonl','inputs/source_memories.jsonl',
                               'inputs/evaluation.jsonl','inputs/protocol.json','captions.jsonl',
                               'retrieval_results.jsonl',f'domains/{DOMAIN}/memory/graph.json']]
    files += [ZSOURCE/'retrieval_results.jsonl',Path(__file__)]
    files += [ROOT/p for p in ['scripts/memory/memeye_backbone_retrieval.py',
                              'scripts/memory/memeye_backbone_long_text.py',
                              'scripts/memory/caption_visual_master.py','scripts/memory/ridge_whitening.py',
                              'scripts/memory/online_zscore.py']]
    for m in members:
        assert sha(m['image_path']) == m['image_sha256']; files.append(Path(m['image_path']))
    plan = {'status':'fixed_before_projection','domain':DOMAIN,'question_id':QID,'master_id':group['id'],
            'caption_backbone':'gpt-5-2025-08-07','target_id':TARGET,'distractor_id':DISTRACTOR,
            'selection':'Home/Q10 selected for directly inspectable monitor ownership and previously verified original evidence, before new projection scores. Distractor is original master_answer_query rank 1 in same master.',
            'queries':{'raw_query':q['question'],'frozen_master_query':master['augmented_query']},
            'master_query_warning':'Actual frozen selected statement is B (neither desk has a monitor), whereas ground truth is C (window desk). It is retained verbatim, never corrected using gold.',
            'roles':'Roles were manually specified from the actual question, original image and dialogue. Answer-invariant visual edits preserve the answer, not necessarily retrieval similarity or episode identity.',
            'basis':'All 5 original dialogue+caption vectors in master_000124, centered SVD with 4 PCs. Not the global retrieval Top-5, and never refit on edited texts.',
            'units':'Original L2-normalized BGE embeddings, stored query prefix, no whitening before decomposition.',
            'measure':'Delta c_j=(q dot u_j)*(u_j dot (e_edited-e_original)); also q dot residual outside original 4-PC span.',
            'sign':'Positive raises cosine similarity to this specific query; negative lowers it. Positive is not automatically semantically correct or helpful.',
            'whitening_diagnostic':{'lambda':.03,'alpha':1,'per_pc_retention':'lambda/(v+lambda)',
                                    'outside_retention':1,'edits':'Frozen original covariance diagnostic only; no production pipeline rerun on edited memory.'},
            'original_pair':'c_j=(q dot u_j)*(u_j dot (e_target-e_distractor)); original pair is in the original cluster span.',
            'sample_size':'One question, one target memory, 16 explicitly listed counterfactual text edits; no population significance or accuracy claims.',
            'image_observation':'D7_IMG_005 visually has a monitor on the window desk, a closed laptop and no monitor on right-wall desk. D13_IMG_001 is a later-use desk close-up, identified by its original dialogue.',
            'limits':['No original images or captions are modified; edited texts are synthetic probes.',
                      'There are only 4 PCs in this cluster; no claim about general high/low semantic factors.',
                      'Counterfactual edits change evidence and can invalidate original gold; they are not evaluated against unchanged QA labels.',
                      'Two query variants are genuine frozen retrieval inputs; neither is an analyst-invented diagnostic query.',
                      'Text probes cannot establish how visual embeddings respond to image edits.'],
            'source_sha256':{str(p):sha(p) for p in files},
            'input_sha256':{n:sha(out/n) for n in ['members.json','question.json','stimuli.json','stimuli.csv']},
            'memory_modified':False,'reader_calls':0,'api_calls':0}
    assert len(stimuli)==16
    dump(out/'plan.json',plan)
    print(json.dumps({'stage':'prepared','question':QID,'members':5,'pcs':4,'edits':len(stimuli),'queries':plan['queries']},ensure_ascii=False),flush=True)


def run(out,device):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    from scripts.memory.ridge_whitening import fit, standardize
    from scripts.memory.online_zscore import ZScoreWeighting
    plan=read(out/'plan.json')
    if (out/'completion_manifest.json').exists():raise FileExistsError('Completed output is immutable.')
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    for p,h in plan['input_sha256'].items():assert sha(out/p)==h,p
    members=read(out/'members.json');stimuli=read(out/'stimuli.json');info=read(out/'question.json')
    args=SimpleNamespace(device=device);original.configure_device(args)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    prep=read(SOURCE/'inputs/protocol.json');encoder=original.load_text_encoder(prep['text_encoder'],args.device)
    ms,qs,caption_model=original.attach_captions(lines(SOURCE/'inputs/source_memories.jsonl'),
        lines(SOURCE/'inputs/source_questions.jsonl'),lines(SOURCE/'captions.jsonl'))
    assert caption_model==plan['caption_backbone']
    ms=[m for m in ms if m['task']==DOMAIN];qs=[q for q in qs if q['task']==DOMAIN]
    e,ei=encode_complete(encoder,[m['retrieval_text'] for m in ms],ids=[m['id'] for m in ms],field='original_domain')
    texts=[];ids=[]
    for q in qs:
        texts.extend([q['retrieval_query']]+[q['retrieval_query']+'\nCandidate answer: '+q['candidate_statements'][k] for k in 'ABCD'])
        ids.extend(q['id']+s for s in ['/baseline','/A','/B','/C','/D'])
    q_all,qi=encode_complete(encoder,texts,prep['query_prefix'],ids=ids,field='original_queries')
    q_all=q_all.reshape(len(qs),5,-1);iq=next(i for i,q in enumerate(qs) if q['id']==QID)
    queries=np.stack([q_all[iq,0],q_all[iq,1+'ABCD'.index(info['frozen_master']['selected_choice'])]])
    qnames=list(plan['queries'])
    imap={m['id']:i for i,m in enumerate(ms)}
    x=e[[imap[m['id']] for m in members]];ti=next(i for i,m in enumerate(members) if m['id']==TARGET);di=next(i for i,m in enumerate(members) if m['id']==DISTRACTOR)
    u,v=fit(x)
    for row in u:
        if row[np.argmax(abs(row))]<0:row*=-1
    assert len(v)==4 and (v>0).all()
    edited,pi=encode_complete(encoder,[s['edited_text'] for s in stimuli],ids=[s['id'] for s in stimuli],field='counterfactual_target_copies')
    repeat,ri=encode_complete(encoder,[members[ti]['retrieval_text']],ids=[TARGET],field='identity_control')
    repeat_error=float(np.linalg.norm(repeat[0]-x[ti]));assert repeat_error<2e-6
    assert all(a['truncated_inputs']==0 for a in [ei,qi,pi,ri])
    delta=edited-x[ti];coefficients=delta@u.T;residual=delta-coefficients@u
    qproj=queries@u.T;contributions=coefficients[None,:,:]*qproj[:,None,:]
    outside=queries@residual.T;total=queries@delta.T
    np.testing.assert_allclose(contributions.sum(2)+outside,total,atol=1e-12,rtol=0)
    retention=.03/(v+.03)
    white_delta=(contributions*retention).sum(2)+outside
    energy=(coefficients**2)/(delta**2).sum(1)[:,None]
    outside_energy=(residual**2).sum(1)/(delta**2).sum(1)
    np.testing.assert_allclose(energy.sum(1)+outside_energy,1,atol=1e-12)
    original_ret=next(r for r in lines(SOURCE/'retrieval_results.jsonl') if r['question_id']==QID)
    zret=next(r for r in lines(ZSOURCE/'retrieval_results.jsonl') if r['question_id']==QID)
    max_old_error=0.;pair_rows=[];original_rows=[];perturb_rows=[]
    original_pair=[]
    for a,(name,q) in enumerate(zip(qnames,queries)):
        r=x@q;white=r-(x@u.T)@((v/(v+.03))*(u@q))
        z,_=standardize(white);weighted=r+r.std()*np.tanh(z)
        c=(u@(x[ti]-x[di]))*(u@q);original_pair.append(c)
        np.testing.assert_allclose(c.sum(),r[ti]-r[di],atol=1e-12)
        np.testing.assert_allclose((c*retention).sum(),white[ti]-white[di],atol=1e-12)
        if name=='frozen_master_query':
            saved={r['memory_id']:r for r in next(c0 for c0 in zret['conditions'] if c0['name']=='master_answer_query__l0p03_a1')['leaves']}
            for j,m in enumerate(members):
                for field,actual in [('raw',r[j]),('white',white[j]),('weighted',weighted[j])]:
                    err=abs(actual-saved[m['id']][field]);max_old_error=max(max_old_error,float(err));assert err<2e-6
        cond=next(c0 for c0 in original_ret['conditions'] if c0['name']==('baseline_question_only' if name=='raw_query' else 'master_answer_query'))
        for j,m in enumerate(members):
            if m['id'] in cond['top5_memory_ids']:
                err=abs(r[j]-cond['top5_scores'][cond['top5_memory_ids'].index(m['id'])]);max_old_error=max(max_old_error,float(err));assert err<2e-6
            original_rows.append({'query':name,'memory_id':m['id'],'raw':float(r[j]),'white':float(white[j]),
                'zscore_final_in_this_master':float(weighted[j]),'original_global_rank':cond['full_rank_memory_ids'].index(m['id'])+1})
        pair_rows.append({'query':name,**{f'PC{j+1}':float(c[j]) for j in range(4)},
                          'raw_margin':float(r[ti]-r[di]),'white_margin':float(white[ti]-white[di]),
                          'zscore_final_margin_in_this_master':float(weighted[ti]-weighted[di]),
                          'raw_target_rank_global':cond['full_rank_memory_ids'].index(TARGET)+1,
                          'raw_distractor_rank_global':cond['full_rank_memory_ids'].index(DISTRACTOR)+1})
        for i,s in enumerate(stimuli):
            # Original target identity is only a reference: counterfactual answer facts may change.
            perturb_rows.append({'id':s['id'],'query':name,'label':s['label'],'role':s['role'],
                **{f'PC{j+1}':float(contributions[a,i,j]) for j in range(4)},'outside_contribution':float(outside[a,i]),
                'total_score_change':float(total[a,i]),'white_score_change_frozen_basis':float(white_delta[a,i]),
                'original_target_score':float(r[ti]),'edited_target_score':float(r[ti]+total[a,i]),
                'original_pair_margin':float(r[ti]-r[di]),'counterfactual_pair_margin':float(r[ti]-r[di]+total[a,i]),
                'delta_l2':float(np.linalg.norm(delta[i]))})
    table(out/'query_attribute_contributions.csv',perturb_rows);table(out/'original_pair_contributions.csv',pair_rows)
    table(out/'original_member_scores.csv',original_rows)
    table(out/'attribute_energy.csv',[{'id':s['id'],'label':s['label'],
            **{f'PC{j+1}':float(energy[i,j]) for j in range(4)},'outside':float(outside_energy[i])} for i,s in enumerate(stimuli)])
    table(out/'spectrum.csv',[{'pc':i+1,'variance':float(value),'variance_share':float(value/v.sum()),
                              'ridge_retention':float(retention[i])} for i,value in enumerate(v)])
    np.savez_compressed(out/'analysis_vectors.npz',originals=x,queries=queries,basis=u,eigenvalues=v,
                        edited=edited,repeated_target=repeat,contributions=contributions,outside=outside,
                        total=total,energy=energy,outside_energy=outside_energy,original_pair=np.array(original_pair))
    summary={'status':'numeric_complete','question_id':QID,'target_id':TARGET,'distractor_id':DISTRACTOR,
             'plan_sha256':sha(out/'plan.json'),'members':5,'pcs':4,'edits':len(stimuli),'query_count':2,
             'queries':plan['queries'],'frozen_master_selected_choice':'B','gold_choice':'C',
             'source_score_max_error':max_old_error,'identity_reencode_l2':repeat_error,
             'decomposition_max_error':float(abs(contributions.sum(2)+outside-total).max()),
             'minimum_edit_l2':float(np.linalg.norm(delta,axis=1).min()),
             'variance_shares':[float(w) for w in v/v.sum()], 'original_pair':pair_rows,
             'encoding_audits':[ei,qi,pi,ri], 'runtime':{'device':args.device,'gpu':torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,'torch':torch.__version__},
             'not_qa_evaluation':True,'api_calls':0,'memory_modified':False}
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    dump(out/'numeric_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.device)
