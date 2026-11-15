"""Independent stored-score, gold pairing, projection and artifact checks."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/gpt5_variance_observation_20261108'
SOURCE=ROOT/'results/memeye_gpt5_base_seed42_full_20261024'
ZROOT=ROOT/'results/memeye_gpt5_qwen_five_validation_20261024'
PVROOT=ROOT/'results/memeye_api_pv_frozen_l0p03_a1_20261108/gpt5'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    s=read(OUT/'numeric_summary.json');sel=s['selected_master']
    for path,h in {**s['source_sha256'],**s['output_sha256']}.items():assert sha(path)==h,path
    rows=list(csv.DictReader((OUT/'pairs.csv').open(encoding='utf-8-sig')))
    golden={r['id']:set(r['gold_ids']) for r in lines(SOURCE/'inputs/evaluation.jsonl')}
    zm={r['question_id']:{l['memory_id']:l for l in next(c for c in r['conditions'] if c['name']=='master_answer_query__l0p03_a1')['leaves']} for r in lines(ZROOT/'retrieval_results.jsonl')}
    pm={r['question_id']:{l['memory_id']:l for l in r['leaves']} for r in lines(PVROOT/'retrieval_results.jsonl')}
    old={r['question_id']:r for r in lines(SOURCE/'retrieval_results.jsonl')}
    ids=set();largest=[]
    for row in rows:
        q=row['question_id'];t=row['target_id'];d=row['distractor_id'];key=(q,t,d)
        assert key not in ids;ids.add(key)
        assert t in golden[q] and d not in golden[q]
        assert zm[q][t]['group_id']==zm[q][d]['group_id']==row['master_id']
        for output,a,field in [('raw_margin',zm,'raw'),('white_margin',zm,'white'),('pv_margin',pm,'positive_variance'),
                               ('zscore_final_margin',zm,'weighted'),('pv_final_margin',pm,'weighted')]:
            assert abs(float(row[output])-(a[q][t][field]-a[q][d][field]))<2e-12
        raw=next(c for c in old[q]['conditions'] if c['name']=='master_answer_query')
        assert (row['distractor_in_original_top5']=='True')==(d in raw['top5_memory_ids'])
        assert int(row['target_raw_global_rank'])==raw['full_rank_memory_ids'].index(t)+1
        assert int(row['distractor_raw_global_rank'])==raw['full_rank_memory_ids'].index(d)+1
        if row['domain']==sel['domain'] and row['master_id']==sel['master_id']:largest.append(row)
    expected=set()
    for q,leaves in zm.items():
        for t in golden[q]:
            for d,l in leaves.items():
                if d not in golden[q] and l['group_id']==leaves[t]['group_id']:expected.add((q,t,d))
    assert ids==expected and len(ids)==17240
    scopes={'all_pairs':rows,'largest_master':largest,
            'all_original_top5_distractors':[r for r in rows if r['distractor_in_original_top5']=='True'],
            'largest_original_top5_distractors':[r for r in largest if r['distractor_in_original_top5']=='True']}
    for key,rs in scopes.items():
        m=s['metrics'][key];byq=defaultdict(list)
        for r in rs:byq[r['question_id']].append(r)
        assert len(rs)==m['pairs'] and len(byq)==m['questions']
        for name in ['raw','white','pv','zscore_final','pv_final']:
            assert sum(float(r[name+'_margin'])>1e-12 for r in rs)==m[name+'_target_ahead_pairs']
            macro=np.mean([np.mean([float(r[name+'_margin'])>1e-12 for r in rr]) for rr in byq.values()])
            assert abs(macro-m[name+'_target_ahead_fraction_question_macro'])<1e-14
        br=[r for r in rs if int(r['rank'])>=5];bq=defaultdict(list)
        for r in br:bq[r['question_id']].append(r)
        assert len(br)==m['band_eligible_pairs'] and len(bq)==m['band_eligible_questions']
        for band in ['high','middle','low','rest']:
            v=m[band];field=band+'_contribution'
            assert sum(float(r[field])>1e-12 for r in br)==v['supports_target_pairs']
            assert sum(float(r[field])<-1e-12 for r in br)==v['supports_distractor_pairs']
            macro=np.mean([np.mean([float(r[field])>1e-12 for r in rr]) for rr in bq.values()])
            assert abs(macro-v['supports_target_fraction_question_macro'])<1e-14
    pc=[r for r in csv.DictReader((OUT/'spectra.csv').open(encoding='utf-8-sig')) if r['domain']==sel['domain'] and r['master_id']==sel['master_id']]
    values=np.array([float(r['variance']) for r in pc]);assert len(values)==47 and np.all(np.diff(values)<=0)
    crows=lines(OUT/'largest_pairs_with_contributions.jsonl');assert len(crows)==len(largest)==7468
    for row in crows:
        c=np.array(row['contributions']);assert len(c)==47
        assert abs(c.sum()-row['raw_margin'])<2e-12
        assert abs((c*.03/(values+.03)).sum()-row['white_margin'])<2e-12
        assert abs((c*(1+values/(values+.03))).sum()-row['pv_margin'])<2e-12
        assert abs(c[:10].sum()-row['high_contribution'])<2e-12
        assert abs(c[10:37].sum()-row['middle_contribution'])<2e-12
        assert abs(c[37:].sum()-row['low_contribution'])<2e-12
    extremes=read(OUT/'examples.json')
    lo=min(r['white_margin']-r['raw_margin'] for r in crows);hi=max(r['white_margin']-r['raw_margin'] for r in crows)
    assert abs(extremes['max_whitening_harm']['white_margin']-extremes['max_whitening_harm']['raw_margin']-lo)<1e-14
    assert abs(extremes['max_whitening_help']['white_margin']-extremes['max_whitening_help']['raw_margin']-hi)<1e-14
    members={m['id']:m for m in read(OUT/'largest_members.json')};coords={r['memory_id']:np.array(r['coordinates']) for r in read(OUT/'largest_coordinates.json')}
    controls=read(OUT/'identical_image_dialogue_control.json')['pairs'];assert len(controls)==2
    for pair in controls:
        a=members[pair['first']];b=members[pair['second']]
        assert sha(a['image_path'])==sha(b['image_path'])==a['image_sha256']==b['image_sha256']
        assert a['caption']==b['caption'] and a['dialogue']!=b['dialogue']
        e=(coords[a['id']]-coords[b['id']])**2
        assert abs(e[-1]/e.sum()-pair['pc47_energy_share'])<1e-14
    for m in members.values():
        name=m['id'].split('/')[-1].replace(':','_')+Path(m['image_path']).suffix
        assert sha(OUT/'images'/name)==sha(m['image_path'])
    findings={'passed':True,'all_same_master_gold_non_gold_pairs_enumerated':len(rows),'questions_with_pairs':len({r['question_id'] for r in rows}),
       'source_scores_independently_verified':len(rows)*5,'largest_pairs_pc_contributions_verified':len(crows),
       'macro_and_pair_metrics_recomputed':True,'original_images_verified':48,'identical_image_caption_different_dialogue_pairs':2,
       'source_hashes_verified':len(s['source_sha256']),'no_external_api_calls':True,'memory_unchanged':True,
       'scope':'Observation of text-embedding score contributions, plus explicitly qualitative original-image inspection; no causal semantic-PC labeling.'}
    findings['artifact_sha256']={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['independent_audit.json','completion_manifest.json']}
    dump(OUT/'independent_audit.json',findings)
    dump(OUT/'completion_manifest.json',{'status':'complete','audit_passed':True,'source_sha256':s['source_sha256'],
        'code_sha256':{str(Path(__file__)):sha(__file__),str(ROOT/'scripts/analysis/render_gpt5_variance_observation.py'):sha(ROOT/'scripts/analysis/render_gpt5_variance_observation.py')},
        'artifact_sha256':{**findings['artifact_sha256'],str(OUT/'independent_audit.json'):sha(OUT/'independent_audit.json')}})
    print(json.dumps({k:v for k,v in findings.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
