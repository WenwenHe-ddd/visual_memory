#!/usr/bin/env python3
"""Evaluate expanded split/merge states with exact cache reuse and no rendering tricks."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/tmp/vamo_detail_mpl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch


def read(p): return json.loads(Path(p).read_text())
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(image_id,vector): return image_id,hashlib.sha256(np.asarray(vector,dtype=np.float64).tobytes()).hexdigest()


@torch.inference_mode()
def run(screen,device):
    base=ROOT/'scripts/analysis/screen_vamo_merge_decode.py'
    spec=importlib.util.spec_from_file_location('detail_decoder_helpers',base)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    helper=mod.load_helpers(device)
    old=ROOT/'reports/vamo_24_triptychs_screen_20261114'
    previous=read(old/'decoded/metrics.json')
    cached={}
    with np.load(old/'shortlist_vectors.npz') as vectors:
        for row in previous:
            for stage in ('after_split','before_merge','after_merge'):
                vector=vectors[f"{row['candidate_id']}__{stage}"]
                cached[signature(row['image_id'],vector)]={
                    'ssim':row[f'{stage}_ssim'],'lpips':row[f'{stage}_lpips'],'psnr':row[f'{stage}_psnr'],
                    'image_path':str(old/'decoded'/row['candidate_id']/f'{stage}.png'),
                    'origin':'previous_exact_vector_cache',
                }
    candidates=read(screen/'all_candidates.json')
    output=screen/'decoded';output.mkdir(exist_ok=False)
    (output/'states').mkdir()
    sources=[Path(__file__),base,screen/'all_candidates.json',screen/'state_vectors.npz',
             old/'shortlist_vectors.npz',old/'decoded/metrics.json']
    source_hashes={str(p):sha(p) for p in sources}
    # Only decode states actually appearing in a comparison, not evaluation references.
    required={}
    for row in candidates:
        for stage in ('after_split','before_merge','after_merge'):
            sid=row[stage+'_state']
            if sid in required: assert required[sid]['image_id']==row['image_id']
            required[sid]=row
    model=None; perceptual=None
    states={}; originals={}; manifests={}; decoded_count=0; reused=0
    from skimage.metrics import structural_similarity,peak_signal_noise_ratio
    with np.load(screen/'state_vectors.npz') as vectors:
        for sid,row in required.items():
            vector=vectors[sid]
            key=signature(row['image_id'],vector)
            if row['domain'] not in manifests:
                arr=read(helper.SOURCE/row['domain']/'input_manifest.json')['arrivals']
                manifests[row['domain']]={x['image_id']:x for x in arr}
            arrival=manifests[row['domain']][row['image_id']]
            if row['image_id'] not in originals:
                original,metadata=helper.prepare(arrival)
                original_path=output/'states'/f'original_{len(originals):04d}.png'
                helper.output_tensor(original,original_path)
                originals[row['image_id']]={'tensor':original,'metadata':metadata,'path':str(original_path)}
            if key in cached:
                states[sid]=dict(cached[key],state_id=sid,image_id=row['image_id'])
                reused+=1
                continue
            if model is None:
                import lpips
                model=helper.model_load()
                perceptual=lpips.LPIPS(net='alex',version='0.1',verbose=False).to(device).eval()
            original=originals[row['image_id']]['tensor'][None].to(device)
            x0,y0,x1,y1=originals[row['image_id']]['metadata']['content_box']
            image=model.decode(torch.tensor(vector.reshape(1,4,32,32),dtype=torch.float32,device=device)).sample.clamp(-1,1)
            path=output/'states'/f'{sid}.png';helper.output_tensor(image[0],path)
            truth=((original[0]+1)/2).permute(1,2,0).cpu().numpy()[y0:y1,x0:x1]
            pred=((image[0]+1)/2).permute(1,2,0).cpu().numpy()[y0:y1,x0:x1]
            score={'ssim':float(structural_similarity(truth,pred,data_range=1,channel_axis=2,
                gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),
                'psnr':float(peak_signal_noise_ratio(truth,pred,data_range=1)),
                'lpips':float(perceptual(image[:,:,y0:y1,x0:x1],original[:,:,y0:y1,x0:x1]).item()),
                'image_path':str(path),'origin':'decoded_frozen_vae'}
            cached[key]=score;states[sid]=dict(score,state_id=sid,image_id=row['image_id'])
            decoded_count+=1
            if decoded_count%25==0:print(json.dumps({'decoded_new':decoded_count,'states_scored':len(states),'states_required':len(required)}),flush=True)
    results=[]
    for row in candidates:
        result=dict(row,original_path=originals[row['image_id']]['path'])
        for stage in ('after_split','before_merge','after_merge'):
            score=states[row[stage+'_state']]
            for key in ('ssim','lpips','psnr','image_path'):result[f'{stage}_{key}']=score[key]
        result['lpips_improvement']=result['after_split_lpips']-result['after_merge_lpips']
        result['ssim_gain']=result['after_merge_ssim']-result['after_split_ssim']
        result['last_merge_lpips_improvement']=result['before_merge_lpips']-result['after_merge_lpips']
        result['last_merge_ssim_gain']=result['after_merge_ssim']-result['before_merge_ssim']
        result['both_overall_and_last_improved']=all(result[k]>0 for k in ('lpips_improvement','ssim_gain','last_merge_lpips_improvement','last_merge_ssim_gain'))
        results.append(result)
    dump(output/'state_metrics.json',states)
    dump(output/'metrics.json',results)
    # Prioritize absolute fidelity. Numerical positivity is only a browsing aid.
    qualified=[r for r in results if r['both_overall_and_last_improved']]
    best={}
    for row in qualified:
        score=(row['after_merge_lpips'],-row['last_merge_lpips_improvement'],-row['lpips_improvement'])
        if row['image_id'] not in best or score<best[row['image_id']][0]:best[row['image_id']]=(score,row)
    ranked=[x[1] for x in sorted(best.values(),key=lambda x:x[0])]
    dump(output/'ranked_unique_candidates.json',ranked)
    draw(output,ranked[:36])
    assert all(sha(path)==expected for path,expected in source_hashes.items())
    dump(output/'validation.json',{'passed':True,'completed':True,'candidate_count':len(candidates),
        'distinct_required_states':len(required),'new_decoder_forwards':decoded_count,'exact_cache_reuses':reused,
        'both_improved_pairs':len(qualified),'both_improved_unique_photos':len(ranked),
        'source_sha256':source_hashes,'selection':'Absolute final fidelity ranking for manual review; not a claim of visual acceptability.',
        'output_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'completed':True,'new_decoder_forwards':decoded_count,'unique_qualified':len(ranked)}),flush=True)


def draw(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    for start in range(0,len(rows),4):
        group=rows[start:start+4]
        fig,axes=plt.subplots(len(group),4,figsize=(13,len(group)*3.4),squeeze=False)
        for axrow,row in zip(axes,group):
            for ax,stage in zip(axrow,('original','after_split','before_merge','after_merge')):
                path=row['original_path'] if stage=='original' else row[stage+'_image_path']
                with Image.open(path) as im:ax.imshow(np.asarray(im),interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if stage=='original':ax.set_title(f"{row['candidate_id']} | {row['filename']}",fontsize=9)
                else:
                    step=row['split_step'] if stage=='after_split' else row['merge_step']-(stage=='before_merge')
                    ax.set_title(f'{stage} | t={step}',fontsize=9)
                    ax.set_xlabel(f"SSIM {row[stage+'_ssim']:.3f} | LPIPS {row[stage+'_lpips']:.3f}",fontsize=8)
        fig.tight_layout(h_pad=2)
        fig.savefig(output/f'review_{start//4+1:02d}.png',dpi=130,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(a.screen.resolve(),a.device)
