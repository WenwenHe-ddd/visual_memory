"""Screen every non-Brand domain with the same rank-2 VAE/VAMO protocol.

Selection is explicitly exploratory: rank clusters of >=3 members by mean
padding-excluded LPIPS, then inspect every member of shortlisted clusters.
"""
from pathlib import Path
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys, json, csv, importlib.util
import numpy as np
import torch
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'reports/brand_vae_vamo_20261105/run_experiment.py'
spec=importlib.util.spec_from_file_location('brand_experiment',BASE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
from scripts.graph.caption_ward_visual import update_visual_partition
from skimage.metrics import structural_similarity,peak_signal_noise_ratio

NAMES={'Card_Playlog_Test':'Card','Cartoon_Entertainment_Companion':'Cartoon',
       'Home_Renovation_Interior_Design':'Home','Multi-Scene_Visual_Case_Archive_Assistant':'Multi-Scene',
       'Outdoor_Navigation_Route_Memory_Assistant':'Outdoor','Personal_Health_Dashboard_Assistant':'Health',
       'Social_Chat_Memory_Test':'Social'}

def load_csv(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def audit_path(name):
    if name=='Home_Renovation_Interior_Design':return ROOT/'reports/home_caption_ward_visual_20261020/insertion_audit.jsonl'
    return b.SOURCE/name/'insertion_audit.jsonl'

def contact(directory, selected, metadata, title):
    # Every member of selected cluster, in original arrival order, never a subset.
    tile=256;gap=12;margin=20;header=82;rh=300
    canvas=Image.new('RGB',(832,header+rh*len(selected)+12),'white');d=ImageDraw.Draw(canvas)
    d.text((20,8),title,fill='#222222',font=b.font(22,True))
    for k,label in enumerate(['Input (256 x 256)','VAE direct','VAMO + VAE decode']):
        d.text((20+k*(tile+gap),45),label,fill='#333333',font=b.font(20,True))
    for j,i in enumerate(selected):
        r=metadata[i];y=header+j*rh
        label=f"{i+1:02d} | {r['filename']}"
        d.text((20,y),label[:85],fill='#333333',font=b.font(17))
        for k,folder in enumerate(['original','vae_direct','vamo']):
            with Image.open(directory/folder/f'{i+1:03d}.png') as im:
                canvas.paste(im.convert('RGB'),(20+k*(tile+gap),y+32))
    return canvas

@torch.inference_mode()
def run_domain(name,model,metric,streams,cache,hash_to_row):
    path=OUT/name;path.mkdir(exist_ok=True)
    if (path/'completed.json').exists():
        print('reuse completed',name,flush=True);return
    rows=streams[name];graph=b.read(b.SOURCE/name/'memory/graph.json')
    target_hashes={r['sha256'] for r in rows}
    # Preserve sorted other-domain/source order used by the Brand experiment.
    external={}
    for other in sorted(streams):
        if other==name:continue
        for r in streams[other]:
            if r['sha256'] not in target_hashes:external.setdefault(r['sha256'],r)
    assert not target_hashes.intersection(external)
    arr=cache[[hash_to_row[h] for h in external]].astype('float64')
    _,sv,vt=np.linalg.svd(arr-arr.mean(0),full_matrices=False)
    u=vt[:2].T.copy()
    for c in range(2):
        if u[np.argmax(np.abs(u[:,c])),c]<0:u[:,c]*=-1
    explained=float(np.sum(sv[:2]**2)/np.sum(sv**2));del arr,vt
    b.write(path/'calibration.json',dict(n=len(external),target_hash_overlap=0,top2_variance=explained,
             hashes=list(external),external_only=True))
    for folder in ['memory','evaluation_only','original','vae_direct','vamo','clusters']:(path/folder).mkdir(exist_ok=True)
    frozen=[json.loads(s) for s in audit_path(name).read_text().splitlines()]
    assert len(frozen)==len(rows)
    owners={n['id']:n['master_id'] for n in graph['image_nodes']}
    master_order=[m['id'] for m in graph['masters']]
    labels={k:f'M{i+1:02d}' for i,k in enumerate(master_order)}
    masters={};coords={};next_index=1;running_sum=np.zeros(4096);references=[];metadata=[];audits=[]
    for i,(row,saved) in enumerate(zip(rows,frozen)):
        assert saved['image_id']==row['image_id']
        x,meta=b.prepare(row)
        v=model.encode(x[None].to(b.DEVICE)).latent_dist.mode()[0].cpu().double().numpy().reshape(-1)
        # The helper sees only current v, prior coordinates, and master aggregates.
        masters,audit,next_index=update_visual_partition(masters,coords,saved['clusters_partition'],{row['image_id']:v},u,next_index)
        coords[row['image_id']]=u.T@v;running_sum+=v
        assert {k:tuple(m['member_ids']) for k,m in masters.items()}=={m['master_id']:tuple(m['member_ids']) for m in saved['clusters']}
        assert max(np.abs(sum(m['visual_sum'] for m in masters.values())-running_sum))<1e-8
        assert audit['max_projection_consistency_l2_error']<1e-8
        # Evaluation-only output; never read by update_visual_partition.
        references.append(v.copy())
        metadata.append(dict(index=i+1,image_id=row['image_id'],filename=row['filename'],sha256=row['sha256'],
                             master_id=owners[row['image_id']],master_label=labels[owners[row['image_id']]],**meta))
        audits.append(dict(step=i+1,**audit));del v
    mem=path/'memory'
    np.save(mem/'basis.npy',u)
    np.save(mem/'master_sums.npy',np.stack([masters[mid]['visual_sum'] for mid in master_order]))
    np.save(mem/'leaf_coords.npy',np.stack([coords[r['image_id']] for r in rows]))
    b.write(mem/'membership.json',dict(leaf_ids=[r['image_id'] for r in rows],master_ids=master_order,
            masters=[{k:v for k,v in masters[mid].items() if k!='visual_sum'} for mid in master_order]))
    b.write(path/'image_metadata.json',metadata)
    with (path/'replay_audit.jsonl').open('w') as f:
        for a in audits:f.write(json.dumps(a)+'\n')
    u=np.load(mem/'basis.npy');sums=np.load(mem/'master_sums.npy');zs=np.load(mem/'leaf_coords.npy')
    idx={r['image_id']:i for i,r in enumerate(rows)};rebuilt=np.empty((len(rows),4096))
    for j,mid in enumerate(master_order):
        group=[idx[k] for k in masters[mid]['member_ids']];mean=sums[j]/len(group)
        rebuilt[group]=mean+(zs[group]-mean@u)@u.T
        assert np.max(np.abs(rebuilt[group].sum(0)-sums[j]))<1e-8
    references=np.stack(references)
    assert np.max(np.abs(references@u-zs))<1e-8
    np.save(path/'evaluation_only/full_latents.npy',references)
    np.save(path/'evaluation_only/reconstructed_latents.npy',rebuilt)
    metrics=[]
    for i,row in enumerate(rows):
        orig=b.prepare(row)[0][None].to(b.DEVICE)
        direct=model.decode(torch.tensor(references[i].reshape(1,4,32,32),device=b.DEVICE,dtype=torch.float32)).sample.clamp(-1,1)
        vamo=model.decode(torch.tensor(rebuilt[i].reshape(1,4,32,32),device=b.DEVICE,dtype=torch.float32)).sample.clamp(-1,1)
        for folder,t in [('original',orig),('vae_direct',direct),('vamo',vamo)]:b.output_tensor(t[0],path/folder/f'{i+1:03d}.png')
        ref=references[i];v=rebuilt[i]
        cosine=float(v@ref/(np.linalg.norm(v)*np.linalg.norm(ref)))
        rel=float(np.linalg.norm(v-ref)/np.linalg.norm(ref))
        x0,y0,x1,y1=metadata[i]['content_box']
        # All content crops must be large enough for AlexNet/SSIM without resizing.
        assert min(y1-y0,x1-x0)>=64
        for comparison,pred,target in [('vae_vs_original',direct,orig),('vamo_vs_original',vamo,orig),('vamo_vs_vae',vamo,direct)]:
            p=(pred[0].permute(1,2,0).cpu().numpy()+1)/2;t=(target[0].permute(1,2,0).cpu().numpy()+1)/2
            pc=p[y0:y1,x0:x1];tc=t[y0:y1,x0:x1]
            kw=dict(data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
            metrics.append(dict(domain=name,short_domain=NAMES[name],index=i+1,image_id=row['image_id'],filename=row['filename'],
                  master_id=owners[row['image_id']],master_label=labels[owners[row['image_id']]],comparison=comparison,
                  psnr=float(peak_signal_noise_ratio(t,p,data_range=1)),ssim=float(structural_similarity(t,p,**kw)),
                  lpips=float(metric(pred,target).item()),content_psnr=float(peak_signal_noise_ratio(tc,pc,data_range=1)),
                  content_ssim=float(structural_similarity(tc,pc,**kw)),
                  content_lpips=float(metric(pred[:,:,y0:y1,x0:x1],target[:,:,y0:y1,x0:x1]).item()),
                  latent_cosine=cosine,latent_relative_l2=rel))
        if (i+1)%10==0 or i+1==len(rows):print(f'{NAMES[name]} evaluate {i+1}/{len(rows)}',flush=True)
    b.csv_write(path/'per_image_metrics.csv',metrics)
    clusters=[]
    for j,m in enumerate(graph['masters']):
        selected=[i for i,r in enumerate(metadata) if r['master_id']==m['id']]
        scores=[r for r in metrics if r['master_id']==m['id'] and r['comparison']=='vamo_vs_original']
        baseline=[r for r in metrics if r['master_id']==m['id'] and r['comparison']=='vae_vs_original']
        item=dict(domain=name,short_domain=NAMES[name],master_id=m['id'],master_label=labels[m['id']],n=m['n'],
                  unique_image_hashes=len({metadata[i]['sha256'] for i in selected}),eligible=m['n']>=3,
                  content_lpips_mean=float(np.mean([r['content_lpips'] for r in scores])),
                  content_lpips_worst=max(r['content_lpips'] for r in scores),
                  content_ssim_mean=float(np.mean([r['content_ssim'] for r in scores])),
                  content_ssim_worst=min(r['content_ssim'] for r in scores),
                  content_psnr_mean=float(np.mean([r['content_psnr'] for r in scores])),
                  latent_cosine_mean=float(np.mean([r['latent_cosine'] for r in scores])),
                  baseline_content_lpips_mean=float(np.mean([r['content_lpips'] for r in baseline])),
                  baseline_content_ssim_mean=float(np.mean([r['content_ssim'] for r in baseline])))
        clusters.append(item)
        if m['n']>=3:
            title=f"{NAMES[name]} | {labels[m['id']]} | n={m['n']} | LPIPS={item['content_lpips_mean']:.3f}"
            sheets=[]
            for offset in range(0,len(selected),6):
                im=contact(path,selected[offset:offset+6],metadata,title)
                im.save(path/'clusters'/f"{labels[m['id']]}_page{offset//6+1}.png");sheets.append(im)
            sheets[0].save(path/'clusters'/f"{labels[m['id']]}_all.pdf",save_all=True,append_images=sheets[1:],resolution=110)
    b.csv_write(path/'cluster_metrics.csv',clusters)
    b.write(path/'completed.json',dict(completed=True,domain=name,n=len(rows),masters=len(graph['masters']),
            eligible_clusters=sum(m['n']>=3 for m in graph['masters']),metric_rows=len(metrics),
            calibration_n=len(external),calibration_top2_variance=explained,source_audit=str(audit_path(name))))
    print('DOMAIN_COMPLETE',name,flush=True)

@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    streams={p.parent.name:b.read(p)['arrivals'] for p in sorted(b.SOURCE.glob('*/input_manifest.json'))}
    byhash={}
    for name in sorted(streams):
        for r in streams[name]:byhash.setdefault(r['sha256'],r)
    assert len(byhash)==433
    tracked={str(BASE):b.sha(BASE),str(ROOT/'scripts/graph/caption_ward_visual.py'):b.sha(ROOT/'scripts/graph/caption_ward_visual.py')}
    manifest=b.read(b.SOURCE/'manifest.json')
    for record in manifest['domains']:
        for f in ['input_manifest.json','memory/graph.json']:
            p=b.SOURCE/record['domain']/f;h=b.sha(p);assert h==record['sha256'][f];tracked[str(p)]=h
        if record['domain'] in NAMES:tracked[str(audit_path(record['domain']))]=b.sha(audit_path(record['domain']))
    brand_protocol=b.read(BASE.parent/'protocol.json')
    for name,h in brand_protocol['model_sha256'].items():assert b.sha(b.MODEL/name)==h
    protocol=dict(rank=2,preprocessing=brand_protocol['preprocessing'],model=brand_protocol['model_source'],
                  model_sha256=brand_protocol['model_sha256'],source_sha256=tracked,
                  source_experiment=str(BASE.parent),domains=list(NAMES),n_target_memories=408,
                  latent_l2_normalization=False,latent_dim=4096,chronological_memberships_frozen=True,
                  calibration='Per domain: all unique other-domain hashes, excluding all target hashes, centered PCA',
                  selection='Exploratory: clusters n>=3 ranked by mean content-only LPIPS; show ALL selected cluster members',
                  full_latent_cache='Evaluation/calibration-only; update helper never reads it; current arrival is reencoded once',
                  versions=brand_protocol['versions'])
    if (OUT/'protocol.json').exists():assert b.read(OUT/'protocol.json')==protocol
    else:b.write(OUT/'protocol.json',protocol)
    model=b.model_load()
    cache_dir=OUT/'calibration_only';cache_dir.mkdir(exist_ok=True)
    hashes=list(byhash);hash_to_row={h:i for i,h in enumerate(hashes)}
    cache_path=cache_dir/'all_unique_latents.npy'
    if cache_path.exists():
        assert b.read(cache_dir/'hash_order.json')==hashes;cache=np.load(cache_path)
    else:
        arrays=[]
        for start in range(0,len(hashes),8):
            x=torch.stack([b.prepare(byhash[h])[0] for h in hashes[start:start+8]]).to(b.DEVICE)
            arrays.append(model.encode(x).latent_dist.mode().cpu().numpy().reshape(len(x),-1))
            if start%64==0:print(f'calibration encoding {start}/{len(hashes)}',flush=True)
        cache=np.concatenate(arrays);np.save(cache_path,cache);b.write(cache_dir/'hash_order.json',hashes)
    assert cache.shape==(433,4096)
    import lpips
    metric=lpips.LPIPS(net='alex',version='0.1',verbose=False).to(b.DEVICE).eval()
    for name in NAMES:run_domain(name,model,metric,streams,cache,hash_to_row)
    for p,h in tracked.items():assert b.sha(p)==h
    combined=[]
    for name in NAMES:combined+=load_csv(OUT/name/'cluster_metrics.csv')
    eligible=[r for r in combined if int(r['n'])>=3]
    eligible.sort(key=lambda r:(float(r['content_lpips_mean']),-float(r['content_ssim_mean'])))
    for rank,r in enumerate(eligible,1):r['rank']=rank
    assert len(eligible)==46 and len(combined)==80
    b.csv_write(OUT/'all_cluster_metrics.csv',combined)
    b.csv_write(OUT/'eligible_cluster_ranking.csv',eligible)
    b.write(OUT/'completed.json',dict(completed=True,domains=7,memories=408,masters=80,eligible_clusters=46,
               metric_rows=408*3,source_files_unchanged=True))
    print('ALL_COMPLETE',flush=True)
    for r in eligible[:10]:print(r['short_domain'],r['master_label'],r['n'],r['content_lpips_mean'],r['content_ssim_mean'],flush=True)

if __name__=='__main__':main()
