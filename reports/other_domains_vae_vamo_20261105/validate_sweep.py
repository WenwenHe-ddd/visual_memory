"""Independent artifact checks for the seven-domain VAE/VAMO sweep."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np

P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=read(P/'protocol.json')
assert read(P/'completed.json')['completed']
for path,h in protocol['source_sha256'].items():assert sha(path)==h,path
reports=[];all_scores=[];n_mem=0;n_clusters=0;n_eligible=0
for name in protocol['domains']:
    p=P/name;done=read(p/'completed.json');assert done['completed']
    metadata=read(p/'image_metadata.json');state=read(p/'memory/membership.json')
    n=len(metadata);assert n==done['n']
    u=np.load(p/'memory/basis.npy');sums=np.load(p/'memory/master_sums.npy');z=np.load(p/'memory/leaf_coords.npy')
    refs=np.load(p/'evaluation_only/full_latents.npy');recon=np.load(p/'evaluation_only/reconstructed_latents.npy')
    assert refs.shape==recon.shape==(n,4096) and z.shape==(n,2)
    assert u.shape==(4096,2) and np.max(np.abs(u.T@u-np.eye(2)))<1e-10
    pe=float(np.max(np.abs(refs@u-z)));assert pe<1e-8
    assert len(set(state['leaf_ids']))==n
    lookup={k:i for i,k in enumerate(state['leaf_ids'])}
    err=0.;sumerr=0.
    for j,m in enumerate(state['masters']):
        ids=[lookup[k] for k in m['member_ids']];mu=sums[j]/len(ids)
        expected=mu+(z[ids]-mu@u)@u.T
        err=max(err,float(np.max(np.abs(expected-recon[ids]))))
        sumerr=max(sumerr,float(np.max(np.abs(recon[ids].sum(0)-sums[j]))))
    assert err<1e-10 and sumerr<1e-8
    cal=read(p/'calibration.json')
    assert not set(cal['hashes']).intersection(m['sha256'] for m in metadata)
    metrics=rows(p/'per_image_metrics.csv');clusters=rows(p/'cluster_metrics.csv')
    assert len(metrics)==3*n and len({(r['image_id'],r['comparison']) for r in metrics})==3*n
    for r in metrics:
        for key in ['lpips','content_lpips','ssim','content_ssim','latent_cosine','latent_relative_l2']:assert math.isfinite(float(r[key]))
    for folder in ['original','vae_direct','vamo']:assert len(list((p/folder).glob('*.png')))==n
    for c in clusters:
        selected=[r for r in metrics if r['master_id']==c['master_id'] and r['comparison']=='vamo_vs_original']
        assert len(selected)==int(c['n'])
        for name_,key in [('content_lpips_mean','content_lpips'),('content_ssim_mean','content_ssim'),('content_psnr_mean','content_psnr')]:
            assert abs(float(c[name_])-np.mean([float(r[key]) for r in selected]))<1e-10
        if int(c['n'])>=3:
            assert (p/'clusters'/f"{c['master_label']}_all.pdf").exists()
            assert len(list((p/'clusters').glob(f"{c['master_label']}_page*.png")))==math.ceil(int(c['n'])/6)
    audits=[json.loads(x) for x in (p/'replay_audit.jsonl').read_text().splitlines()]
    assert len(audits)==n and max(a['max_projection_consistency_l2_error'] for a in audits)<1e-8
    n_mem+=n;n_clusters+=len(clusters);n_eligible+=sum(int(c['n'])>=3 for c in clusters)
    all_scores+=metrics
    reports.append(dict(domain=name,memories=n,masters=len(clusters),eligible_clusters=sum(int(c['n'])>=3 for c in clusters),
                        projection_error=pe,reconstruction_error=err,aggregate_error=sumerr,calibration_overlap=0))
assert (n_mem,n_clusters,n_eligible)==(408,80,46)
ranking=rows(P/'eligible_cluster_ranking.csv');assert len(ranking)==46
assert all(int(c['n'])>=3 for c in ranking)
assert [float(c['content_lpips_mean']) for c in ranking]==sorted(float(c['content_lpips_mean']) for c in ranking)
assert [int(c['rank']) for c in ranking]==list(range(1,47))
with (P/'all_per_image_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(all_scores[0]));w.writeheader();w.writerows(all_scores)
audit=dict(passed=True,domains=reports,total_memories=408,total_masters=80,eligible_clusters=46,metric_rows=1224,
           all_selected_cluster_members_exported=True,selection='Exploratory, mean content LPIPS ascending',
           source_hashes_unchanged=True,script_sha256={str(s):sha(s) for s in [P/'run_domains.py',Path(__file__)]})
(P/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
