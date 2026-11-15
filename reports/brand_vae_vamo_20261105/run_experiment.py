"""Frozen Brand-caption partitions, VAMO replay on raw VAE posterior means.

Independent evaluation reference tensors never enter the memory update helper.
Run direct first, inspect it, then full. Inference is Python; no learned decoder
maps the existing CLIP features. This is a VAE-latent adaptation of VAMO.
"""
from pathlib import Path
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['TORCH_HOME'] = str(ROOT / 'models/torch-metrics')
import argparse
import csv
import hashlib
import json
import time
import importlib.metadata
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scripts.graph.caption_ward_visual import update_visual_partition

SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
DOMAIN = SOURCE / 'Brand_Memory_Test'
MODEL = ROOT / 'models/sd-vae-ft-mse'
EVAL = OUT / 'evaluation_only'
MEMORY = OUT / 'memory'
DEVICE = 'cuda:0'
SIZE = 256

def read(p): return json.loads(Path(p).read_text())
def write(p, d): Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def csv_write(p, rows):
    with Path(p).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def prepare(row):
    p = Path(row['source_path'])
    assert sha(p) == row['sha256'], p
    with Image.open(p) as src:
        im = ImageOps.exif_transpose(src).convert('RGB')
        original_size = im.size
        small = ImageOps.contain(im, (SIZE, SIZE), Image.Resampling.LANCZOS)
    x, y = (SIZE-small.width)//2, (SIZE-small.height)//2
    result = Image.new('RGB', (SIZE, SIZE), (127,127,127))
    result.paste(small, (x,y))
    arr = np.asarray(result, dtype=np.float32) / 255
    tensor = torch.from_numpy(arr.copy()).permute(2,0,1)*2-1
    return tensor, dict(original_size=original_size, content_box=[x,y,x+small.width,y+small.height])

def output_tensor(tensor, path):
    x = ((tensor.detach().cpu().float().clamp(-1,1)+1)/2).permute(1,2,0).numpy()
    Image.fromarray(np.rint(x*255).astype('uint8')).save(path)

def model_load():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    assert torch.cuda.is_available(), 'GPU unavailable; run with host GPU access'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = AutoencoderKL.from_pretrained(str(MODEL), local_files_only=True,
                                          torch_dtype=torch.float32, use_safetensors=True)
    return model.to(DEVICE).eval().requires_grad_(False)

def sources():
    manifest = read(SOURCE/'manifest.json')
    tracked = {str(SOURCE/'manifest.json'):sha(SOURCE/'manifest.json')}
    for r in manifest['domains']:
        names = ['input_manifest.json']
        if r['domain']=='Brand_Memory_Test':
            names += ['memory/graph.json','insertion_audit.jsonl']
        for name in names:
            p = SOURCE/r['domain']/name
            actual = sha(p)
            if name in r['sha256']: assert actual == r['sha256'][name]
            tracked[str(p)] = actual
    brand = read(DOMAIN/'input_manifest.json')['arrivals']
    target_hashes = {r['sha256'] for r in brand}
    external = {}
    for p in sorted(SOURCE.glob('*/input_manifest.json')):
        if p.parent == DOMAIN: continue
        for row in read(p)['arrivals']:
            if row['sha256'] not in target_hashes: external.setdefault(row['sha256'], row)
    assert len(brand)==30 and len(external)==403 and not target_hashes.intersection(external)
    return brand, list(external.values()), tracked

def identity_protocol(tracked):
    return dict(model_source=read(OUT/'model_source.json'), model_sha256={f:sha(MODEL/f) for f in ['config.json','diffusion_pytorch_model.safetensors']},
                source_sha256=tracked, size=SIZE, preprocessing='EXIF transpose, RGB, aspect-preserving Lanczos resize and centered 127-gray letterbox',
                latent='posterior mode/mean, 4x32x32; flattened 4096; no L2 normalization; no diffusion scaling factor',
                dtype='VAE float32; memory/PCA float64', seed=42, latent_rank=2,
                frozen_caption_partitions=True, reference_data_directory='evaluation_only',
                shared_basis='Centered PCA on 403 unique non-Brand images, coordinates U.T @ raw_latent',
                decoder_finetuned=False, existing_clip_experiment_modified=False,
                preview_selection='First arrival in each of the eight final masters, fixed before outcomes',
                versions={p:importlib.metadata.version(p) for p in ['torch','diffusers','transformers','huggingface-hub','torchvision','scikit-image','lpips','numpy']},
                device=torch.cuda.get_device_name(0), metric_policy='clamped float images [0,1], macro mean across 30; LPIPS AlexNet v0.1; SSIM Gaussian sigma=1.5 sample_covariance=False; extra PSNR/SSIM exclude padding')

def font(size, bold=False):
    return ImageFont.truetype('/storage/hewenwen/.local/share/fonts/TimesNewRoman/'+('Timesbd.TTF' if bold else 'Times.TTF'), size)

def montage(indices, metadata, title, full=False, tile=256):
    columns = [('Input (256 x 256)','original'), ('VAE direct','vae_direct')]
    if full: columns.append(('VAMO + VAE decode','vamo'))
    margin, gap, label_h, header = 20, 12, 35, 75
    w = margin*2 + len(columns)*tile + gap*(len(columns)-1)
    h = header+len(indices)*(tile+label_h+gap)+12
    canvas = Image.new('RGB',(w,h),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((margin,8),title,fill='#222222',font=font(22,True))
    for c,(name,_) in enumerate(columns): draw.text((margin+c*(tile+gap),43),name,fill='#333333',font=font(20,True))
    for j,i in enumerate(indices):
        y=header+j*(tile+label_h+gap)
        row=metadata[i]
        draw.text((margin,y),f"{i+1:02d} | {row['master_label']} | {row['filename']}",fill='#333333',font=font(18))
        for c,(_,folder) in enumerate(columns):
            im=Image.open(OUT/folder/f'{i+1:02d}.png').convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS)
            canvas.paste(im,(margin+c*(tile+gap),y+label_h))
    return canvas

@torch.inference_mode()
def direct(model, brand):
    EVAL.mkdir(exist_ok=True)
    for name in ['original','vae_direct']: (OUT/name).mkdir(exist_ok=True)
    graph=read(DOMAIN/'memory/graph.json')
    labels={m['id']:f'M{k+1}' for k,m in enumerate(graph['masters'])}
    owners={n['id']:n['master_id'] for n in graph['image_nodes']}
    arrays=[]; metadata=[]
    for i,row in enumerate(brand):
        x,meta=prepare(row); z=model.encode(x[None].to(DEVICE)).latent_dist.mode()
        assert tuple(z.shape)==(1,4,32,32)
        pred=model.decode(z).sample[0]
        output_tensor(x,OUT/'original'/f'{i+1:02d}.png')
        output_tensor(pred,OUT/'vae_direct'/f'{i+1:02d}.png')
        arrays.append(z[0].cpu().numpy())
        metadata.append(dict(index=i+1,image_id=row['image_id'],filename=row['filename'],master_id=owners[row['image_id']],
                             master_label=labels[owners[row['image_id']]],**meta))
        print(f'direct {i+1}/30 {row["filename"]}',flush=True)
    np.save(EVAL/'full_latents.npy',np.stack(arrays))
    write(OUT/'image_metadata.json',metadata)
    selected=[next(i for i,r in enumerate(metadata) if r['master_id']==m['id']) for m in graph['masters']]
    write(OUT/'preview_selection.json',selected)
    montage(selected,metadata,'Brand: VAE direct reconstruction').save(OUT/'direct_preview.png')
    print('DIRECT_COMPLETE',flush=True)

@torch.inference_mode()
def full(model, brand, external, tracked):
    metadata=read(OUT/'image_metadata.json')
    print('Encoding 403 external calibration images',flush=True)
    cal=[]
    for start in range(0,len(external),8):
        batch=torch.stack([prepare(r)[0] for r in external[start:start+8]]).to(DEVICE)
        z=model.encode(batch).latent_dist.mode().cpu().double().numpy().reshape(len(batch),-1)
        cal.append(z)
        print(f'calibration {min(start+8,len(external))}/403',flush=True)
    cal=np.concatenate(cal); cal_mean=cal.mean(0)
    _,sv,vt=np.linalg.svd(cal-cal_mean,full_matrices=False)
    basis=vt[:2].T.copy()
    for c in range(2):
        if basis[np.argmax(np.abs(basis[:,c])),c]<0: basis[:,c]*=-1
    explained=float(np.sum(sv[:2]**2)/np.sum(sv**2))
    write(OUT/'calibration.json',dict(n=403,target_hash_overlap=0,top2_variance_ratio=explained,
          records=[{k:r[k] for k in ['image_id','source_path','sha256']} for r in external]))
    del cal,vt,cal_mean
    MEMORY.mkdir(exist_ok=True); np.save(MEMORY/'basis.npy',basis)
    # Separate chronological stream. Never reads evaluation_only/full_latents.npy.
    masters={}; coords={}; next_index=1; total=np.zeros(4096); audits=[]
    partitions=[json.loads(line) for line in (DOMAIN/'insertion_audit.jsonl').read_text().splitlines()]
    assert len(partitions)==len(brand)
    for step,(row,saved) in enumerate(zip(brand,partitions),1):
        assert row['image_id']==saved['image_id']
        x,_=prepare(row)
        vector=model.encode(x[None].to(DEVICE)).latent_dist.mode()[0].cpu().double().numpy().reshape(-1)
        total+=vector
        masters,audit,next_index=update_visual_partition(masters,coords,saved['clusters_partition'],{row['image_id']:vector},basis,next_index)
        coords[row['image_id']]=basis.T@vector
        assert {k:tuple(v['member_ids']) for k,v in masters.items()} == {v['master_id']:tuple(v['member_ids']) for v in saved['clusters']}
        assert np.max(np.abs(sum(m['visual_sum'] for m in masters.values())-total))<1e-9
        assert audit['max_projection_consistency_l2_error']<1e-9
        audits.append(dict(step=step,**audit)); del vector,x
    graph=read(DOMAIN/'memory/graph.json')
    order=[m['id'] for m in graph['masters']]
    assert all(masters[m['id']]['member_ids']==m['member_ids'] for m in graph['masters'])
    sums=np.stack([masters[k]['visual_sum'] for k in order])
    coordinate_array=np.stack([coords[r['image_id']] for r in brand])
    np.save(MEMORY/'master_sums.npy',sums); np.save(MEMORY/'leaf_coords.npy',coordinate_array)
    write(MEMORY/'membership.json',dict(master_ids=order,leaf_ids=[r['image_id'] for r in brand],
          masters=[{k:v for k,v in masters[mid].items() if k!='visual_sum'} for mid in order]))
    with (OUT/'replay_audit.jsonl').open('w') as f:
        for a in audits:f.write(json.dumps(a)+'\n')
    # Reconstruct exclusively from the saved state; full latents are scored later.
    basis=np.load(MEMORY/'basis.npy'); sums=np.load(MEMORY/'master_sums.npy'); zs=np.load(MEMORY/'leaf_coords.npy')
    rebuilt=np.empty((30,4096)); idx={r['image_id']:i for i,r in enumerate(brand)}
    for j,mid in enumerate(order):
        ids=[idx[s] for s in masters[mid]['member_ids']]; mean=sums[j]/len(ids)
        rebuilt[ids]=mean+(zs[ids]-mean@basis)@basis.T
        assert np.max(np.abs(rebuilt[ids].sum(0)-sums[j]))<1e-8
    np.save(EVAL/'reconstructed_latents.npy',rebuilt)
    (OUT/'vamo').mkdir(exist_ok=True)
    for i in range(30):
        out=model.decode(torch.tensor(rebuilt[i].reshape(1,4,32,32),dtype=torch.float32,device=DEVICE)).sample[0]
        output_tensor(out,OUT/'vamo'/f'{i+1:02d}.png')
    print('VAMO decoding complete; evaluating',flush=True)
    import lpips
    metric=lpips.LPIPS(net='alex',version='0.1',verbose=False).to(DEVICE).eval()
    references=np.load(EVAL/'full_latents.npy').reshape(30,-1)
    metric_rows=[]
    for i,meta in enumerate(metadata):
        # Metrics use float decoder outputs, rather than saved 8-bit PNGs.
        orig=prepare(brand[i])[0][None].to(DEVICE)
        direct_out=model.decode(torch.tensor(references[i].reshape(1,4,32,32),device=DEVICE,dtype=torch.float32)).sample.clamp(-1,1)
        vamo_out=model.decode(torch.tensor(rebuilt[i].reshape(1,4,32,32),device=DEVICE,dtype=torch.float32)).sample.clamp(-1,1)
        v,ref=rebuilt[i],references[i].astype('float64')
        cosine=float(v@ref/(np.linalg.norm(v)*np.linalg.norm(ref)))
        relative_l2=float(np.linalg.norm(v-ref)/np.linalg.norm(ref))
        for name,pred,target in [('vae_vs_original',direct_out,orig),('vamo_vs_original',vamo_out,orig),('vamo_vs_vae',vamo_out,direct_out)]:
            p=(pred[0].permute(1,2,0).cpu().numpy()+1)/2
            t=(target[0].permute(1,2,0).cpu().numpy()+1)/2
            x0,y0,x1,y1=meta['content_box']
            kw=dict(data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
            row=dict(index=i+1,image_id=meta['image_id'],filename=meta['filename'],master_label=meta['master_label'],comparison=name,
                     psnr=float(peak_signal_noise_ratio(t,p,data_range=1)),ssim=float(structural_similarity(t,p,**kw)),
                     lpips=float(metric(pred,target).item()),
                     content_psnr=float(peak_signal_noise_ratio(t[y0:y1,x0:x1],p[y0:y1,x0:x1],data_range=1)),
                     content_ssim=float(structural_similarity(t[y0:y1,x0:x1],p[y0:y1,x0:x1],**kw)),
                     latent_cosine=cosine,latent_relative_l2=relative_l2)
            metric_rows.append(row)
    csv_write(OUT/'per_image_metrics.csv',metric_rows)
    keys=['psnr','ssim','lpips','content_psnr','content_ssim']
    summaries=[]
    for group in ['ALL']+[f'M{k+1}' for k in range(8)]:
        for comp in ['vae_vs_original','vamo_vs_original','vamo_vs_vae']:
            subset=[r for r in metric_rows if r['comparison']==comp and (group=='ALL' or r['master_label']==group)]
            # Exact equality produces infinite PSNR; serialize it as the string 'Infinity'.
            summaries.append(dict(group=group,comparison=comp,n=len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in keys}))
    csv_write(OUT/'summary_metrics.csv',summaries)
    selected=read(OUT/'preview_selection.json')
    montage(selected,metadata,'Brand: fixed first member of each master',True).save(OUT/'comparison_preview.png')
    pages=[]
    for j in range(0,30,6):
        page=montage(list(range(j,min(j+6,30))),metadata,f'Brand: all memories ({j+1}-{min(j+6,30)})',True)
        page.save(OUT/f'comparison_page_{j//6+1}.png'); pages.append(page)
    pages[0].save(OUT/'all_30_comparisons.pdf',save_all=True,append_images=pages[1:],resolution=110)
    for p,h in tracked.items(): assert sha(p)==h, p
    elements=sums.size+basis.size+zs.size
    numeric_bytes=sum(p.stat().st_size for p in MEMORY.glob('*.npy'))
    statistics=dict(n_images=30,n_masters=8,external_calibration_images=403,calibration_top2_variance=explained,
                    latent_dim=4096,leaf_coordinates=2,latent_cosine_mean=float(np.mean([r['latent_cosine'] for r in metric_rows[::3]])),
                    latent_relative_l2_mean=float(np.mean([r['latent_relative_l2'] for r in metric_rows[::3]])),
                    actual_numeric_state_dtype='float64',numeric_state_elements=elements,numeric_state_npy_bytes=numeric_bytes,
                    full_30_latents_elements=30*4096,equal_dtype_element_ratio=elements/(30*4096),
                    current_state_float64_vs_full_latents_float32_payload_ratio=elements*8/(30*4096*4),
                    metadata_bytes=(MEMORY/'membership.json').stat().st_size,
                    decoder_checkpoint_bytes=(MODEL/'diffusion_pytorch_model.safetensors').stat().st_size,
                    source_files_unchanged=True,completed=True,
                    exact_singleton_reconstruction_max_abs=float(np.max(np.abs(rebuilt[idx[graph['masters'][-1]['member_ids'][0]]]-references[idx[graph['masters'][-1]['member_ids'][0]]]))))
    write(OUT/'results.json',statistics)
    print(json.dumps(statistics,indent=2),flush=True)
    print('FULL_COMPLETE',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['direct','full'],required=True);args=parser.parse_args()
    brand,external,tracked=sources();model=model_load()
    protocol=identity_protocol(tracked)
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==protocol
    else: write(OUT/'protocol.json',protocol)
    if args.stage=='direct': direct(model,brand)
    else:
        assert (EVAL/'full_latents.npy').exists(), 'Run direct stage first'
        full(model,brand,external,tracked)

if __name__=='__main__':main()
