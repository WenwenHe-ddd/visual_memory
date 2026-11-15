"""Independent saved-artifact audit and readable report; no model calls."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
protocol=read(P/'protocol.json'); results=read(P/'results.json')
meta=read(P/'image_metadata.json'); state=read(P/'memory/membership.json')
cal=read(P/'calibration.json'); metrics=rows(P/'per_image_metrics.csv')
assert len(meta)==30 and len(metrics)==90 and len(cal['records'])==403
assert len({m['image_id'] for m in meta})==30
for path,h in protocol['source_sha256'].items():assert sha(path)==h,path
root=P.parents[1]
for file,h in protocol['model_sha256'].items():assert sha(root/'models/sd-vae-ft-mse'/file)==h
originals=read(root/'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Brand_Memory_Test/input_manifest.json')['arrivals']
assert not {r['sha256'] for r in originals}.intersection(r['sha256'] for r in cal['records'])
u=np.load(P/'memory/basis.npy');sums=np.load(P/'memory/master_sums.npy');z=np.load(P/'memory/leaf_coords.npy')
ref=np.load(P/'evaluation_only/full_latents.npy').reshape(30,4096).astype('float64')
rebuilt=np.load(P/'evaluation_only/reconstructed_latents.npy')
assert u.shape==(4096,2) and sums.shape==(8,4096) and z.shape==(30,2)
assert max(np.abs(u.T@u-np.eye(2)).ravel())<1e-10
projection_error=float(np.max(np.abs(ref@u-z)))
assert projection_error<1e-9
lookup={image:i for i,image in enumerate(state['leaf_ids'])}
mean_errors=[]; reconstruction_errors=[]; singleton=[]
for j,m in enumerate(state['masters']):
 ids=[lookup[i] for i in m['member_ids']]
 mean=sums[j]/len(ids)
 expected=mean+(z[ids]-mean@u)@u.T
 reconstruction_errors.append(float(np.max(np.abs(expected-rebuilt[ids]))))
 mean_errors.append(float(np.max(np.abs(rebuilt[ids].sum(0)-sums[j]))))
 if len(ids)==1:singleton+=ids
assert max(reconstruction_errors)<1e-10 and max(mean_errors)<1e-8
assert singleton==[21]
assert np.max(np.abs(rebuilt[singleton]-ref[singleton]))==0
assert sha(P/'vamo/22.png')==sha(P/'vae_direct/22.png')
assert len({(r['image_id'],r['comparison']) for r in metrics})==90
for r in metrics:
 for k in ['ssim','lpips','content_ssim','latent_cosine','latent_relative_l2']:assert math.isfinite(float(r[k]))
 for k in ['psnr','content_psnr']:
  if not math.isfinite(float(r[k])):assert int(r['index'])==22 and r['comparison']=='vamo_vs_vae'
for folder in ['original','vae_direct','vamo']:
 assert len(list((P/folder).glob('*.png')))==30
selected=read(P/'preview_selection.json')
assert len(selected)==8
for k,i in enumerate(selected):assert i==next(j for j,r in enumerate(meta) if r['master_label']==f'M{k+1}')
replay=[json.loads(line) for line in (P/'replay_audit.jsonl').read_text().splitlines()]
assert len(replay)==30
assert [r['step'] for r in replay]==list(range(1,31))
assert max(r['max_projection_consistency_l2_error'] for r in replay)<1e-9
summary=rows(P/'summary_metrics.csv')
allrows={r['comparison']:r for r in summary if r['group']=='ALL'}
audit=dict(passed=True,images=30,masters=8,metric_rows=90,preview_fixed_before_results=True,
 source_and_model_hashes_verified=True,external_calibration_hash_overlap=0,
 reference_projection_max_abs=projection_error,reconstruction_formula_max_abs=max(reconstruction_errors),
 reconstructed_master_sum_max_abs=max(mean_errors),singleton_control_image='22 / Pepsi_4.png',
 singleton_full_latent_error=0,singleton_decoded_png_identical=True,
 note_infinite_psnr='VAMO vs VAE includes one exact match, so mean per-image PSNR is Infinity; this is not used as a headline score.',
 code_sha256={str(p):sha(p) for p in [P/'run_experiment.py',Path(__file__),root/'scripts/graph/caption_ward_visual.py']})
(P/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
v=allrows['vae_vs_original'];c=allrows['vamo_vs_original']
report=f'''Brand VAE/VAMO reconstruction experiment — completed

Scope: 30 Brand images, 8 frozen caption masters, 30 chronological partition updates.
This is a VAE-latent adaptation of VAMO, not inversion of the existing CLIP vectors.
Model: stabilityai/sd-vae-ft-mse, revision {protocol['model_source']['revision']}.
Input: aspect-preserving 256x256 with gray letterbox. Posterior mean, no sampling.
Raw VAE latents: 4x32x32 = 4096 values; no per-image L2 normalization.
Calibration: 403 unique external images, zero SHA overlap with Brand; rank 2.
Top-two centered calibration variance: {results['calibration_top2_variance']:.6%}.
Each leaf retains two coordinates; masters retain sums. Partial redistributions
use the existing VAMO residual-sharing implementation. No decoder fine-tuning.

Macro means across all 30 images, compared with the preprocessed input:
                         PSNR (dB)    SSIM      LPIPS (lower is better)
VAE direct               {float(v['psnr']):.4f}      {float(v['ssim']):.6f}  {float(v['lpips']):.6f}
VAMO + VAE decode        {float(c['psnr']):.4f}      {float(c['ssim']):.6f}  {float(c['lpips']):.6f}

Padding-excluded PSNR / SSIM:
VAE direct: {float(v['content_psnr']):.4f} dB / {float(v['content_ssim']):.6f}
VAMO:      {float(c['content_psnr']):.4f} dB / {float(c['content_ssim']):.6f}
SSIM uses Gaussian weights, sigma=1.5, data_range=1, no sample covariance.
LPIPS uses the pretrained AlexNet v0.1 metric on complete frames in [-1,1].
Metrics use clamped floating-point decoder outputs before PNG quantization.
VAMO-vs-VAE per-image PSNR includes an exact singleton match and therefore has
an infinite arithmetic mean. See individual rows; do not interpret this as good
overall VAMO reconstruction.

Latent cosine mean: {results['latent_cosine_mean']:.6f}.
Latent relative L2 mean: {results['latent_relative_l2_mean']:.6f}.
Interpretation from the fixed preview: most non-singleton outputs retain coarse
color/layout but lose objects and text; VAE alone already distorts small text.
The singleton M8 reconstructs the exact latent and decodes identically to VAE
direct. The zero-error control and saved-state audits pass.
These observations concern rank-2 reconstruction of spatial VAE latents, and
do not establish failure of the original CLIP retrieval pipeline.

Storage audit (visual state only):
4096x2 basis + 8x4096 master sums + 30x2 coordinates = 41020 scalar values.
Relative to 30 full 4096-value latents at matched precision: 33.3822%.
Current state is float64, while VAE outputs are float32: actual payload ratio
is 66.7643%, before file headers/metadata. Do not claim 3x current-byte savings.
Numeric .npy files: {results['numeric_state_npy_bytes']} bytes.
Membership metadata: {results['metadata_bytes']} bytes.
Shared autoencoder checkpoint: {results['decoder_checkpoint_bytes']} bytes (separate).
Evaluation-only full/reconstructed latents and output images are excluded from
algorithm state; they are kept only to audit the experiment.

Files:
comparison_preview.png: first arrival in each master, selected before scoring.
all_30_comparisons.pdf: all 30 original/direct/VAMO triplets, chronological order.
per_image_metrics.csv / summary_metrics.csv: detailed and per-master metrics.
memory/: the saved compact state, with no full leaf-latent matrix.
evaluation_only/: reference and reconstructed matrices, never used in updates.
protocol.json / replay_audit.jsonl / validation.json: provenance and audit.

Reproduce in the current host:
PYTHONPATH=/tmp/vamo-vae-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 run_experiment.py --stage direct
PYTHONPATH=/tmp/vamo-vae-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 run_experiment.py --stage full
python3 validate_and_report.py
GPU access is required. Download the model at the pinned revision first.
Exact environment versions are recorded in protocol.json.
'''
(P/'experiment_report.txt').write_text(report)
for p in [P/'protocol.json',P/'evaluation_only/full_latents.npy',P/'evaluation_only/reconstructed_latents.npy',P/'per_image_metrics.csv',P/'summary_metrics.csv',P/'comparison_preview.png',P/'all_30_comparisons.pdf']:
 assert p.exists() and p.stat().st_size>0
print(json.dumps(audit,indent=2))
