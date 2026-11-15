"""Export explicitly exploratory examples, complete shortlisted clusters, notes."""
from pathlib import Path
import json,csv
from PIL import Image,ImageDraw,ImageFont

P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def font(n,bold=False):return ImageFont.truetype('/storage/hewenwen/.local/share/fonts/TimesNewRoman/'+('Timesbd.TTF' if bold else 'Times.TTF'),n)
rank=list(csv.DictReader((P/'eligible_cluster_ranking.csv').open(encoding='utf-8-sig')))
selected=[('Personal_Health_Dashboard_Assistant','M01'),('Personal_Health_Dashboard_Assistant','M02'),('Multi-Scene_Visual_Case_Archive_Assistant','M06')]
examples=[];pages=[];records=[]
for domain,label in selected:
    meta=read(P/domain/'image_metadata.json')
    members=[m for m in meta if m['master_label']==label]
    r=next(r for r in rank if r['domain']==domain and r['master_label']==label)
    records.append(r)
    examples.append((domain,members[0],f"{r['short_domain']} {label}: n={r['n']}, unique images={r['unique_image_hashes']}"))
    for page in sorted((P/domain/'clusters').glob(f'{label}_page*.png')):
        with Image.open(page) as im:pages.append(im.convert('RGB').copy())
    if label=='M06':
        examples.append((domain,members[2],'Multi-Scene M06: counterexample, input has no mug'))
canvas=Image.new('RGB',(832,82+300*len(examples)+12),'white');draw=ImageDraw.Draw(canvas)
draw.text((20,8),'Exploratory examples: layout recovery and remaining errors',fill='#222222',font=font(21,True))
for k,label in enumerate(['Input (256 x 256)','VAE direct','VAMO + VAE decode']):
    draw.text((20+268*k,45),label,fill='#333333',font=font(20,True))
for j,(domain,meta,label) in enumerate(examples):
    y=82+j*300;draw.text((20,y),label,fill='#333333',font=font(17,True))
    for k,folder in enumerate(['original','vae_direct','vamo']):
        with Image.open(P/domain/folder/f"{meta['index']:03d}.png") as im:canvas.paste(im.convert('RGB'),(20+268*k,y+32))
canvas.save(P/'selected_examples.png')
pages[0].save(P/'selected_clusters_all_members.pdf',save_all=True,append_images=pages[1:],resolution=110)
(P/'selected_clusters.json').write_text(json.dumps(records,indent=2)+'\n')
notes={
 'Personal_Health_Dashboard_Assistant/M01':'42 distinct images. Dashboard layout, dark panels and circular indicators remain recognizable. Text, exact numeric readings and individual line-curve shapes are not faithfully preserved.',
 'Personal_Health_Dashboard_Assistant/M02':'6 memories but 4 distinct image hashes. Shared message-card layout is clear. Message content/dates blur; direct VAE already harms small text.',
 'Multi-Scene_Visual_Case_Archive_Assistant/M06':'3 distinct images. Sink, tiled wall, fruit bowl and mug are recognizable in reconstructions. The third input S5-IMG4.png has no mug, but reconstruction adds one: shared appearance is retained more faithfully than individual facts.',
 'Multi-Scene_Visual_Case_Archive_Assistant/M10':'3 distinct images. Desktop arrangement remains identifiable but object boundaries/positions and compass details blur.',
 'Card_Playlog_Test/M01':'26 distinct images. Shared tabletop and card-back layout recover reasonably; changing face-up cards collapse to blurred average regions. Not faithful game-state reconstruction.',
 'Card_Playlog_Test/M02':'4 memories correspond to only 1 unique image hash. Do not treat the low LPIPS as a successful multi-distinct-image cluster.',
 'Social_Chat_Memory_Test/M05':'4 distinct images. White chat UI yields favorable global similarity, but text and face details become blurred. Not a clear-content reconstruction.',
 'Social_Chat_Memory_Test/M09':'4 distinct images. Chat layout remains but portraits and message content blur; high blank-area similarity does not imply readability.',
 'Social_Chat_Memory_Test/M11':'3 distinct images. Similar blank UI but poor text and face reconstruction.',
 'Home_Renovation_Interior_Design/M11':'4 distinct images. Mostly light colors and blurry material boards; high SSIM does not correspond to clear texture recovery.',
 'Home_Renovation_Interior_Design/M05':'4 distinct images. Table silhouette can be seen, but different table types collapse to a shared blended form.'}
(P/'visual_review.json').write_text(json.dumps(notes,indent=2)+'\n')
lines=['Seven-domain VAE/VAMO rank-2 reconstruction screening — completed',
       'Scope: 408 memories, 80 final masters, 46 clusters with n>=3, 1224 metric rows.',
       'Same pretrained VAE, 256x256 letterbox preprocessing, frozen chronological caption partitions and rank-2 memory as Brand.',
       'Each domain has its own externally calibrated basis, excluding every target-image hash.',
       'This is a VAE-latent adaptation, not inversion of existing CLIP embeddings.',
       'Selection is exploratory after scoring, NOT an unbiased representative sample.',
       'Primary ranking: mean LPIPS on the content box, with padding excluded. Lower is better.',
       'SSIM is also content-only; simple/blank interfaces can still score well despite unreadable text.',
       '', 'Highlighted clusters:']
for r in records:
    lines.append(f"{r['short_domain']} {r['master_label']} ({r['master_id']}): n={r['n']}, unique={r['unique_image_hashes']}, LPIPS={float(r['content_lpips_mean']):.4f}, SSIM={float(r['content_ssim_mean']):.4f}, worst LPIPS={float(r['content_lpips_worst']):.4f}")
lines+=['','Visual review:']
lines += [f'{key}: {value}' for key,value in notes.items()]
lines += ['', 'Conclusion: some shared layouts are visibly clearer than Brand. The scan does not establish faithful recovery of all individual facts within a multi-image cluster.',
          'selected_examples.png shows one first-arrival example from each highlighted cluster plus an explicit no-mug counterexample.',
          'selected_clusters_all_members.pdf includes ALL 51 memories in the three highlighted clusters (42+6+3), across 9 pages.',
          'Every one of the 46 eligible clusters also has a complete PDF under its domain/clusters directory.',
          'eligible_cluster_ranking.csv contains all eligible clusters, including unique-image counts and worst-member metrics.',
          'validation.json verifies coverage, frozen-source hashes, calibration separation and saved-state reconstruction.',
          'Full model references/calibration vectors are evaluation-only; the replay helper receives only current-arrival vectors plus compact historical state.']
(P/'screening_report.txt').write_text('\n'.join(lines)+'\n')
assert len(pages)==9
print('Exported 4-row preview, all 51 shortlisted memories in 9 PDF pages, and visual-review notes.')
