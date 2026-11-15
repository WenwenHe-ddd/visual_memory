"""Display original sample projections on saved residual-PCA axes for annotation."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "reports/home_q10_top5_embeddings_20261020"
PCA = ROOT / "reports/home_q10_residual_pca_20261020"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visualmemory_q10_pc_extremes_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_out(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if (OUT / "gallery_data.json").exists():
        raise FileExistsError("Gallery already exists; choose a new directory.")
    source_check = json.loads((SOURCE / "validation.json").read_text())
    pca_check = json.loads((PCA / "validation.json").read_text())
    assert sha(SOURCE / "embeddings.npz") == source_check["artifact_sha256"]["embeddings.npz"]
    assert sha(PCA / "residuals_and_pca.npz") == pca_check["output_sha256"]["residuals_and_pca.npz"]
    inputs = json.loads((SOURCE / "inputs.json").read_text())
    with np.load(SOURCE / "embeddings.npz", allow_pickle=False) as z:
        ids = z["memory_ids"].tolist()
        names = z["filenames"].tolist()
        originals = {"image": z["image_embeddings"], "caption": z["text_full_caption"]}
    with np.load(PCA / "residuals_and_pca.npz", allow_pickle=False) as z:
        assert ids == z["memory_ids"].tolist() and names == z["filenames"].tolist()
        axes = {m: z[m + "_pca_components"][:5] for m in originals}
        ratios = {m: z[m + "_pca_explained_variance_ratio"][:5] for m in originals}
    assert ids == [r["memory_id"] for r in inputs]
    assert names == [r["filename"] for r in inputs]
    for r in inputs:
        assert sha(Path(r["image_path"])) == r["image_sha256"]

    data = {"source_filenames": names, "modalities": {}, "requested_components": 5,
            "valid_components": 4, "extreme_count_per_end": 2}
    arrays, projection_rows, template, checks = {}, [], [], {}
    for modality, x in originals.items():
        centered = x - x.mean(axis=0, keepdims=True)
        scores = centered @ axes[modality].T
        assert np.linalg.matrix_rank(centered) == 4
        score_energy_ratio = np.sum(scores[:, :4] ** 2, axis=0) / np.sum(centered ** 2)
        assert np.allclose(score_energy_ratio, ratios[modality][:4], atol=1e-12, rtol=0)
        assert np.max(np.abs(scores[:, 4])) < 1e-12
        # Retain the existing signs. SVD signs do not have intrinsic semantic meaning.
        arrays[f"{modality}_sample_mean"] = x.mean(axis=0)
        arrays[f"{modality}_components"] = axes[modality]
        arrays[f"{modality}_projections"] = scores
        cards = []
        for pc in range(5):
            valid = pc < 4
            samples = []
            for i, r in enumerate(inputs):
                samples.append({
                    "row_index": i, "filename": names[i], "memory_id": ids[i],
                    "image_path": r["image_path"],
                    "image_url": os.path.relpath(r["image_path"], OUT),
                    "caption": r["full_caption"],
                    "score": float(scores[i, pc]) if valid else 0.0,
                })
                projection_rows.append({
                    "modality": modality, "pc": pc + 1, "interpretable": valid,
                    "row_index": i, "filename": names[i], "memory_id": ids[i],
                    "projection": float(scores[i, pc]),
                    "explained_variance_ratio": float(ratios[modality][pc]),
                    "rank_low_to_high": int(np.flatnonzero(np.argsort(scores[:, pc], kind="stable") == i)[0] + 1)
                    if valid else "",
                })
            ordered = sorted(samples, key=lambda r: r["score"]) if valid else samples
            cards.append({
                "pc": pc + 1, "valid": valid,
                "explained_variance_ratio": float(ratios[modality][pc]),
                "all_samples": ordered,
                "low": ordered[:2] if valid else [],
                "high": ordered[-2:] if valid else [],
                "middle": ordered[2:3] if valid else [],
                "status": "supported" if valid else "zero_variance_no_ranking",
            })
            template.append({
                "modality": modality, "pc": pc + 1, "valid": valid,
                "direction_low_to_high": "", "confidence": "unlabeled", "notes": "",
            })
        data["modalities"][modality] = cards
        checks[modality] = {
            "rank": 4,
            "max_evr_error": float(np.max(np.abs(score_energy_ratio - ratios[modality][:4]))),
            "pc5_max_projection_abs": float(np.max(np.abs(scores[:, 4]))),
            "pc5_range": float(np.ptp(scores[:, 4])),
        }
    arrays["memory_ids"] = np.asarray(ids)
    arrays["filenames"] = np.asarray(names)
    arrays["component_valid"] = np.asarray([True, True, True, True, False])
    np.savez_compressed(OUT / "projections.npz", **arrays)
    csv_out("projections.csv", projection_rows)
    save("gallery_data.json", data)
    save("annotations_template.json", {
        "source_embedding_sha256": sha(SOURCE / "embeddings.npz"),
        "source_pca_sha256": sha(PCA / "residuals_and_pca.npz"),
        "annotations": template,
    })

    # Scientific contact sheets preserve each full source image without cropping.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#263449", "svg.fonttype": "none", "pdf.fonttype": 42})
    titles = {"image": "Image PCA", "caption": "Full-caption PCA"}
    ends = ["Minimum", "2nd lowest", "2nd highest", "Maximum"]
    colors = ["#3978BA", "#3978BA", "#CB7434", "#CB7434"]
    image_cache = {}
    for r in inputs:
        with Image.open(r["image_path"]) as im:
            image_cache[r["filename"]] = np.array(im.convert("RGB"))
    for modality in originals:
        fig, panels = plt.subplots(4, 4, figsize=(14, 16))
        fig.subplots_adjust(left=.055, right=.975, top=.885, bottom=.10, hspace=.50, wspace=.075)
        fig.text(.055, .967, f"{titles[modality]} · low / high projection examples",
                 fontsize=22, weight="bold")
        fig.text(.055, .939, "Original samples projected onto the saved residual-PCA directions",
                 fontsize=12, color="#667085")
        fig.text(.055, .913, "Lowest two  ←", color=colors[0], fontsize=13, weight="bold")
        fig.text(.975, .913, "→  Highest two", color=colors[-1], fontsize=13,
                 weight="bold", ha="right")
        for pc, card in enumerate(data["modalities"][modality][:4]):
            examples = card["low"] + card["high"]
            for col, sample in enumerate(examples):
                ax = panels[pc, col]
                ax.imshow(image_cache[sample["filename"]])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(colors[col])
                    spine.set_linewidth(1.8)
                ax.set_xlabel(
                    f"{ends[col]} · {sample['filename']}\nProjection {sample['score']:+.4f}",
                    fontsize=10.5, color=colors[col], labelpad=7,
                )
            bounds = panels[pc, 0].get_position()
            fig.text(.055, bounds.y1 + .010,
                     f"PC{pc + 1}   |   explained variance {card['explained_variance_ratio'] * 100:.2f}%",
                     fontsize=13, weight="bold")
        fig.text(.055, .045, "PC5: zero variance — no meaningful low/high ranking (only five source samples).",
                 fontsize=13, weight="bold", color="#667085")
        fig.text(.055, .020, "Signs are arbitrary; image and caption PCs are fitted independently. No semantic labels are preassigned.",
                 fontsize=10.5, color="#667085")
        for suffix in ["png", "pdf"]:
            fig.savefig(OUT / f"{modality}_pc_extremes.{suffix}", dpi=160, facecolor="white")
        plt.close(fig)

    serialized = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    annotations_json = json.dumps(template, ensure_ascii=False)
    source_hash = sha(SOURCE / "embeddings.npz")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q10 主成分方向标注</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#243247;font:16px/1.6 system-ui,sans-serif}
main{max-width:1480px;margin:auto;padding:28px}h1{font-size:28px;margin-bottom:10px}h2{font-size:21px;margin:0}
.intro,.pc,.toolbar{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:20px;margin:18px 0}
.note{color:#667085;font-size:14px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:2;box-shadow:0 4px 12px #00000008}
button,select{font:inherit;border:1px solid #bcc9da;border-radius:7px;padding:9px 14px;background:white;color:#243247;cursor:pointer}
button.active{background:#243247;color:white}.export{margin-left:auto}.groups{display:grid;grid-template-columns:1fr 1fr;gap:30px;margin:18px 0}
.group{min-width:0}.group h3{font-size:15px;margin:0 0 8px}.low h3{color:#2869aa}.high h3{color:#b85e24}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}.sample{border:1px solid #dce3ec;border-radius:8px;overflow:hidden;min-width:0}
.sample img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#edf1f5}
.sample .meta{padding:10px;overflow-wrap:anywhere;font-size:13px}.score{font-size:16px;font-weight:700}
.caption{font-size:13px;line-height:1.55}.annotation{border-top:1px solid #e2e7ed;padding-top:16px}
textarea{display:block;width:100%;resize:vertical;min-height:75px;font:inherit;border:1px solid #bcc9da;border-radius:6px;padding:10px;margin:7px 0 12px}
.annotation select{font-size:14px}.invalid{background:#eef1f5;color:#667085}.tag{font-size:14px;font-weight:400;background:#eef2f7;padding:4px 9px;border-radius:5px;margin-left:8px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #e2e7ed;text-align:left}
a{color:#245e9e}summary{cursor:pointer}.all{margin:16px 0}.save-status{font-size:13px;color:#667085}
@media(max-width:850px){.groups{grid-template-columns:1fr;gap:20px}.toolbar{position:static}.export{margin-left:0}}
@media(max-width:460px){main{padding:12px}.cards{grid-template-columns:1fr}.tag{display:block;margin:6px 0}}
</style><main>
<h1>看主成分方向在变什么</h1>
<div class="intro">
<p>每个主成分显示投影最低的两张和最高的两张，方向为<strong>左侧低值 → 右侧高值</strong>。你可以填写变化特征并导出标注。</p>
<p class="note">沿用上一轮残差 PCA 方向，把五张图各自的原始表征减去五图均值后投影。保留原方向符号，不再拟合 PCA。
Caption 使用 190 分版本的完整描述，不含对话。两种表征分别拟合，同编号 PC 不代表相同语义方向；正负号本身没有含义。</p>
<p class="note">只有五个样本，PC1–PC4 有效；PC5 为零方差，不能用数值误差给图片排高低。Caption 的 PC2 / PC3 方差接近，标注仅描述当前五图的结果。</p>
<p><a href="image_pc_extremes.png">图片表征总览 PNG</a> · <a href="caption_pc_extremes.png">Caption 表征总览 PNG</a> ·
<a href="image_pc_extremes.pdf">图片总览 PDF</a> · <a href="caption_pc_extremes.pdf">Caption 总览 PDF</a></p>
</div>
<div class="toolbar"><button id="image-tab" class="active" onclick="switchMode('image')">图像表征</button>
<button id="caption-tab" onclick="switchMode('caption')">Caption 表征</button>
<span id="progress"></span><button class="export" onclick="downloadAnnotations()">导出标注 JSON</button>
<span id="save-status" class="save-status">标注会在当前浏览器暂存，请导出保存。</span></div>
<div id="panels"></div>
<p class="note"><a href="projections.csv">全部投影值 CSV</a> · <a href="projections.npz">投影与方向数组</a> ·
<a href="annotations_template.json">空白标注模板</a> · <a href="protocol.json">计算口径</a></p>
</main><script>
const data=__DATA__;
const template=__ANNOTATIONS__;
const storageKey='visualmemory-q10-pc-annotations-__HASH__';
let annotations=structuredClone(template), mode='image';
try{const previous=JSON.parse(localStorage.getItem(storageKey));if(Array.isArray(previous)&&previous.length===10)annotations=previous;}catch(e){}
const el=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fmt(x){return (x>=0?'+':'')+x.toFixed(4);}
function sample(s){return `<article class="sample"><a href="${esc(s.image_url)}" target="_blank"><img loading="lazy" src="${esc(s.image_url)}" alt="${esc(s.filename)}"></a><div class="meta"><b>${esc(s.filename)}</b><br><span class="score">投影 ${fmt(s.score)}</span><br><span class="note">${esc(s.memory_id)}</span><details><summary>展开完整 caption</summary><p class="caption">${esc(s.caption)}</p></details></div></article>`;}
function render(){
 el('image-tab').classList.toggle('active',mode==='image');el('caption-tab').classList.toggle('active',mode==='caption');
 el('panels').innerHTML=data.modalities[mode].map(pc=>{
  if(!pc.valid)return `<section class="pc invalid"><h2>PC5 · 无有效方差</h2><p>五张图的居中表征最多有四个独立变化方向。PC5 投影差异约为 10⁻¹⁶，仅为浮点误差，全部视为并列，因此不展示最低／最高样例。</p></section>`;
  const a=annotations.find(a=>a.modality===mode&&a.pc===pc.pc);
  return `<section class="pc"><h2>PC${pc.pc}<span class="tag">解释方差 ${(pc.explained_variance_ratio*100).toFixed(2)}%</span></h2>
  <div class="groups"><div class="group low"><h3>最低的两张 · 从低到高</h3><div class="cards">${pc.low.map(sample).join('')}</div></div>
  <div class="group high"><h3>最高的两张 · 从低到高</h3><div class="cards">${pc.high.map(sample).join('')}</div></div></div>
  <details class="all"><summary>查看全部五张的投影排序（含中间样本）</summary><table><tr><th>从低到高</th><th>图片</th><th>投影值</th></tr>${pc.all_samples.map((s,i)=>`<tr><td>${i+1}</td><td><a href="${esc(s.image_url)}" target="_blank">${esc(s.filename)}</a></td><td>${fmt(s.score)}</td></tr>`).join('')}</table></details>
  <div class="annotation"><label for="direction-${pc.pc}">从左侧低值到右侧高值，你观察到什么变化？</label>
  <textarea id="direction-${pc.pc}" data-pc="${pc.pc}" data-field="direction_low_to_high">${esc(a.direction_low_to_high)}</textarea>
  <label>把握程度 <select data-pc="${pc.pc}" data-field="confidence">${[['unlabeled','未标注'],['uncertain','不确定'],['tentative','初步判断'],['clear','比较清楚']].map(([v,t])=>`<option value="${v}" ${a.confidence===v?'selected':''}>${t}</option>`).join('')}</select></label>
  <details><summary>补充备注</summary><textarea data-pc="${pc.pc}" data-field="notes">${esc(a.notes)}</textarea></details></div></section>`;
 }).join('');
 document.querySelectorAll('[data-field]').forEach(node=>node.addEventListener('input',()=>{const a=annotations.find(a=>a.modality===mode&&a.pc===Number(node.dataset.pc));a[node.dataset.field]=node.value;persist();}));
 updateProgress();
}
function updateProgress(){el('progress').textContent=`已填写 ${annotations.filter(a=>a.valid&&a.direction_low_to_high.trim()).length} / 8 个方向`;}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(annotations));el('save-status').textContent='已在当前浏览器暂存；请导出保存。';}catch(e){el('save-status').textContent='浏览器无法暂存，请用“导出标注 JSON”保存。';}updateProgress();}
function switchMode(value){mode=value;render();}
function downloadAnnotations(){
 const payload={source_embedding_sha256:'__HASH__',source_pca_sha256:'__PCAHASH__',projection_formula:'(x - mean_of_five_originals) @ saved_residual_PCA_components.T',created_utc:new Date().toISOString(),annotations};
 const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='q10_pc_direction_annotations.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
render();
</script></html>"""
    page = page.replace("__DATA__", serialized).replace("__ANNOTATIONS__", annotations_json)
    page = page.replace("__HASH__", source_hash).replace("__PCAHASH__", sha(PCA / "residuals_and_pca.npz"))
    (OUT / "report.html").write_text(page)
    save("protocol.json", {
        "source_sha256": {str(p): sha(p) for p in [
            SOURCE / "embeddings.npz", SOURCE / "inputs.json", SOURCE / "protocol.json",
            PCA / "residuals_and_pca.npz", PCA / "protocol.json",
        ]},
        "script_sha256": sha(Path(__file__)),
        "projection": "(original_embedding - mean_of_five_originals) @ saved_residual_PCA_components.T",
        "pca_refitted": False, "original_component_signs_preserved": True,
        "selected_image_array": "image_embeddings", "selected_caption_array": "text_full_caption",
        "number_requested_components": 5, "number_interpretable_components": 4,
        "pc5_handling": "Retain raw projection in NPZ/CSV for audit; treat as tied zero in UI and omit extreme ranking.",
        "low_count": 2, "high_count": 2, "display_order": "low1,low2,high2,high1; ascending within each side.",
        "annotation_labels_preassigned": False, "image_transform": "Display full image with preserved aspect ratio; no cropping.",
        "annotation_storage": "Browser localStorage plus explicit JSON download; no server writes.",
        "interpretation": "Image and caption PCs are independent axes; signs and semantic interpretation are not intrinsic.",
    })
    save("validation.json", {
        "status": "passed", "source_alignment": True, "image_hashes_checked": 5,
        "pc_projection_checks": checks, "valid_panels": 8, "disabled_pc5_panels": 2,
        "source_images_per_panel": 4,
        "output_sha256": {name: sha(OUT / name) for name in [
            "projections.npz", "projections.csv", "gallery_data.json", "annotations_template.json",
            "report.html", "protocol.json", "image_pc_extremes.png", "caption_pc_extremes.png",
            "image_pc_extremes.pdf", "caption_pc_extremes.pdf",
        ]},
    })
    print(json.dumps({"output": str(OUT), "checks": checks, "valid_panels": 8}, indent=2))


if __name__ == "__main__":
    main()
