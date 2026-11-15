"""Audit paired caption runs and render comparison and natural-observation figures."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'results/.matplotlib'))
from plot_mem_gallery_injection_groups import SCENARIOS, TYPES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def jl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='qwen3_vl_8b_20261018_v1')
    args = parser.parse_args()
    old_nat = ROOT/'results/mem_gallery_natural_observation/v1'
    old_inj = ROOT/'results/mem_gallery_distractor_injection/v1'
    new_nat = old_nat.parent/args.version
    new_inj = old_inj.parent/args.version
    out = ROOT/'results/mem_gallery_caption_comparison'/args.version
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for folder in [old_nat, old_inj, new_nat, new_inj]:
        assert read(folder/'independent_audit.json')['status'] == 'passed'
        for name, digest in read(folder/'artifact_manifest.json').items():
            assert sha(folder/name) == digest, folder/name
        for name in ['summary.json', 'protocol.json', 'artifact_manifest.json']:
            inputs[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    old_p, new_p = [read(p/'protocol.json') for p in [old_nat, new_nat]]
    for key in ['retriever', 'retriever_revision', 'model_file_hashes', 'query_prefix', 'seed', 'device',
                'scope', 'memory_unit', 'index_content', 'ties', 'competitors']:
        assert old_p[key] == new_p[key], key
    for name in ['gold.jsonl']:
        assert (old_nat/name).read_bytes() == (new_nat/name).read_bytes()
    for name in ['sampling_pools.jsonl', 'sampling_plans.jsonl']:
        assert (old_inj/name).read_bytes() == (new_inj/name).read_bytes(), name
    old_q, new_q = [jl(p/'queries.jsonl') for p in [old_nat, new_nat]]
    old_m, new_m = [jl(p/'memories.jsonl') for p in [old_nat, new_nat]]
    assert len(old_m) == len(new_m) == 1003 and len(old_q) == len(new_q) == 1711
    for a, b in zip(old_m, new_m):
        assert {k:v for k,v in a.items() if k not in ['caption','source_captions']} == {k:v for k,v in b.items() if k not in ['caption','source_captions']}
    for a, b in zip(old_q, new_q):
        assert {k:v for k,v in a.items() if k not in ['provided_caption','retrieval_text']} == {k:v for k,v in b.items() if k not in ['provided_caption','retrieval_text']}
        if not a['question_image']:
            assert a['retrieval_text'] == b['retrieval_text']
    nat = [read(p/'summary.json') for p in [old_nat,new_nat]]
    inj = [read(p/'summary.json') for p in [old_inj,new_inj]]
    labels = ['Official captions', 'Qwen captions']
    comparison = []
    for label,n,i in zip(labels,nat,inj):
        row = {'caption_source':label, **n['primary']}
        for condition in ['same_scenario','other_scenarios']:
            value = next(r for r in i['overall'] if r['N']==20 and r['condition']==condition)
            row[condition+'_N20_recall'] = value['recall_at_5']
            row[condition+'_N20_exact_recall'] = value['exact_expected_recall_at_5']
            row[condition+'_N20_similarity'] = value['target_distractor_cosine']
        row['N20_same_minus_other'] = next(r['target_equal_gap'] for r in i['paired'] if r['N']==20)
        comparison.append(row)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    checks = {'status':'passed','identical_primary_questions':456,'identical_primary_targets':1688,
              'identical_sampling_plans':len(jl(new_inj/'sampling_plans.jsonl')),
              'identical_candidate_pools':True,'same_retriever_weights_settings_and_source_order':True,
              'changed_history_captions':sum(a['caption']!=b['caption'] for a,b in zip(old_m,new_m)),
              'changed_question_captions':sum(a['provided_caption']!=b['provided_caption'] for a,b in zip(old_q,new_q)),
              'text_only_queries_unchanged':True,'all_four_run_artifact_manifests_verified':True}
    assert all(r['questions']==456 and r['targets']==1688 for r in comparison)
    dump(out/'paired_audit.json', checks)
    dump(out/'comparison.json', comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    colors=['#61758a','#b96537']
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for x,r in enumerate(comparison):
        value=r['target_recall_at_5']
        axes[0].bar(x,value,color=colors[x]);axes[0].text(x,value+.015,f'{value:.2%}',ha='center')
        value=r['missed_minus_retrieved']
        axes[1].bar(x,value,color=colors[x]);axes[1].annotate(f'{value:+.4f}',(x,value),xytext=(0,-14 if value<0 else 8),textcoords='offset points',ha='center')
    axes[0].set_ylim(0,1);axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].set_title('Full-history target Recall@5')
    axes[1].set_ylim(min(r['missed_minus_retrieved'] for r in comparison)-.025,max(0,max(r['missed_minus_retrieved'] for r in comparison))+.02)
    axes[1].axhline(0,color='#777',linewidth=.8);axes[1].set_title('Natural observation: missed minus retrieved');axes[1].set_ylabel('Target-to-competitor caption cosine gap')
    for ax in axes[:2]:ax.set_xticks([0,1],labels)
    for x,r in enumerate(comparison):
        for offset,condition,color in [(-.18,'same_scenario','#b96537'),(.18,'other_scenarios','#287d9b')]:
            value=r[condition+'_N20_recall']
            axes[2].bar(x+offset,value,width=.34,color=color,label=condition.replace('_',' ') if x==0 else None)
            axes[2].text(x+offset,value+.01,f'{value:.2%}',ha='center',fontsize=9)
    axes[2].set_xticks([0,1],labels);axes[2].set_ylim(0,1.15);axes[2].set_yticks([0,.2,.4,.6,.8,1]);axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_title('One target + 20 distractors: Recall@5');axes[2].legend(loc='lower left',fontsize=9)
    fig.suptitle('Mem-Gallery: same 456 questions / 1,688 targets and identical sampled images')
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/('comparison.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Keep one figure per scenario and question type for the new natural observation.
    figures = new_nat/'individual_figures'
    figures.mkdir(exist_ok=False)
    index=[]
    for category,key,names in [('scenario','by_scenario',SCENARIOS),('question_type','by_question_type',TYPES)]:
        folder=figures/category;folder.mkdir()
        for r in nat[1][key]:
            if not r['targets']:continue
            english,chinese=names[r['group']]
            fig,ax=plt.subplots(figsize=(7,5))
            for x,field,color in [(0,'retrieved','#287d9b'),(1,'missed','#b96537')]:
                value=r[field+'_mean'];count=r[field+'_n']
                if value is None:
                    ax.text(x,.05,'NA (n=0)',ha='center')
                else:
                    ax.bar(x,value,color=color,width=.55);ax.text(x,value+.02,f'{value:.4f}\nn={count}',ha='center')
            ax.set_xticks([0,1],['Retrieved','Missed']);ax.set_ylim(0,1.08)
            ax.set_ylabel('Mean target-to-competitor caption cosine')
            ax.set_title(f'{english}\nQwen captions | {r["questions"]} questions / {r["targets"]} targets',fontsize=11)
            gap='NA' if r['missed_minus_retrieved'] is None else f'{r["missed_minus_retrieved"]:+.4f}'
            fig.text(.5,.025,f'Recall@5: {r["target_recall_at_5"]:.2%} | Missed minus retrieved: {gap}',ha='center')
            fig.tight_layout(rect=(0,.06,1,1))
            for ext in ['png','svg','pdf']:fig.savefig(folder/(r['group']+'.'+ext),dpi=170,bbox_inches='tight')
            plt.close(fig)
            index.append({'category':category,'group':r['group'],'label_zh':chinese,'path':category+'/'+r['group']+'.png'})
    dump(figures/'index.json',index)
    cards=''.join(f'<details><summary>{html.escape(r["label_zh"])}</summary><img loading="lazy" src="{r["path"]}"></details>' for r in index)
    (figures/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen caption 自然观测分组图</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto}img{max-width:100%}details{padding:12px}</style><h1>Qwen caption 自然观测：每组一张图</h1><p>成功/失败目标与前五个非目标的 caption 相似度。NA 表示该组没有成功或失败样本。描述性关联，不代表因果效应。</p>'+cards)
    dump(figures/'manifest.json',{'source_sha256':sha(new_nat/'summary.json'),'script_sha256':sha(Path(__file__)),
        'artifacts':{str(p.relative_to(figures)):sha(p) for p in sorted(figures.rglob('*')) if p.is_file()}})
    rows=''.join(f'<tr><td>{r["caption_source"]}</td><td>{r["target_recall_at_5"]:.2%}</td><td>{r["retrieved_mean"]:.4f}</td><td>{r["missed_mean"]:.4f}</td><td>{r["missed_minus_retrieved"]:+.4f}</td><td>{r["same_scenario_N20_recall"]:.2%}</td><td>{r["other_scenarios_N20_recall"]:.2%}</td><td>{100*r["N20_same_minus_other"]:+.2f} pp</td></tr>' for r in comparison)
    links=''.join(f'<li><a href="{os.path.relpath(p,out)}">{label}</a></li>' for label,p in [
        ('Qwen 自然观测报告',new_nat/'report.html'),('Qwen 自然观测：每场景／题型一张图',figures/'index.html'),
        ('Qwen 干扰实验报告',new_inj/'report.html'),('Qwen 干扰实验：每场景／题型一张图',new_inj/'individual_figures/index.html')])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Mem-Gallery caption 对照</title>
<style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:30px auto;padding:20px}}img{{max-width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccc}}table{{border-collapse:collapse}}</style>
<h1>Mem-Gallery：官方 caption 与 Qwen caption 对照</h1>
<p>固定 456 题、1,688 个目标；同一 BGE 检索模型、Top-5 和完整同场景历史。历史图片与问题图片的 caption 同时替换为 Qwen3-VL-8B-Instruct 缓存。</p>
<p>干扰实验保留单个目标，加入 0、10、20 个干扰；每目标、每条件抽样 5 次，seed=42。已逐项确认两轮候选池与全部 16,880 条抽样计划完全相同。原始结果和官方 caption 均保留。</p>
<img src="comparison.png" alt="两种 caption 来源下的自然观测与干扰实验对照">
<table><tr><th>Caption</th><th>完整历史 Recall@5</th><th>成功组相似度</th><th>失败组相似度</th><th>失败减成功</th><th>N20 同场景 Recall@5</th><th>N20 其他场景 Recall@5</th><th>同场景减其他</th></tr>{rows}</table>
<p>自然观测是描述性关联，竞争记忆由查询排名选取。干扰实验操纵数量与来源，同场景不等于已确认的高线索重叠。
完整历史与单目标小记忆库的难度不同，不直接比较它们的召回率。结果衡量目标检索，不是回答准确率；没有因果或显著性检验。</p>
<p>本轮同时替换历史和问题 caption，不能区分两者各自贡献。Qwen 对完全相同图片复用一个 caption；官方 caption 在重复图片间可能不同，因此对照也包含描述一致性的变化。</p>
<ul>{links}</ul><p><a href="comparison.csv">对照数据 CSV</a> · <a href="paired_audit.json">配对条件审计</a></p></html>'''
    (out/'report.html').write_text(page)
    dump(out/'manifest.json',{'inputs':inputs,'script_sha256':sha(Path(__file__)),
        'artifacts':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({'comparison':comparison,'audit':checks,'report':str(out/'report.html')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
