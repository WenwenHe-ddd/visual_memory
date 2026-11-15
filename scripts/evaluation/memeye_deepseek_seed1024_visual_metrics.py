"""History-only bytes and source-round recall for fresh DeepSeek visual methods."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation import memeye_seed1024_visual_metrics as historical
from scripts.evaluation import memeye_deepseek_seed1024_visual as driver

DEFAULT=driver.DEFAULT


def measure_hippo(root):
    memory=root/'hippomm/offline'
    online=root/'hippomm/online'
    m,s=(json.loads(p.read_text()) for p in [memory/'memory.json',online/'summary.json'])
    completion=json.loads((online/'completion_manifest.json').read_text())
    if completion['status']!='complete' or s['questions']!=371 or m['model']!=driver.api.MODEL or m['seed']!=1024:
        raise ValueError('Completed native DeepSeek seed1024 memory and 371 MCQ responses required')
    qs=driver.base.inputs.unique(driver.read(online/'questions.jsonl'))
    rs=driver.base.inputs.unique(driver.read(online/'qa_responses.jsonl'))
    gold=driver.base.inputs.unique(driver.read(driver.BENCHMARK))
    if set(qs)!=set(rs) or set(qs)!=set(gold) or len(rs)!=371:
        raise ValueError('Incomplete MCQ ID coverage')
    names=(*driver.hip.original.MEMORY_FILES,'representative_images.jsonl','caption_provenance.jsonl')
    components=[historical.component(memory/n,'persisted_memory_artifact') for n in names]
    records=driver.read(memory/'representative_images.jsonl')
    for r in records:
        p=(memory/r['file']).resolve()
        if not p.is_relative_to((memory/'images').resolve()) or p.stat().st_size!=r['bytes']:
            raise ValueError('Retained representative path/size changed')
        components.append(historical.component(p,'historical_representative_image'))
    image_bytes=sum(r['bytes'] for r in components if r['category']=='historical_representative_image')
    artifact_bytes=sum(r['bytes'] for r in components if r['category']=='persisted_memory_artifact')
    total=image_bytes+artifact_bytes
    if total!=s['total_memory_bytes'] or image_bytes!=m['original_image_bytes'] or artifact_bytes!=m['memory_artifact_bytes']:
        raise ValueError('Native storage accounting does not reproduce')
    hits=sum(len(set(gold[k]['gold_ids'])&set(r['context_round_ids'])) for k,r in rs.items())
    denominator=sum(len(set(g['gold_ids'])) for g in gold.values())
    correct=sum(r['answer']==gold[k]['gold_choice'] for k,r in rs.items())
    if correct!=s['correct'] or abs(hits/denominator-s['context_source_micro_recall'])>1e-12:
        raise ValueError('Native correctness/context recall does not reproduce')
    result=dict(status='complete',method='hippomm',model=driver.api.MODEL,seed=1024,questions=371,
        correct=correct,accuracy=correct/371,recall_at5_micro=hits/denominator,
        context_source_micro_recall=hits/denominator,recall_numerator=hits,recall_denominator=denominator,
        recall_is_literal_top5=False,
        recall_metric='Context source-round coverage; fast summaries include same-task event provenance; detailed retrieval uses up to five events.',
        storage_bytes=total,storage_mib=total/(1024**2),storage_unit='MiB = bytes / 1048576',
        image_bytes=image_bytes,memory_artifact_bytes=artifact_bytes,
        storage_policy='Actual representative originals plus six native memory artifacts; no queries, answers, logs, or LLM caches.',
        components=components,source_sha256={str(p):driver.sha(p) for p in
            [Path(__file__),memory/'memory.json',online/'summary.json',online/'qa_responses.jsonl',driver.BENCHMARK]})
    driver.dump(root/'hippomm/metric_summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['clip_image_dialogue','hippomm','all'],required=True)
    p.add_argument('--phase',choices=['measure'],required=True)
    p.add_argument('--visual-root',type=Path,default=DEFAULT)
    a=p.parse_args();root=a.visual_root.resolve()
    for method in ['clip_image_dialogue','hippomm'] if a.method=='all' else [a.method]:
        if method=='hippomm':r=measure_hippo(root)
        else:
            r=historical.clip_metrics(SimpleNamespace(visual_root=root,
                clip_mcq_source=root/'clip_image_dialogue/online'))
        print(json.dumps({k:r[k] for k in ['method','questions','correct','storage_bytes','storage_mib']}),flush=True)


if __name__=='__main__':main()
