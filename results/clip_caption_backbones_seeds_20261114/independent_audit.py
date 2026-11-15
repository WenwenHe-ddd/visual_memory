"""Read-only independent grid audit and seed aggregation; writes reports only."""
import csv
import gzip
import json
from pathlib import Path
import statistics
import sys
import numpy as np
ROOT=Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0,str(ROOT))
from scripts.memory.clip_caption import rows,read,digest,sha,dump,reader_payload
OUT=Path(__file__).resolve().parent

def main():
    complete=read(OUT/'completion_manifest.json')
    assert complete['status']=='complete' and complete['complete_cells']==18
    for filename,expected in complete['artifact_sha256'].items():assert sha(filename)==expected,filename
    protocol=read(OUT/'suite_protocol.json');checked=[];predictions={};timelines={}
    for cell in protocol['new_cells']:
        target=Path(cell['path']);p=read(target/'protocol.json');audit=read(target/'audit.json')
        assert audit['status']=='passed'
        for filename,expected in audit['artifact_sha256'].items():assert sha(filename)==expected,filename
        source=Path(p['source']);memory=Path(p['memory_path'])
        assert set(f.name for f in memory.iterdir())=={'memory.json','records.jsonl','image_embeddings.npy'}
        records=rows(memory/'records.jsonl');query=rows(source/'questions.jsonl');gold=rows(source/'gold.jsonl')
        assert len(query)==len(gold)==p['questions']
        v=np.load(memory/'image_embeddings.npy').astype('float64');qv=np.load(source/'query_embeddings.npy').astype('float64')
        v/=np.linalg.norm(v,axis=1,keepdims=True);qv/=np.linalg.norm(qv,axis=1,keepdims=True)
        ranked={}
        for q,vector in zip(query,qv):
            indices=[i for i,r in enumerate(records) if r['task']==q['task']]
            ranked[q['id']]=[records[indices[i]] for i in np.argsort(-(v[indices]@vector),kind='stable')[:5]]
        qmap={q['id']:q for q in query};manifest=rows(target/'reader_manifest.jsonl');times=[];answers={}
        assert len(manifest)==len(list((target/'responses').glob('*.json')))==p['answer_requests']
        assert len({(r['id'],r['kind']) for r in manifest})==len(manifest)
        for item in manifest:
            assert item['payload']==reader_payload(qmap[item['id']],ranked[item['id']],item['kind'])
            path=target/'responses'/(digest([item['id'],item['kind']])+'.json');saved=read(path)
            times.append(path.stat().st_mtime)
            assert saved['request']==item and saved['experiment_signature']==p['experiment_signature']
            out=saved['output'];assert out['reader_image_count']==0
            if p['backbone']=='qwen':
                assert out['seed']==p['seed']
                assert not any('pixel' in k or 'image' in k for k in out['processor_keys'])
            else:
                wire=json.loads(gzip.decompress((target/'api_cache'/item['request_sha256']/'wire.json.gz').read_bytes()))
                assert wire==item['wire'] and digest(wire)==item['request_sha256']
                assert wire['seed']==out['actual_wire_seed']==p['seed']
                assert all(isinstance(m['content'],str) for m in wire['messages'])
                assert wire['messages']==[{'role':'system','content':item['payload']['system']},{'role':'user','content':item['payload']['user_text']}]
                assert out['request_sha256']==item['request_sha256']
            answers[item['id'],item['kind']]=out['raw_response']
        predictions[p['backbone'],p['dataset'],p['seed']]=answers
        timelines.setdefault(p['seed'],[]).extend(times)
        checked.append(dict(backbone=p['backbone'],dataset=p['dataset'],seed=p['seed'],questions=len(query),answers=len(manifest),status='passed'))
    assert max(timelines[42])<=min(timelines[1024])
    assert max(timelines[1024])<=min(timelines[2026])
    for ds in ['memeye','mem_gallery']:
        target=Path(complete['results'][0].get('source',''))
        source=ROOT/'results/clip_caption_qwen_20261114'/ds/'evaluation'
        predictions['qwen',ds,42]={(r['id'],r['kind']):r['raw_response'] for r in rows(source/'scored_questions.jsonl')}
    groups={};comparisons=[]
    for r in complete['results']:
        for kind,metrics in r['metrics'].items():groups.setdefault((r['backbone'],r['dataset'],kind),[]).append((r['seed'],metrics,r['storage_mib']))
    aggregated=[]
    for (backbone,ds,kind),group in sorted(groups.items()):
        assert sorted(s for s,m,z in group)==[42,1024,2026]
        names=['accuracy'] if kind=='mcq' else ['em','f1','bleu_1','bleu_2','bleu']
        record=dict(backbone=backbone,dataset=ds,format=kind,seeds=[42,1024,2026],questions_per_seed=group[0][1]['questions'],storage_mib=group[0][2])
        record['metrics']={name:{'mean':statistics.mean(m[name] for s,m,z in group),'sample_std':statistics.stdev(m[name] for s,m,z in group)} for name in names}
        aggregated.append(record)
    for backbone in ['qwen','gpt5','deepseek']:
        for ds in ['memeye','mem_gallery']:
            a=predictions[backbone,ds,42]
            for seed in [1024,2026]:
                b=predictions[backbone,ds,seed];assert set(a)==set(b)
                comparisons.append(dict(backbone=backbone,dataset=ds,seed=seed,answer_requests=len(a),changed_answers=sum(a[k]!=b[k] for k in a)))
    dump(OUT/'independent_audit.json',dict(status='passed',new_cells=len(checked),new_answers=sum(x['answers'] for x in checked),cells=checked,
        actual_seed_order_verified=True,wire_seed_and_text_only_verified=True,rankings_independently_recomputed=True))
    dump(OUT/'seed_summary.json',dict(groups=aggregated,comparison_to_seed42=comparisons))
    print(json.dumps({'status':'passed','cells':len(checked),'answer_requests':sum(x['answers'] for x in checked),'aggregates':aggregated,'changes':comparisons}))

if __name__=='__main__':main()
