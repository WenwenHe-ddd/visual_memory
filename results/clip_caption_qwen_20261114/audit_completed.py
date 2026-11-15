"""Independent artifact, coverage, ranking and caption payload audit."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0,str(ROOT))
from scripts.memory.clip_caption import rows, read, sha, dump, digest, reader_payload
OUT=Path(__file__).resolve().parent

def main():
    completion=read(OUT/'completion_manifest.json')
    assert completion['status']=='complete'
    for filename,expected in completion['artifact_sha256'].items():
        assert sha(filename)==expected,filename
    results={};csv_rows=[]
    for dataset,total in [('memeye',371),('mem_gallery',1711)]:
        folder=OUT/dataset;memory=folder/'memory';evaluation=folder/'evaluation'
        summary=read(evaluation/'summary.json');audit=read(evaluation/'audit.json')
        for filename,expected in audit['artifact_sha256'].items():
            assert sha(filename)==expected,filename
        records=rows(memory/'records.jsonl');vectors=np.load(memory/'image_embeddings.npy').astype(np.float64)
        questions=rows(evaluation/'questions.jsonl');retrieval=rows(evaluation/'retrieval.jsonl')
        queries=np.load(evaluation/'query_embeddings.npy').astype(np.float64)
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        queries/=np.linalg.norm(queries,axis=1,keepdims=True)
        manifest=rows(evaluation/'reader_manifest.jsonl')
        scored=rows(evaluation/'scored_questions.jsonl')
        assert len(questions)==total
        assert len({q['id'] for q in questions})==total
        assert len(manifest)==len(scored)==(total*2 if dataset=='memeye' else total)
        assert len(list((evaluation/'responses').glob('*.json')))==len(manifest)
        assert set(p.name for p in memory.iterdir())=={'records.jsonl','image_embeddings.npy','memory.json'}
        assert all(set(r)=={'id','task','caption','image_ids','image_sha256','source_memory_ids'} for r in records)
        selected={}
        for q,r,v in zip(questions,retrieval,queries):
            indices=[i for i,record in enumerate(records) if record['task']==q['task']]
            scores=vectors[indices]@v
            ranking=np.argsort(-scores,kind='stable')[:5]
            hits=[records[indices[i]] for i in ranking]
            assert r['selected_ids']==[h['id'] for h in hits],q['id']
            assert np.allclose(r['scores'],scores[ranking],atol=1e-6,rtol=0)
            selected[q['id']]=hits
        question_map={q['id']:q for q in questions}
        response_keys=set()
        for request in manifest:
            assert request['payload']==reader_payload(question_map[request['id']],selected[request['id']],request['kind'])
            response=read(evaluation/'responses'/(digest([request['id'],request['kind']])+'.json'))
            assert response['request']==request
            output=response['output'];assert output['reader_image_count']==0
            assert not any('pixel' in k or 'image' in k or 'video' in k for k in output['processor_keys'])
            response_keys.update(output['processor_keys'])
        by_group={}
        for group in sorted({'all'}|{q.get('point','memeye') for q in questions}|{'query_image','no_query_image'}):
            ids={q['id'] for q in questions if group=='all' or q.get('point','memeye')==group or (group=='query_image' and q['has_query_image']) or (group=='no_query_image' and not q['has_query_image'])}
            group_metrics={}
            for kind in summary['metrics']:
                values=[r for r in scored if r['id'] in ids and r['kind']==kind]
                if not values:continue
                metric_names=['correct'] if kind=='mcq' else ['em','f1','bleu_1','bleu_2','bleu']
                metrics={('accuracy' if k=='correct' else k):sum(r[k] for r in values)/len(values) for k in metric_names}
                metrics['questions']=len(values)
                group_metrics[kind]=metrics
                csv_rows.append(dict(dataset=dataset,group=group,kind=kind,**metrics))
            by_group[group]=group_metrics
        results[dataset]=dict(status='passed',questions=total,answer_requests=len(manifest),rankings_recomputed_in_float64=total,
            memory_file_bytes={p.name:p.stat().st_size for p in memory.iterdir()},actual_processor_keys=sorted(response_keys),metrics_by_group=by_group)
    dump(OUT/'independent_audit.json',dict(status='passed',datasets=results))
    with (OUT/'metrics_by_group.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','group','kind','questions','accuracy','em','f1','bleu_1','bleu_2','bleu'])
        writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({k:dict(status=v['status'],questions=v['questions'],answers=v['answer_requests']) for k,v in results.items()}))

if __name__=='__main__':main()
