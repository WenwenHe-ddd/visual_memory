"""Official SimpleMem 0.3.0 on Gallery, using the matched local Qwen reader.

Uses the recently completed MemEye configuration (K10/reflection/channel RRF5).
Only Gallery input metadata and open-answer reader differ. No gold in inference.
"""
import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from scripts.evaluation import mem_gallery_qwen_text_methods as base
from scripts.memory import simplemem_official as official

ROOT=base.ROOT
OUTPUT=ROOT/'results/simplemem_official_qwen_memgallery_k10_rrf5_reflection_20261029'
REFERENCE=ROOT/'results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028/top5/simplemem_official_text_context5/protocol.json'
METHOD='simplemem'

class GalleryMemory(official.OfficialSimpleMemMemory):
    def __init__(self, method, protocol, workdir):
        assert method == METHOD
        super().__init__(protocol,workdir)
    def store(self,row):
        row=copy.deepcopy(row)
        ids=row['raw'].get('image_id') or []
        if isinstance(ids,str): ids=[ids]
        captions=row['raw'].get('image_caption') or []
        assert len(ids)==len(captions)
        row['user']='Date: '+row['timestamp']+'\nRound: '+row['id']+'\n'+row['user']
        # Keep each original ID adjacent to its unmodified caption in official input.
        row['raw']['image_caption']=['image_id: '+i+'\nimage_caption: '+c for i,c in zip(ids,captions)]
        super().store(row)
    def recall(self,query):
        self.finalize()
        return super().recall(query)


def verify(output,method):
    p=base.read(output/method/'protocol.json')
    assert p['experiment_signature']==base.obj({k:v for k,v in p.items() if k!='experiment_signature'})
    assert official.verify_installation()==p['upstream']
    for file,digest in p['input_sha256'].items(): assert base.sha(file)==digest,file
    return p


def prepare(output):
    if (output/METHOD/'protocol.json').exists(): return verify(output,METHOD)
    parent=base.read(REFERENCE)
    upstream=official.verify_installation()
    assert upstream['version']=='0.3.0'
    source=base.SOURCE/'inputs'
    for name,digest in base.read(source/'artifact_manifest.json').items(): assert base.sha(source/name)==digest
    qs=base.rows(source/'questions.jsonl'); memories=base.rows(source/'memories.jsonl')
    assert len(qs)==1711 and len(memories)==3962
    assert all(not ({'answer','clue','gold_ids','options'} & q.keys()) for q in qs)
    obs=base.make_observations(memories)
    (output/'inputs').mkdir(parents=True,exist_ok=True)
    for name in ('questions.jsonl','memories.jsonl','gold.jsonl','protocol.json','artifact_manifest.json'):
        shutil.copyfile(source/name,output/'inputs'/name)
    base.dump(output/'observations.json',obs)
    p={k:copy.deepcopy(parent[k]) for k in ('method_defaults','embedding_model_path','context_fusion','fusion_rank_constant','context_memory_limit','preserve_empty_context','memory_max_new_tokens','context_budget')}
    assert p['method_defaults']['ENABLE_REFLECTION'] and p['method_defaults']['SEMANTIC_TOP_K']==10
    p.update(mode=METHOD,simplemem_backend='official_text',upstream=upstream,model=base.gallery.model_provenance(),reader_model='Qwen3-VL-8B-Instruct',seed=42,do_sample=False,dtype='bfloat16',max_new_tokens=256,query_image_max_pixels=base.gallery.QUERY_IMAGE_MAX_PIXELS,reader_system_prompt=base.gallery.READER_SYSTEM,task_instructions=base.gallery.TASK_INSTRUCTIONS,generation_batch_size=2,max_padded_batch_tokens=48000,task_workers=8,reader_batch_size_text=8,reader_batch_size_images=2,tasks=sorted(obs),questions=1711,memory_rounds=3962,questions_sha256=base.obj(qs),observations_sha256=base.obj(obs),reference_config=str(REFERENCE),reference_config_sha256=base.sha(REFERENCE),caption_policy='Reuse verified Qwen history captions; official supplied query captions; original query pixels for final reader',metadata_adaptation='Preserve source date/round in user text and image_id adjacent to its original caption; upstream algorithms unchanged',gold_loaded_only_in_analysis=True,source_recall_policy='Construction-window membership is not exact fact attribution. Do not report it as evidence recall.',retrieval_count_policy=parent['retrieval_count_policy'])
    deps=[Path(__file__),Path(base.__file__),Path(base.native.__file__),Path(base.gallery.__file__),ROOT/'scripts/memory/text_memory_transport.py',ROOT/'scripts/memory/reflexion_memory.py',ROOT/'scripts/data/mem_gallery_inputs.py',REFERENCE,*official.source_files(),output/'observations.json',*[output/'inputs'/n for n in ('questions.jsonl','memories.jsonl','gold.jsonl')]]
    p['input_sha256']={str(f.resolve()):base.sha(f) for f in deps}
    p['experiment_signature']=base.obj(p)
    snapshots={}
    for f in deps:
        if f.suffix=='.py':
            target=output/'source_snapshot'/(base.sha(f)[:12]+'_'+f.name)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
            snapshots[str(f.resolve())]={'path':str(target),'sha256':base.sha(target)}
    base.dump(output/'source_snapshot/manifest.json',snapshots)
    base.dump(output/METHOD/'protocol.json',p)
    print(json.dumps({'prepared':str(output),'questions':len(qs),'rounds':len(memories),'version':upstream['version']}),flush=True)
    return p


def analyze(output):
    p=verify(output,METHOD); root=output/METHOD
    qs=base.rows(output/'inputs/questions.jsonl')
    gold={r['id']:r for r in base.rows(output/'inputs/gold.jsonl')}
    answers=base.rows(root/'reader/responses.jsonl'); retrieval=base.rows(root/'retrieval.jsonl')
    amap={r['id']:r for r in answers};rmap={r['id']:r for r in retrieval}
    assert len(amap)==len(answers)==len(rmap)==len(retrieval)==1711
    assert set(amap)==set(rmap)==set(gold)=={q['id'] for q in qs}
    scored=[]
    for q in qs:
        a=amap[q['id']];r=rmap[q['id']]
        assert a['experiment_signature']==r['experiment_signature']==p['experiment_signature']
        detail=r['retrieval_detail'];ids=detail['context_memory_ids']
        assert len(ids)==len(set(ids)) and len(ids) in (0,5)
        scored.append(dict(id=q['id'],task=q['task'],point=q['point'],answer=gold[q['id']]['answer'],raw_response=a['raw_response'],**base.gallery.score_answer(a['raw_response'],gold[q['id']]['answer']),context_count=len(ids),hit_token_limit=a['hit_token_limit']))
    def stats(items):
        return dict(questions=len(items),correct=sum(int(r['em']) for r in items),**{k:sum(r[k] for r in items)/len(items) for k in ('em','f1','bleu_1','bleu_2','bleu')},context_counts=dict(Counter(r['context_count'] for r in items)),empty_answers=sum(not r['raw_response'].strip() for r in items),answer_token_limit_records=sum(r['hit_token_limit'] for r in items))
    result=dict(status='complete',method='SimpleMem official 0.3.0 (K10, reflection, RRF5)',model='Qwen3-VL-8B-Instruct',overall=stats(scored),by_type={k:stats([r for r in scored if r['point']==k]) for k in sorted({r['point'] for r in scored})},by_task={k:stats([r for r in scored if r['task']==k]) for k in p['tasks']},source_recall=None,llm_judge=None,source_recall_note=p['source_recall_policy'])
    base.write_rows(output/'scored_questions.jsonl',scored)
    base.dump(output/'final_results.json',result)
    # Verify actual builder coverage and stored entries, including tail windows.
    obs=base.read(output/'observations.json');audit=[]
    for task in p['tasks']:
        target=root/task;windows=base.read(target/'runtime/construction_windows.json');src=base.read(target/'runtime/dialogue_sources.json')
        assert {s['round_id'] for s in src.values()}=={r['id'] for r in obs[task]}
        assert {i for w in windows for i in w['dialogue_ids']}=={int(i) for i in src}
        assert all(w['parsed_successfully'] for w in windows)
        entries=base.read(target/'final_memory.json');known={r['entry_id'] for r in entries}
        for r in retrieval:
            if r['task']==task: assert set(r['retrieval_detail']['context_memory_ids'])<=known
        audit.append(dict(task=task,rounds=len(obs[task]),entries=len(entries),windows=len(windows)))
    calls=list((root/'calls').glob('*.json'));limits=0
    for path in calls:
        c=base.read(path);assert base.obj(c['request'])==path.stem;assert c['request']['experiment_signature']==p['experiment_signature']
        limits+=bool(c['output']['hit_token_limit'])
    base.dump(output/'audit.json',dict(status='passed',scenarios=audit,questions=1711,memory_calls=len(calls),memory_token_limit_records=limits,original_inputs_verified=True,source_recall_not_inferred_from_windows=True))
    print(json.dumps(result['overall']),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','analyze','all']);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.stage in ('prepare','all'):prepare(args.output)
    if args.stage in ('run','all'):
        base.verify=verify;base.Memory=GalleryMemory
        base.run(args.output,METHOD,args.device)
    if args.stage in ('analyze','all'):analyze(args.output)

if __name__=='__main__':main()
