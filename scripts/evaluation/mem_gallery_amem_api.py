"""A-Mem on full Mem-Gallery with model-matched cached captions and API backbones.

Uses existing native A-Mem storage/evolution/query/link expansion; fixed primary
K5 and five complete distinct final notes. Original query pixels reach reader.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from scripts.evaluation import mem_gallery_deepseek_text_methods as runner
from scripts.evaluation import mem_gallery_gpt5_memoryos as gpt
from scripts.evaluation import mem_gallery_gpt5_image_formats as gpt_images
from scripts.evaluation import gallery_deepseek_memory_transport as deep_memory
from scripts.evaluation import gallery_deepseek_outcomes as deep_reader
from scripts.evaluation import gallery_deepseek_bounded_images as deep_images

base=runner.ga
ROOT=base.ROOT
read,rows,sha,obj,dump,write_rows,freeze=base.read,base.rows,base.sha,base.obj,base.dump,base.write_rows,base.freeze
SOURCES={'gpt5':base.DEFAULT_OUTPUT,'deepseek':runner.SOURCE}
OUTPUT=ROOT/'results/mem_gallery_amem_backbones_20261104'
REFERENCE=ROOT/'results/mem_gallery_qwen_seed42_text_methods_20261026_v2/a_mem/protocol.json'


def verify_captions(source,backbone):
    marker=read(source/'caption_source/complete.json')
    if backbone=='gpt5':
        assert sha(source/'observations.json')==marker['observations_sha256']
        assert sha(source/'caption_source/captions.jsonl')==marker['captions_sha256']
    else:
        for f,h in marker['artifact_sha256'].items():assert sha(f)==h,f
        assert read(source/'caption_audit.json')['status']=='passed'
    caps=rows(source/'caption_source/captions.jsonl');cmap={c['id']:c for c in caps}
    assert len(caps)==len(cmap)==979
    model=read(source/'protocol.json')['expected_model']
    rejected=0
    for ident,c in cmap.items():
        saved=read(source/'caption_source/responses'/(ident+'.json'));out=saved['output']
        if out.get('status')=='provider_rejection':
            assert backbone=='deepseek' and c['caption']=='';rejected+=1
        else:
            assert out['returned_model']==model and out['seed']==42
            assert out['finish_reason']=='stop' and not out.get('refusal')
            assert c['caption']==out['raw_response'].strip() and c['caption'].strip()
    obs=read(source/'observations.json')
    assert len(obs)==20 and sum(map(len,obs.values()))==3962
    for group in obs.values():
        for r in group:assert r['raw']['image_caption']==[cmap[k]['caption'] for k in r['caption_ids']]
    return dict(status='passed',captions=979,provider_rejections=rejected,model=model)


def prepare(output,backbone,env):
    if (output/'protocol.json').exists():return gpt.verify(output)
    source=SOURCES[backbone];cap_audit=verify_captions(source,backbone)
    reference=read(REFERENCE);upstream=base.native.verify_vendor()
    output.mkdir(parents=True,exist_ok=True)
    for name in ('observations.json','inputs/questions.jsonl','inputs/gold.jsonl'):
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
    qs=rows(output/'inputs/questions.jsonl');obs=read(output/'observations.json')
    assert len(qs)==len({q['id'] for q in qs})==1711
    for q in qs:
        assert set(q)=={'id','task','question','point','query_image_paths','query_image_caption','retrieval_query'}
        assert q['retrieval_query']==q['question']+("\nquestion's image:\nimage_caption: "+q['query_image_caption'] if q['query_image_caption'] else '')
    # Short routing probe, no dataset labels and no memory content.
    payload={'user_text':'Reply with OK.','query_images':[]}
    if backbone=='gpt5':
        model=gpt.Model(output/'probe',env,cap_audit['model']);wire=gpt_images.wire_for(payload,2048)
    else:
        model=deep_reader.Model(output/'probe',env,deep_reader.MODEL);wire=deep_images.wire_for(payload,2048)
    out=model.invoke(wire,'probe','routing')
    assert out['finish_reason']=='stop' and out['returned_model']==cap_audit['model']
    dump(output/'api_probe.json',out);dump(output/'caption_reuse_audit.json',cap_audit)
    p=dict(schema='mem_gallery_amem_api_v1',mode='a_mem',method='A-Mem (primary K5, final context5)',backbone=backbone,
        requested_model='gpt-5' if backbone=='gpt5' else deep_reader.MODEL,expected_model=cap_audit['model'],seed=42,
        reasoning_effort='minimal' if backbone=='gpt5' else None,enable_thinking=False if backbone=='deepseek' else None,
        temperature_wire_policy='omitted' if backbone=='gpt5' else 'native memory temperature; reader temperature0',
        task_workers=8,api_workers=8,memory_max_new_tokens=2048,memory_output_floor=2048,max_new_tokens=2048,
        memory_budget_policy='max(2048,native requested cap); native A-Mem requests1000',
        method_defaults=reference['method_defaults'],context_memory_limit=5,embedding_device='cpu',
        embedding_model_path=reference['embedding_model_path'],upstream=upstream,tasks=sorted(obs),questions=1711,memory_rounds=3962,
        reference_config=str(REFERENCE),caption_source=str(source),caption_reuse=cap_audit,
        query_policy='Original question and official supplied caption; no gold, clue IDs or options.',
        reader_policy='Shared Gallery system/task prompts; original question images; GPT original bytes, DeepSeek existing oversized-image byte-limit adapter.',
        reader_system_prompt=base.gallery.READER_SYSTEM,task_instructions=base.gallery.TASK_INSTRUCTIONS,
        memory_policy='Original round dialogue and model-matched caption/image_id; timestamp and round ID retained by native wrapper. Native note metadata, evolution and links unchanged.',
        retrieval_policy='Generate query keywords; MiniLM cosine K5; interleave link expansion; first five distinct complete notes.',
        gold_policy='Gold hashed at prepare, only opened during analysis after all answers complete.',
        failure_policy='Terminal memory API error or truncated response stops run. DeepSeek reader content rejection retained as zero; no retry of rejected content.')
    assert p['method_defaults']['retrieve_k']==5
    paths={Path(__file__),REFERENCE,source/'protocol.json',source/'caption_source/complete.json',source/'caption_source/captions.jsonl',source/'observations.json',output/'api_probe.json',output/'caption_reuse_audit.json',output/'observations.json',output/'inputs/questions.jsonl',output/'inputs/gold.jsonl'}
    paths.update(base.native.VENDOR/name for name in upstream['files'])
    paths.update(f for f in Path(p['embedding_model_path']).rglob('*') if f.is_file())
    for q in qs:paths.update(Path(f) for f in q['query_image_paths'])
    for mod in tuple(sys.modules.values()):
        filename=getattr(mod,'__file__',None)
        if filename and str(filename).startswith(str(ROOT/'scripts')) and str(filename).endswith('.py'):paths.add(Path(filename))
    p['source_sha256']={str(f.resolve()):sha(f) for f in paths}
    snapshots={}
    for f in paths:
        if f.suffix=='.py':
            dest=output/'source_snapshot'/(sha(f)[:16]+'_'+f.name);dest.parent.mkdir(exist_ok=True);shutil.copyfile(f,dest)
            snapshots[str(f.resolve())]=dict(path=str(dest),sha256=sha(dest))
    freeze(output/'source_snapshot/manifest.json',snapshots)
    p['experiment_signature']=obj(p);freeze(output/'protocol.json',p)
    print('Prepared',backbone,1711,'questions; captions',cap_audit,flush=True)
    return p


class AMem(base.native.Memory):
    def __init__(self,p,workdir):
        super().__init__('a_mem',p,workdir)
        dump(workdir/'resolved_config.json',dict(method='a_mem',actual_model=p['expected_model'],retrieve_k=self.core.retrieve_k,context_memory_limit=p['context_memory_limit'],embedding_model_path=p['embedding_model_path']))


def configure(p):
    runner.NativeMemory=AMem
    if p['backbone']=='gpt5':
        runner.memory_api=SimpleNamespace(Model=lambda root,env,workers,floor:gpt.Model(root,env,p['expected_model']))
        runner.reader_api=SimpleNamespace(Model=gpt.Model,request_sha256=base.request_sha256)
        runner.images=gpt_images


def render_notes(notes,ids):
    return ''.join('talk start time:'+notes[k]['timestamp']+'memory content: '+notes[k]['content']+'memory context: '+notes[k]['context']+'memory keywords: '+str(notes[k]['keywords'])+'memory tags: '+str(notes[k]['tags'])+'\n' for k in ids)


def analyze(output):
    p=gpt.verify(output);root=output/'memory';reader=output/'reader'
    for stage in (root,reader):
        done=read(stage/'complete.json');assert done['status']=='complete'
        for f,h in done['artifact_sha256'].items():assert sha(f)==h,f
    qs=rows(output/'inputs/questions.jsonl');obs=read(output/'observations.json')
    gold={r['id']:r for r in rows(output/'inputs/gold.jsonl')}
    answers=rows(reader/'responses.jsonl');retrieved=rows(root/'retrieval.jsonl');manifests=rows(reader/'manifest.jsonl')
    amap={r['id']:r for r in answers};rmap={r['id']:r for r in retrieved};mmap={r['record_id']:r for r in manifests}
    expected={q['id'] for q in qs};assert len(expected)==len(answers)==len(retrieved)==len(manifests)==1711
    assert expected==set(amap)==set(rmap)==set(mmap)==set(gold)
    notes={t:read(root/t/'final_memory.json') for t in p['tasks']}
    helper=base.native.helpers('a_mem')
    for task in p['tasks']:
        assert set(notes[task])=={r['id'] for r in obs[task]}
        for r in obs[task]:
            note=notes[task][r['id']]
            assert note['content']==helper['build_a_mem_note_text'](r,r['speaker'],'assistant')
            assert note['timestamp']==r['timestamp']
    scored=[]
    for q in qs:
        a=amap[q['id']];r=rmap[q['id']];ids=r['retrieval_detail']['context_memory_ids']
        assert len(ids)==len(set(ids))==5
        assert r['retrieved_context']==render_notes(notes[q['task']],ids)
        assert mmap[q['id']]==base.reader_payload(q,r['retrieved_context'])
        assert a['experiment_signature']==r['experiment_signature']==p['experiment_signature']
        rejected=a.get('status')=='provider_rejection'
        if not rejected:assert a['returned_model']==p['expected_model'] and a['finish_reason']=='stop' and not a.get('refusal')
        target=gold[q['id']];sources={q['task']+'/'+i for i in ids};gids=set(target['gold_ids'])
        scored.append(dict(id=q['id'],task=q['task'],point=q['point'],answer=target['answer'],raw_response=a['raw_response'],**base.gallery.score_answer(a['raw_response'],target['answer']),context_count=5,source_ids=sorted(sources),gold_ids=sorted(gids),gold_hits=len(sources&gids),gold_count=len(gids),provider_rejection=rejected))
    def stats(rr):
        vv=[r for r in rr if r['gold_count']]
        return dict(questions=len(rr),correct=sum(int(r['em']) for r in rr),**{k:sum(r[k] for r in rr)/len(rr) for k in ('em','f1','bleu_1','bleu_2','bleu')},source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in vv)/len(vv) if vv else None,source_recall_micro=sum(r['gold_hits'] for r in vv)/sum(r['gold_count'] for r in vv) if vv else None,provider_rejections=sum(r['provider_rejection'] for r in rr))
    calls=list((root/'calls').glob('*.json'));invalid=[]
    for f in calls:
        c=read(f);assert obj(c['request'])==f.stem and c['request']['experiment_signature']==p['experiment_signature']
        out=c['output'];assert out['returned_model']==p['expected_model'] and out['finish_reason']=='stop' and not out.get('refusal')
        if out.get('structured_output_valid') is False:invalid.append(f.stem)
    result=dict(status='complete',model=p['expected_model'],method=p['method'],overall=stats(scored),by_type={t:stats([r for r in scored if r['point']==t]) for t in sorted({r['point'] for r in scored})},memory_calls=len(calls),structured_format_validation_failures=len(invalid),llm_judge=None)
    write_rows(output/'scored_questions.jsonl',scored);dump(output/'final_results.json',result)
    dump(output/'audit.json',dict(status='passed',questions=1711,original_notes=3962,final_contexts_reconstructed=1711,native_note_content_and_dates_verified=True,model_verified=True,artifact_hashes_verified=True,structured_format_failure_call_keys=invalid))
    print(json.dumps(result['overall']),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','memory','answers','analyze','all']);parser.add_argument('--backbone',choices=SOURCES,required=True);parser.add_argument('--output',type=Path);parser.add_argument('--env-file',type=Path,default=base.DEFAULT_ENV);args=parser.parse_args();output=args.output or OUTPUT/args.backbone
    if args.stage in ('prepare','all'):prepare(output,args.backbone,args.env_file)
    p=gpt.verify(output);configure(p)
    if args.stage in ('memory','all'):runner.memory_stage(output,args.env_file)
    if args.stage in ('answers','all'):runner.answers(output,args.env_file)
    if args.stage in ('analyze','all'):analyze(output)

if __name__=='__main__':main()
