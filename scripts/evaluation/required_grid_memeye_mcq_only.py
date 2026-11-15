"""MemEye-only continuation: native memory, retrieval and MCQ; no Open/Judge stages.

The original signed handlers and running scheduler remain unchanged. Use this
entrypoint for future MemEye retries; completion_mcq.json is a separate proof
and never impersonates an MCQ+Open+Judge completion.json.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
POLICY=c.RUN/'scheduler/memeye_evaluation_policy.json'

def stages(job):
    if job['dataset']!='memeye':
        raise ValueError('MemGallery must retain its original evaluation pipeline')
    if job['backbone']=='gpt5':
        if job['method'] in ('semrag','ours_1_ours_2'):
            return ['memory','mcq-retrieval','mcq-answer','mcq-audit']
        return ['native-mcq','mcq-audit']
    if job['method']=='hippomm' and job['backbone'] in ('qwen','deepseek'):
        return ['native-mcq','mcq-audit']
    raise ValueError('No outstanding native MCQ continuation registered for this combination')

def native_run(job):
    bb=job['backbone'];method=job['method'];run=Path(job['run_root'])
    if bb=='gpt5' and method in ('semrag','ours_1_ours_2'):
        from scripts.evaluation import required_grid_memeye_gpt_caption_cuda as handler
        from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
        with gpu(job) as device:
            value=handler.runtime(device);root=run/'caption_methods_cuda'
            with c.lock(root/'common.lock'):
                p=value.prepare(job);value.build(job,root,p)
                value.retrieve(job,root,p,'mcq')
        value.mcq(job,root)
        return Path(job['method_root'])/'native_mcq.jsonl'
    if bb=='gpt5' and method in ('gen_agents','memoryos','simplemem','a_mem'):
        from scripts.evaluation import required_grid_memeye_gpt_text as handler
        native,_,gateway=handler.runtime(job);handler.prepare(job,native,gateway)
        native.run(method,native.OUTPUT);native.audit(method,native.OUTPUT)
        return native.OUTPUT/method/'qa_responses.jsonl'
    if bb=='gpt5':
        from scripts.evaluation import required_grid_memeye_gpt_visual as handler
        handler.stage(job,'prepare-retrieval');handler.stage(job,'mcq')
        return Path(job['method_root'])/'native/mcq_records.jsonl'
    if bb=='qwen':
        from scripts.evaluation import required_grid_memeye_qwen_visual as handler
        from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
        with gpu(job) as device:handler.stage(job,'mcq',device)
        return run/'visual_methods/hippomm/online/qa_responses.jsonl'
    from scripts.evaluation import required_grid_memeye_deepseek_hippo as handler
    value,args,_=handler.runtime(job);value.prepare(args);value.run(args)
    return args.output_root/'hippomm/online/qa_responses.jsonl'

def complete(job,source):
    gold={r['id']:r for r in c.rows(c.ROOT/'results/mcq_top15_caption_qa/qa_manifest.jsonl')}
    raw=c.rows(source);ids=[r['id'] for r in raw]
    if len(ids)!=len(set(ids)) or set(ids)!=set(gold):
        raise ValueError('MCQ requires every canonical question exactly once')
    answers=[]
    for row in raw:
        qid=row['id'];choice=row.get('predicted_choice',row.get('answer'))
        choice=choice if choice in ('A','B','C','D') else None
        if row.get('gold_choice',gold[qid]['gold_choice'])!=gold[qid]['gold_choice']:
            raise ValueError('Gold choice differs')
        answers.append(dict(id=qid,predicted_choice=choice,gold_choice=gold[qid]['gold_choice'],correct=choice==gold[qid]['gold_choice']))
    root=Path(job['method_root']);out=root/'mcq_only_records.jsonl';c.write_rows(out,answers)
    proof=dict(status='complete',evaluation_scope='mcq_only',dataset='memeye',backbone=job['backbone'],
        seed=job['seed'],method=job['method'],questions=len(gold),correct=sum(r['correct'] for r in answers),
        accuracy=sum(r['correct'] for r in answers)/len(gold),llm_judge=None,judge_status='not_requested',
        open_questions=0,judge_questions=0,completed_at=c.now(),
        source_sha256={str(p):c.sha(p) for p in [source,Path(__file__),POLICY]},
        artifact_sha256={str(out):c.sha(out)})
    c.dump(root/'completion_mcq.json',proof)
    return proof

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();job=c.read(args.job);schedule=stages(job)
    policy=c.read(POLICY)
    if policy['future_memeye_llm_judge'] or policy['future_memeye_open_answers']:
        raise ValueError('Expected MCQ-only policy')
    if args.plan_only:print(dict(id=job['id'],stages=schedule,llm_judge=False,open_answers=False))
    else:
        with c.lock(Path(job['method_root'])/'execution.lock',False):
            print(complete(job,native_run(job)))
