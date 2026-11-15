"""Fresh HippoMM seed completion using the already audited native API pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c

OLD=c.ROOT/'scripts/evaluation/memeye_deepseek_seed1024_visual.py'
LINES=(83,106,148,159,161,192,220,244,275,316,331,337,356,380,396,410,429,447,462,486,493)

def runtime(job):
    if job['seed']!=2026 or job['dataset']!='memeye' or job['backbone']!='deepseek' or job['method']!='hippomm':
        raise ValueError('This reviewed handler completes only the missing MemEye/DeepSeek/2026 HippoMM cell')
    from scripts.evaluation import memeye_deepseek_seed2026_transport as transport
    value=c.clone(OLD,'grid_deepseek_hippo_runtime',seed=job['seed'],seed_lines=LINES)
    value.api=transport
    args=SimpleNamespace(method='hippomm',output_root=Path(job['run_root'])/'visual_methods',
                         device='cpu',workers=job['workers'],open_workers=job['workers'])
    value.patch(args)
    return value,args,transport

def run(job):
    value,args,api=runtime(job)
    sources=[Path(__file__),Path(c.__file__),OLD,Path(api.__file__),*value.source_files()]
    c.job_protocol(job,sources)
    value.prepare(args)
    value.run(args)
    value.open_prepare(args)
    value.hip_open_run(args)
    source=args.output_root/'hippomm'
    from scripts.evaluation.memeye_deepseek_seed1024_open import judge
    judge(source/'open','hippomm','HippoMM',workers=job['workers'])
    questions={r['id']:r for r in c.rows(c.ROOT/'results/mcq_top15_caption_qa/qa_manifest.jsonl')}
    raw=c.rows(source/'online/qa_responses.jsonl')
    mcq=[]
    for row in raw:
        qid=row.get('id',row.get('question_id'))
        choice=row['answer']
        if choice not in ('A','B','C','D'):choice=None
        mcq.append(dict(id=qid,task=questions[qid]['task'],question=questions[qid]['question'],
            predicted_choice=choice,gold_choice=questions[qid]['gold_choice'],
            correct=choice==questions[qid]['gold_choice'],raw_response=row.get('raw_response',row.get('answer')),
            native_outcome='ok' if choice else 'native_invalid_choice'))
    native_open=c.rows(source/'open/responses.jsonl')
    opened=[dict(r,model_output=r['raw_response']) for r in native_open]
    if len(raw)!=371 or len(native_open)!=371:
        raise ValueError('Incomplete native coverage')
    if any(r['seed']!=job['seed'] for r in native_open):
        raise ValueError('Open actual generation seed differs')
    # The native cache contains the full original-image wire for every model
    # role. Check original response metadata before publishing the final marker.
    count=0
    for base in (source/'offline/llm_cache',source/'online',source/'open'):
        for path in base.rglob('call_*.json'):
            item=c.read(path)
            if 'response' not in item:continue
            response=item['response']
            if response['seed']!=2026 or response['returned_model']!=api.MODEL:
                raise ValueError('Native HippoMM response seed/model differs')
            count+=1
    if not count:raise ValueError('No native response provenance')
    result=c.complete(job,mcq,opened,source/'open/judge',
        [source/'online/summary.json',source/'online/qa_responses.jsonl',source/'open/responses.jsonl',
         source/'open/completion.json',source/'offline/memory.json',Path(__file__),OLD,Path(api.__file__)],
        dict(native_model_calls_checked=count,context_metric='Native variable HippoMM context; not literal Recall@5'))
    print({'status':result['status'],'correct':result['correct'],'llm_judge':result['llm_judge']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    j=c.read(a.job)
    if a.prepare_only:
        value,args,api=runtime(j);value.prepare(args)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j)
