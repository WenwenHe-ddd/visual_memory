"""Original GPT-5 Open reader and fixed-seed Judge for fresh MemEye seeds."""
from pathlib import Path
import copy
from types import SimpleNamespace
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway,ENV

READER=c.ROOT/'scripts/evaluation/memeye_gpt5_open_reader.py'

def reader(job):
    g=Gateway(job)
    value=c.clone(READER,'grid_gpt_open_native',job['seed'],(152,202),old_seed=42)
    template=g.build_message_request([{'role':'user','content':'fixture'}]);template.pop('messages')
    value.api=SimpleNamespace(__file__=g.__file__,settings=lambda model='gpt-5':copy.deepcopy(template),
        chat=g.chat_request,load_config=lambda *args,**kw:g.config,APIError=g.APIError)
    value.text_api=SimpleNamespace(__file__=g.__file__,build_request=g.build_request,chat_request=g.chat_request)
    value.RUN=Path(job['run_root'])
    value.code_files=lambda:[Path(__file__),READER,Path(c.__file__),Path(g.__file__)]
    return value

def answers(job,root,correct):
    value=reader(job)
    args=SimpleNamespace(root=root,method_id=job['method'],reference_correct=correct,
        cache=Path(job['run_root'])/'open_gpt_answer_cache'/job['method'],workers=job['workers'],env_file=ENV,limit=0)
    value.run(args);value.collect(args)

def judge(job,root):
    from scripts.evaluation import memeye_gpt5_open_judge as native
    args=SimpleNamespace(input=root/'judge_inputs.jsonl',output=root/'judge',method_id=job['method'],
        method_label=c.LABELS[job['method']],source_protocol=root/'protocol.json',cache_root=[],env_file=ENV,
        workers=job['workers'],max_attempts=5,limit=0)
    p,requests,mapping=native.prepare(args)
    report=native.base.run(args,p,requests,mapping);native.publish_summary(args,p,report)
    if report['status']!='complete':raise ValueError('Incomplete native five-grade Judge')
