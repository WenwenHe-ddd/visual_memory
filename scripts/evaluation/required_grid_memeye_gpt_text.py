"""Fresh GPT-5 native text memories with frozen model-matched MemEye captions."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway,APIModel,ENV
OLD=c.ROOT/'scripts/evaluation/memeye_deepseek_seed1024_text_methods.py'
READER=c.ROOT/'scripts/evaluation/memeye_gpt5_open_reader.py'
REFS={m:c.ROOT/('results/memeye_gpt5_seed42_text_methods_20261025/'+m) for m in ('gen_agents','memoryos','a_mem')}
REFS['simplemem']=c.ROOT/'results/simplemem_official_gpt5_memeye_k10_rrf5_reflection_20261029/top5/simplemem_official_text_context5'

def runtime(job):
    g=Gateway(job);root=Path(job['run_root']);method=job['method']
    native=c.clone(OLD,'grid_gpt_text_native',job['seed'],(33,))
    native.MODEL='gpt-5';native.SEED=job['seed'];native.OUTPUT=root/'text_methods';native.OPEN_OUTPUT=root/'open/text_methods'
    native.api=lambda:SimpleNamespace(__file__=str(Path(g.__file__)),build_request=g.build_request,request_sha256=c.digest)
    native.wire=lambda messages,cap,temperature:g.build_message_request(messages,cap,temperature)
    native.new_model=lambda directory,p,env_file=ENV:APIModel(directory,job,job['workers'])
    # The actual query is still independently checked against base.retrieval_query.
    # Exclude only the stale derived Qwen-cache field from metadata equality.
    tree=ast.parse(OLD.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='open_run')
    class Derived(ast.NodeTransformer):
        changes=0
        def visit_GeneratorExp(self,n):
            self.generic_visit(n)
            if (len(n.generators)==1 and ast.unparse(n.generators[0].iter)=="question_map[row['id']].items()"
                and ast.unparse(n.elt)=='row[k] != v'):
                n.generators[0].ifs.append(ast.parse("k != 'retrieval_query'",mode='eval').body);self.changes+=1
            return n
    edit=Derived();node=edit.visit(node)
    if edit.changes!=1:raise ValueError('Derived-query guard boundary changed')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(OLD),'exec'),native.__dict__)
    def validate_calls(directory,p):
        raw={r['output']['response_id']:r for r in c.rows(directory/'api_responses.jsonl')} if (directory/'api_responses.jsonl').exists() else {}
        calls={};failures=[];counts=Counter()
        for path in (directory/'calls').glob('*.json'):
            call=c.read(path);request=call['request'];out=call['output']
            wire=native.wire(request['messages'],request['max_new_tokens'],request['temperature'])
            logged=raw.get(out.get('response_id'))
            if (native.base.hash_object(request)!=path.stem or request['seed']!=job['seed'] or out['seed']!=job['seed']
                or request['experiment_signature']!=p['experiment_signature'] or out['returned_model']!=g.expected
                or out['requested_model']!='gpt-5' or out['request_sha256']!=c.digest(wire)
                or logged is None or logged['request']!=wire
                or any(out.get(k)!=v for k,v in logged['output'].items())):
                failures.append({'check':'original_actual_GPT_wire_seed_output','call':path.stem})
            setting=(request['temperature'],request['max_new_tokens'])
            if (request['kind']=='answer' and setting!=(0.,16)) or (request['kind']=='memory' and setting not in native.NATIVE_MEMORY_SETTINGS[method]):
                failures.append({'check':'native_temperature_cap','call':path.stem})
            counts[(request['kind'],*setting)]+=1;calls[path.stem]=call
        return calls,failures,counts
    native.validate_calls=validate_calls
    reader=c.clone(READER,'grid_gpt_open_reader',job['seed'],(152,202),old_seed=42)
    template=g.build_message_request([{'role':'user','content':'x'}]);template.pop('messages')
    reader.api=SimpleNamespace(__file__=g.__file__,settings=lambda model='gpt-5':copy.deepcopy(template),
        chat=g.chat_request,load_config=lambda *args,**kw:g.config,APIError=g.APIError)
    reader.text_api=SimpleNamespace(__file__=g.__file__,build_request=g.build_request,chat_request=g.chat_request)
    reader.RUN=root
    reader.code_files=lambda:[Path(__file__),Path(g.__file__),OLD,READER,Path(c.__file__)]
    return native,reader,g

def prepare(job,native,g):
    root=native.OUTPUT/job['method'];source=REFS[job['method']]
    if (root/'protocol.json').exists():return native.verify(root)
    parent=c.read(source/'protocol.json')
    if parent['seed']!=42 or parent.get('model','gpt-5')!='gpt-5':raise ValueError('Wrong seed42 native reference')
    observations=c.read(source/'observations.json');questions=c.rows(source/'questions.jsonl')
    canonical={q['id']:q for q in c.rows(native.CANONICAL)}
    if len(questions)!=371 or sum(map(len,observations.values()))!=848:raise ValueError('Wrong input cohort')
    gold=[dict(id=q['id'],**{k:canonical[q['id']][k] for k in ('gold_choice','gold_ids','x','y')}) for q in questions]
    for q in questions:
        if any(q[k]!=canonical[q['id']][k] for k in ('question','options','task')):raise ValueError('MCQ changed')
    p=copy.deepcopy(parent)
    for key in ('experiment_signature','input_hashes','input_sha256','control','ablation','recovery'):p.pop(key,None)
    p.update(mode=job['method'],model='gpt-5',seed=job['seed'],caption_model=g.expected,
        expected_returned_model=g.expected,caption_source_run_seed=42,embedding_device=parent.get('embedding_device') or 'cpu',
        max_new_tokens=16,task_workers=job['workers'],api_workers=job['workers'],tasks=sorted(observations),
        questions=371,memory_rounds=848,questions_sha256=native.base.hash_object(questions),observations_sha256=native.base.hash_object(observations),
        gold_sha256=native.base.hash_object(gold),prior_generated_memory_reused=False,prior_reader_answers_reused=False,
        study='Required seed grid: original same-backbone captions, fresh native memory/retrieval/readers')
    if job['method']=='simplemem':
        p.update(simplemem_backend='official_text',context_fusion='channel_rrf',fusion_rank_constant=60,
                 context_memory_limit=5,preserve_empty_context=True)
    files=[source/'protocol.json',source/'observations.json',source/'questions.jsonl',native.CANONICAL,
        Path(__file__),OLD,READER,Path(g.__file__),Path(c.__file__),*native.code_files()]
    p['input_sha256']={str(Path(f).resolve()):c.sha(f) for f in set(files)}
    p['experiment_signature']=native.signature(p)
    c.freeze(root/'protocol.json',p);c.freeze(root/'observations.json',observations)
    c.write_rows(root/'questions.jsonl',questions);c.write_rows(root/'gold.jsonl',gold)
    return native.verify(root)

def stage(job,phase):
    native,reader,g=runtime(job);prepare(job,native,g);method=job['method']
    if phase=='mcq':return native.run(method,native.OUTPUT)
    if phase=='open-retrieval':return native.open_run(method,native.OUTPUT,native.OPEN_OUTPUT)
    opened=native.OPEN_OUTPUT/method
    args=SimpleNamespace(root=opened,method_id=method,reference_correct=c.read(native.OUTPUT/method/'summary.json')['correct'],
        cache=Path(job['run_root'])/'gpt_text_answer_cache'/method,workers=job['workers'],env_file=ENV,limit=0)
    if phase=='open-answers':reader.run(args);reader.collect(args);return
    if phase=='judge':
        from scripts.evaluation import memeye_gpt5_open_judge as judge
        a=SimpleNamespace(input=opened/'judge_inputs.jsonl',output=opened/'judge',method_id=method,
            method_label=c.LABELS[method],source_protocol=opened/'protocol.json',cache_root=[],env_file=ENV,
            workers=job['workers'],max_attempts=5,limit=0)
        p,requests,mapping=judge.prepare(a);report=judge.base.run(a,p,requests,mapping);judge.publish_summary(a,p,report)
        if report['status']!='complete':raise ValueError('Incomplete Judge')
        return
    if phase!='audit':raise ValueError('Unknown native stage')
    source=native.OUTPUT/method;native.audit(method,native.OUTPUT)
    current=c.rows(source/'qa_responses.jsonl');opened_rows=c.rows(opened/'reader/responses.jsonl')
    if any(r['seed']!=job['seed'] or r['returned_model']!=g.expected for r in opened_rows):raise ValueError('Open seed/revision differs')
    c.complete(job,[dict(r,native_outcome='ok' if r['predicted_choice'] else 'native_invalid_choice') for r in current],
        [dict(r,model_output=r['raw_response']) for r in opened_rows],opened/'judge',
        [source/'qa_responses.jsonl',source/'summary.json',source/'completion_manifest.json',
         opened/'reader/responses.jsonl',opened/'reader/completion.json',Path(__file__),OLD,READER,Path(g.__file__)],
        dict(caption_seed=c.read(source/'protocol.json').get('caption_seed'),caption_source_run_seed=42))

def run(job,path):
    if job['dataset']!='memeye' or job['backbone']!='gpt5':raise ValueError('Wrong native handler')
    c.job_protocol(job,[Path(__file__),Path(c.__file__),OLD,READER,c.ROOT/'scripts/evaluation/required_grid_api.py'])
    for phase in ('mcq','open-retrieval','open-answers','judge','audit'):
        print({'id':job['id'],'stage':phase},flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase],cwd=c.ROOT,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        native,reader,g=runtime(j);prepare(j,native,g);print({'status':'prepared','id':j['id'],'api_calls':0})
    elif a.stage:stage(j,a.stage)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
