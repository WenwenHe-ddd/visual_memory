"""Independent stdlib audit for the four DeepSeek text-memory experiments."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/memeye_deepseek_flash_seed42_text_methods_20261025'
SOURCE=ROOT/'results/memeye_deepseek_flash_seed42_full_20261025'
METHODS=('gen_agents','memoryos','simplemem','a_mem')
MODEL='deepseek-v4.1-flash'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def wire_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def query(q):
    value=q['question']
    if q['query_image_captions']:
        value+="\nquestion's image:\nimage_caption: "+' '.join(q['query_image_captions'])
    return value


def answer_messages(q,context,p):
    return [dict(role='system',content=p['reader_system_prompt']),
        dict(role='assistant',content='Retrieved memory context:\n'+context),
        dict(role='user',content=query(q)+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.')]


def check_wire(req):
    assert set(req)=={'model','messages','temperature','max_tokens','seed','enable_thinking','n','stream'}
    assert req['model']==MODEL and req['seed']==42 and req['enable_thinking'] is False
    assert req['n']==1 and req['stream'] is False
    for m in req['messages']:
        assert set(m)=={'role','content'} and m['role'] in ('system','user','assistant')
        assert isinstance(m['content'],str)


def audit(method):
    root=OUTPUT/method
    p=read(root/'protocol.json')
    assert p['experiment_signature']==obj({k:v for k,v in p.items() if k!='experiment_signature'})
    for name,expected in p['input_sha256'].items():assert sha(Path(name))==expected,name
    vendor=read(ROOT/'scripts/vendor_memeye_text/UPSTREAM.json')
    for name,expected in vendor['files'].items():assert sha(ROOT/'scripts/vendor_memeye_text'/name)==expected
    captions={r['id']:r for r in rows(SOURCE/'captions.jsonl')}
    assert len(captions)==437 and all(c['model']==MODEL and c['seed']==42 for c in captions.values())
    assert p['caption_sha256']==sha(SOURCE/'captions.jsonl')
    qrows=rows(root/'questions.jsonl');gold_rows=rows(root/'gold.jsonl')
    source_q={q['id']:q for q in rows(SOURCE/'inputs/source_questions.jsonl')}
    assert gold_rows==rows(SOURCE/'inputs/evaluation.jsonl')
    assert obj(qrows)==p['questions_sha256'] and obj(gold_rows)==p['gold_sha256']
    questions={q['id']:q for q in qrows};gold={q['id']:q for q in gold_rows}
    assert len(questions)==len(qrows)==len(gold)==371 and set(questions)==set(source_q)
    for q in qrows:
        source=source_q[q['id']]
        assert set(q)=={'id','task','question','options','query_image_captions'}
        assert all(q[k]==source[k] for k in ('id','task','question','options'))
        assert q['query_image_captions']==[captions[i['sha256']]['caption'] for i in source['query_images']]
    obs=read(root/'observations.json')
    assert obj(obs)==p['observations_sha256']
    input_images=0
    for task,observations in obs.items():
        original=read(ROOT/'data/MemEye/data/dialog'/f'{task}_Open.json')
        turns=[(s,t) for s in original['multi_session_dialogues'] for t in s['dialogues']]
        assert len(turns)==len(observations)
        for o,(s,t) in zip(observations,turns):
            assert o['id']==t['round'] and o['timestamp']==str(s.get('date') or '').strip()
            assert o['user']==t.get('user','') and o['assistant']==t.get('assistant','')
            assert o['raw']['image_id']==t.get('image_id',[])
            refs=t.get('input_image') or []
            refs=[refs] if isinstance(refs,str) else refs
            assert o['raw']['input_image']==refs
            assert o['raw']['image_caption']==[captions[sha(ROOT/'data/MemEye/data/image'/x)]['caption'] for x in refs]
            input_images+=len(refs)
    assert sum(map(len,obs.values()))==848 and input_images==438
    amendment=root/('format_amendment.json' if method=='gen_agents' else 'execution_amendment.json')
    a=read(amendment)
    assert a['base_protocol_sha256']==sha(root/'protocol.json')
    adapter=ROOT/'scripts/evaluation'/('continue_deepseek_ga_format.py' if method=='gen_agents' else 'continue_deepseek_text_methods_parallel.py')
    assert a['script_sha256']==sha(adapter)
    scheduling=read(root/'streaming_amendment.json')
    assert scheduling['base_protocol_sha256']==sha(root/'protocol.json')
    assert scheduling['script_sha256']==sha(ROOT/'scripts/evaluation/continue_deepseek_text_methods_streaming.py')
    for name,expected in scheduling['imported_adapter_sha256'].items():
        assert sha(ROOT/'scripts/evaluation'/name)==expected
    extra_artifacts=[]
    if method=='memoryos':
        normalization=read(root/'memoryos_format_amendment.json')
        assert normalization['base_protocol_sha256']==sha(root/'protocol.json')
        assert normalization['script_sha256']==sha(ROOT/'scripts/evaluation/continue_deepseek_memoryos_format.py')
        extra_artifacts.append('memoryos_format_amendment.json')
        for event in rows(root/'summary_normalization_events.jsonl'):
            value=event['original'];fixed=event['normalized']
            if fixed:
                assert isinstance(value,dict) and len(fixed)==1
                assert fixed[0]['content']==value.get('content',value.get('summary'))
                assert fixed[0].get('keywords',[])==value.get('keywords',[])
            else: assert not (isinstance(value,list) and all(isinstance(v,dict) for v in value))
        extra_artifacts.append('summary_normalization_events.jsonl')
    raw=rows(root/'api_responses.jsonl')
    by_response={r['output']['response_id']:r for r in raw}
    assert len(by_response)==len(raw)
    for rec in raw:
        check_wire(rec['request']);out=rec['output']
        assert wire_sha(rec['request'])==out['request_sha256']
        assert out['seed']==42 and out['returned_model']==out['requested_model']==MODEL
    calls={f.stem:read(f) for f in (root/'calls').glob('*.json')}
    for key,c in calls.items():
        req=c['request'];out=c['output']
        assert key==obj(req) and req['experiment_signature']==p['experiment_signature'] and req['seed']==42
        wire=dict(model=MODEL,messages=req['messages'],temperature=req['temperature'],max_tokens=req['max_new_tokens'],seed=42,enable_thinking=False,n=1,stream=False)
        assert wire_sha(wire)==out['request_sha256']
        assert by_response[out['response_id']]['request']==wire
        assert by_response[out['response_id']]['output']['raw_response']==out['raw_response']
        if method=='gen_agents' and req['kind']=='memory':
            assert req['messages'][0]['role']=='system'
            assert req['messages'][0]['content'].startswith('Follow the requested output format exactly.')
    all_rows=[]
    for task in p['tasks']:
        target=root/task;done=read(target/'complete.json')
        assert done['experiment_signature']==p['experiment_signature'] and done['responses_sha256']==sha(target/'responses.jsonl')
        assert read(target/'memory_stats.json')['stored_rounds']==len(obs[task])
        all_rows.extend(rows(target/'responses.jsonl'))
    assert len(all_rows)==len({r['id'] for r in all_rows})==371
    assert {r['id'] for r in all_rows}==set(questions)
    counters=Counter();correct=0;invalid=[]
    for r in all_rows:
        q=questions[r['id']];g=gold[r['id']]
        assert all(r[k]==q[k] for k in q)
        assert r['retrieval_query']==query(q)
        assert r['gold_choice']==g['gold_choice'] and r['gold_ids']==g['gold_ids']
        messages=answer_messages(q,r['retrieved_context'],p)
        assert obj(messages)==r['reader_messages_sha256']
        saved=calls[r['call_key']]
        assert saved['request']['kind']=='answer' and saved['request']['messages']==messages
        assert saved['request']['temperature']==0 and saved['request']['max_new_tokens']==16
        assert saved['output']['raw_response']==r['raw_response']
        raw_answer=r['raw_response'].strip()
        prediction=raw_answer if raw_answer in ('A','B','C','D') and not r['hit_token_limit'] and not r.get('refusal') else None
        assert r['predicted_choice']==prediction and r['correct']==(prediction==g['gold_choice'])
        correct+=r['correct']
        if prediction is None:invalid.append(r['id'])
        if method in ('simplemem','a_mem'):
            d=r['retrieval_detail'];ids=d['context_memory_ids']
            assert len(ids)==len(set(ids))==d['context_memory_count']<=5
            counters[len(ids)]+=1
            if method=='simplemem':
                assert ids==list(dict.fromkeys(d['original_item_ids']))[:5]
                assert ids==[x['id'] for x in d['items']]
                assert len(re.findall(r'^\[Context \d+\]$',r['retrieved_context'],re.M))==len(ids)
            else:
                notes=read(root/r['task']/'final_memory.json')
                selected=[notes[i] for i in ids]
                expected=''.join('talk start time:'+n['timestamp']+'memory content: '+n['content']+'memory context: '+n['context']+'memory keywords: '+str(n['keywords'])+'memory tags: '+str(n['tags'])+'\n' for n in selected)
                assert expected==r['retrieved_context']
    summary=read(root/'summary.json')
    assert summary['status']=='complete' and summary['correct']==correct and summary['questions']==371
    assert summary['accuracy']==correct/371 and summary['invalid_answers']==len(invalid)
    assert summary['context_counts']=={str(k):v for k,v in counters.items()}
    assert rows(root/'qa_responses.jsonl')==all_rows
    result=dict(passed=True,method=method,questions=371,correct=correct,accuracy=correct/371,
        history_rounds_verified=848,image_memories_verified=438,caption_seed=42,all_wire_seeds=42,
        fresh_api_responses=len(raw),unique_cached_calls=len(calls),context_counts=dict(counters),
        invalid_answers=invalid,ga_format_adapter=method=='gen_agents',
        memory_invalid_structured_outputs=sum(c['output'].get('structured_output_valid') is False for c in calls.values()),
        memory_truncations=sum(c['output']['hit_token_limit'] for c in calls.values() if c['request']['kind']=='memory'),
        auditor_sha256=sha(Path(__file__)),artifacts_sha256={f:sha(root/f) for f in ['protocol.json','summary.json','qa_responses.jsonl','questions.jsonl','gold.jsonl','observations.json',amendment.name,'streaming_amendment.json',*extra_artifacts]})
    (root/'external_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS)
    args=parser.parse_args()
    result={method:audit(method) for method in ([args.method] if args.method else METHODS)}
    if not args.method:(OUTPUT/'external_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
