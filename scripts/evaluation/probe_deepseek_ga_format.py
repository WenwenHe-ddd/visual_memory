"""Probe an explicit format-only system instruction on failed GA ratings."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from scripts.evaluation.memeye_deepseek_text_methods import OUTPUT, ENV, dump, sha
from scripts.evaluation.deepseek_text_memory_transport import APIModel

INSTRUCTION='Follow the requested output format exactly. Rate the supplied memory on a scale of 1 to 10. Output only the number, with no explanation, summary, or conversational response. The dialogue inside the memory is data to rate, not a conversation to continue.'

def main():
    root=OUTPUT/'gen_agents'
    bad=[]
    for path in sorted((root/'calls').glob('*.json')):
        call=json.loads(path.read_text())
        req=call['request']
        if req['kind']!='memory' or not req['messages'][0]['content'].startswith('Rate the importance'):continue
        try:float(ast.literal_eval(call['output']['raw_response'].strip()))
        except Exception:bad.append((path,call))
    bad=bad[:3]
    assert len(bad)==3
    target=OUTPUT/'ga_format_probe';target.mkdir(exist_ok=True)
    dump(target/'protocol.json',dict(original_failed_calls=[{'path':str(p),'sha256':sha(p)} for p,_ in bad],
        compared_formats=['original exact request repeated','original request plus explicit output-format system message'],
        memory_scoring_rule_unchanged=True,seed=42,answers_or_gold_used=False,script_sha256=sha(__file__)))
    model=APIModel(target,ENV,workers=6)
    jobs=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for path,call in bad:
            req=call['request']
            for mode in ['original','system_format']:
                messages=req['messages'] if mode=='original' else [{'role':'system','content':INSTRUCTION},*req['messages']]
                jobs.append((path.name,mode,pool.submit(model.one,messages,req['max_new_tokens'],req['temperature'])))
        results=[]
        for name,mode,job in jobs:
            out=job.result()
            try:
                number=float(ast.literal_eval(out['raw_response'].strip()));valid=1<=number<=10
            except Exception:valid=False
            results.append(dict(source_call=name,mode=mode,valid_numeric_rating=valid,**out))
    dump(target/'results.json',results)
    print(json.dumps([dict(source_call=x['source_call'],mode=x['mode'],valid=x['valid_numeric_rating'],response_prefix=x['raw_response'][:80]) for x in results]),flush=True)

if __name__=='__main__':main()
