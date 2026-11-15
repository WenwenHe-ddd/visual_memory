"""Validate sealed HippoMM Open answers against every native wire/attempt/output."""
from pathlib import Path
import json, hashlib, gzip, sys
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
from scripts.evaluation.mem_gallery_master_reader import READER_SYSTEM, digest
from scripts.memory import hippomm_core as core

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def read(path): return json.loads(Path(path).read_text())
def main():
    root=ROOT/'results/memeye_deepseek_open_20261106/visual_methods/hippomm'
    seal=read(root/'completion.json'); validation=read(root/'evaluation_validation.json'); protocol=read(root/'protocol.json')
    assert seal['status']=='complete' and seal['stage']=='evaluation_records' and seal['answered']==370 and seal['provider_rejections']==1
    assert sha(root/seal['responses_file'])==seal['responses_sha256']
    assert sha(root/'responses.jsonl')==seal['native_success_responses_sha256']
    assert sha(root/'evaluation_validation.json')==seal['validation_sha256']
    assert validation['successful_native_full_replay'] is True and validation['failed_native_prefix_replay'] is True and validation['questions']==371
    assert validation['successful_records']==370 and validation['provider_rejections']==1
    for path, expected in validation['source_sha256'].items(): assert sha(path)==expected
    canonical={r['id']:r for r in rows(ROOT/'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl')}
    values=rows(root/seal['responses_file']); assert len(values)==len({r['id'] for r in values})==371
    originals={r['id']:r for r in rows(root/'responses.jsonl')}; assert len(originals)==370
    assert (root/seal['responses_file']).read_bytes().startswith((root/'responses.jsonl').read_bytes())
    memory=Path(protocol['memory_dir']); events=rows(memory/'events.jsonl')
    for name, expected in protocol['source_sha256'].items(): assert sha(name)==expected,name
    calls=set(); roles=Counter(); routes=Counter(); attempts_manifest={}; wire_manifest={}; finish=Counter(); errors=Counter(); failed_wires={}; failed_ids=[]
    for row in values:
        q=canonical[row['id']]
        assert all(row[k]==q[k] for k in ('id','task','point','question'))
        assert row['model']==row['answer_model']==transport.DEEPSEEK_MODEL and row['backbone']=='deepseek'
        assert row['source_mcq_correct']==207 and not row['hit_token_limit']
        routes[row['route']]+=1
        assert row['route'] in ('fast','detailed')
        if row['route']=='detailed': assert len(row['selected_event_ids'])<=5
        if row.get('status')=='permanent_http_error':
            failed_ids.append(row['id']); failed=row['failed_call']
            assert row['raw_response']==row['answer']=='' and row['returned_model'] is None
            assert not set(row)&{'response_id','usage','input_tokens','generated_tokens'}
            assert row['llm_calls'][-1]==failed['path'] and not (root/failed['path']).exists()
            req=failed['request']; assert req['role']=='answer_with_representatives'
            images_in=[{'path':p,'sha256':sha(p)} for p in req['image_paths']]
            body=transport.build_request({'system':READER_SYSTEM,'user_text':req['prompt'],'query_images':images_in},model=transport.DEEPSEEK_MODEL,max_output_tokens=2048,image_detail='high')
            wire_path=root/failed['wire_file']
            assert json.loads(gzip.decompress(wire_path.read_bytes()))==body
            assert sha(wire_path)==failed['wire_file_sha256']
            assert digest(body)==failed['request_sha256']==row['payload_sha256']==row['request_sha256']
            rejected=[a for a in rows(root/failed['attempts_file']) if a['call_id']==failed['call_id']]
            assert len(rejected)==1 and rejected[0]['attempt']==1 and rejected[0]['status']=='error'
            assert rejected[0]['kind']=='http' and rejected[0]['http_status']==400 and rejected[0]['retryable'] is False
            assert rejected[0]['request_sha256']==failed['request_sha256']
            assert failed['error']=={'kind':'http','http_status':400,'retryable':False}
            record=read(root/row['failed_call_record_file'])
            assert record['status']=='permanent_http_error' and record['response'] is None and record['returned_model'] is None
            assert record['request']==req and record['source_attempt']==rejected[0]
            assert not set(record)&{'response_id','usage'}
            failed_wires[str(wire_path)]=sha(wire_path)
            attempts_manifest[str(root/failed['attempts_file'])]=sha(root/failed['attempts_file'])
            call_names=row['llm_calls'][:-1]
            final=None;raw=''
        else:
            assert row==originals[row['id']]
            final=read(root/row['llm_calls'][-1]); raw=final['response']['raw_response']
            call_names=row['llm_calls']

        if row['route']=='fast':
            local=[e['event_id'] for e in events if e['task']==q['task']]
            parsed=core.parse_fast(raw,local)
            assert parsed['confidence']>.75 and parsed==row['fast']
            expected_answer=parsed['answer'].strip()
        else: expected_answer=raw.strip()
        assert row['raw_response']==row['answer']==expected_answer
        if final is not None: assert row['payload_sha256']==row['request_sha256']==final['request_sha256']
        for name in call_names:
            if name in calls: continue
            calls.add(name); path=root/name; call=read(path); req=call['request']; response=call['response']
            assert req['backbone']=='deepseek' and req['max_output_tokens']==2048
            assert set(req)=={'role','prompt','image_paths','backbone','max_output_tokens'}
            roles[req['role']]+=1
            images=[{'path':p,'sha256':sha(p)} for p in req['image_paths']]
            assert {r['path']:r['sha256'] for r in images}==call['image_sha256']
            wire=transport.build_request({'system':READER_SYSTEM,'user_text':req['prompt'],'query_images':images},model=transport.DEEPSEEK_MODEL,max_output_tokens=2048,image_detail='high')
            wire_path=path.with_suffix('.wire.json.gz')
            assert json.loads(gzip.decompress(wire_path.read_bytes()))==wire
            assert sha(wire_path)==call['wire_file_sha256'] and digest(wire)==call['request_sha256']
            assert response['returned_model']==response['requested_model']==transport.DEEPSEEK_MODEL
            assert response['seed']==42 and response['enable_thinking'] is False
            assert not response.get('refusal') and response['raw_response'].strip()
            assert response['finish_reason']=='stop'; finish[response['finish_reason']]+=1
            attempt_path=path.parent/'api_attempts.jsonl'
            attempts=[a for a in rows(attempt_path) if a['call_id']==call['call_id']]
            assert [a['attempt'] for a in attempts]==list(range(1,len(attempts)+1)) and 1<=len(attempts)<=4
            successful=[a for a in attempts if a['status']=='success']
            assert len(successful)==1 and attempts[-1]['status']=='success'
            assert successful[0]['response']==response
            assert all(a['request_sha256']==call['request_sha256'] for a in attempts)
            for attempt in attempts:
                if attempt['status']=='error':
                    assert attempt['retryable']; errors[attempt['kind']]+=1
            attempts_manifest[str(attempt_path)]=sha(attempt_path)
            wire_manifest[str(path)]=sha(path)
    assert len(calls)==validation['successful_native_calls']
    assert failed_ids==validation['failed_ids']==['Cartoon_Entertainment_Companion/COMIC_Q48']
    assert dict(roles)==validation['roles']
    result={'status':'complete','method_id':'hippomm','questions':371,'source_mcq_correct':207,
      'answer_model':transport.DEEPSEEK_MODEL,'unique_native_calls':len(calls),'successful_records':370,'provider_rejections':1,'roles':dict(roles),'routes':dict(routes),
      'native_source_full_replay_verified':True,'successful_native_full_replay':True,'failed_native_prefix_replay':True,'original_bytes_and_complete_wire_verified':True,
      'all_successful_attempts_match_raw_outputs':True,'native_output_unchanged':True,'gold_accessed':False,
      'transient_errors':dict(errors),'finish_reasons':dict(finish),
      'responses_sha256':sha(root/seal['responses_file']),'source_responses_file':seal['responses_file'],'native_success_responses_sha256':sha(root/'responses.jsonl'),'completion_sha256':sha(root/'completion.json'),
      'protocol_sha256':sha(root/'protocol.json'),'attempt_files_sha256':attempts_manifest,
      'call_files_sha256':wire_manifest,'failed_wire_files_sha256':failed_wires,'failed_ids':failed_ids,'evaluation_validation_sha256':sha(root/'evaluation_validation.json'),'audit_source_sha256':sha(__file__)}
    out=root/'independent_output_audit.json'; text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if out.exists(): assert out.read_text()==text
    out.write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ('call_files_sha256','attempt_files_sha256')}))
if __name__=='__main__':main()
