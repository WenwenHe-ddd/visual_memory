"""Independent completed-run audit: original requests plus seed, IDs and scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation.openai_backbone_transport import build_request

SOURCE=ROOT/'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
OUTPUT=ROOT/'results/memeye_gpt54_seed42_baselines_20261024'
MODEL='gpt-5.4-2026-03-05'


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,
                                     separators=(',',':'),allow_nan=False).encode()).hexdigest()


def index(rows,key):
    result={r[key]:r for r in rows}
    assert len(result)==len(rows)
    return result


def choice(text):
    m=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',text.strip(),re.I)
    return m.group(1).upper() if m else None


def main():
    protocol=read(OUTPUT/'protocol.json');c=protocol['contract']
    assert c['model']==MODEL and c['caption_model']==MODEL and c['seed']==42
    assert digest(c)==protocol['experiment_signature']
    for name,expected in c['source_sha256'].items():
        assert sha(Path(name))==expected,name
    historical=index(lines(SOURCE/'reader/requests.jsonl'),'record_id')
    records=index(lines(OUTPUT/'inputs/requests.jsonl'),'record_id')
    manifests=index(lines(SOURCE/'reader_manifest.jsonl'),'record_id')
    wires=index(lines(OUTPUT/'api_request_manifest.jsonl'),'record_id')
    responses=index(lines(OUTPUT/'responses.jsonl'),'record_id')
    fresh=index(lines(OUTPUT/'fresh_responses.jsonl'),'payload_sha256')
    gold=index(lines(SOURCE/'inputs/evaluation.jsonl'),'id')
    assert set(historical)==set(records)==set(responses)==set(wires)==set(manifests)
    assert len(records)==742 and len(fresh)==720 and len(gold)==371
    assert digest(list(wires.values()))==c['request_manifest_sha256']
    assert lines(OUTPUT/'inputs/gold.jsonl')==list(gold.values())
    counts=Counter();domains={};errors=empty=length=reasoning=0
    pairs=index(read(OUTPUT/'paired_results.json'),'question_id')
    for rid,r in records.items():
        h=historical[rid]
        body=build_request({k:h[k] for k in ['system','user_text','query_images']},
                           model=MODEL,max_output_tokens=16,image_detail='high')
        assert digest(body)==h['request_sha256']==r['historical_request_sha256']
        body['seed']=42
        wire_hash=digest(body)
        assert wire_hash==wires[rid]['request_sha256']
        assert set(body)=={'model','messages','seed','temperature','reasoning_effort',
                          'max_completion_tokens','n','stream','store'}
        assert c['reader_system_prompt']==h['system']
        assert r['reader_text']==h['user_text'] and r['query_images']==h['query_images']
        assert r['top_memory_ids']==manifests[rid]['top_memory_ids']
        assert r['payload_sha256']==h['payload_sha256']
        a=responses[rid];f=fresh[r['payload_sha256']]
        assert a['request_sha256']==wire_hash and a['returned_model']==MODEL
        assert a['run_id']==protocol['run_id'] and a['experiment_signature']==protocol['experiment_signature']
        assert a['historical_response_reuse'] is False
        assert a['top_memory_ids']==r['top_memory_ids']
        assert a['fresh_inference_executed_for_this_record']==(rid==f['record_id'])
        assert a['same_run_shared_inference_record_id']==(None if rid==f['record_id'] else f['record_id'])
        for key in ['raw_response','response_id','request_sha256','request_id','usage','execution_id']:
            assert a[key]==f[key]
        pred=choice(a['raw_response']);assert pred==a['predicted_choice']
        qid,condition=rid.rsplit('::',1)
        ok=pred==gold[qid]['gold_choice']
        counts[condition]+=ok
        domains.setdefault(r['task'],Counter())[condition]+=ok
        prefix='baseline' if condition=='baseline_question_only' else 'expanded'
        assert pairs[qid][prefix+'_answer']==pred and pairs[qid][prefix+'_correct']==ok
        errors+=pred is None;empty+=not a['raw_response'];length+=a['finish_reason']=='length'
        reasoning+=a.get('reasoning_tokens') or 0
    fresh_rows=list(fresh.values())
    new_ids={r['response_id'] for r in fresh_rows}
    old_ids={r['response_id'] for r in lines(SOURCE/'reader/fresh_responses.jsonl')}
    probe=read(OUTPUT/'seed_support_probe.json')
    probe_ids={r['response']['id'] for r in probe['attempts']}
    assert len(new_ids)==720 and len({r['request_id'] for r in fresh_rows})==720
    assert not new_ids & (old_ids|probe_ids)
    attempts=lines(OUTPUT/'api_attempts.jsonl')
    assert {r['response_id'] for r in attempts if r['status']=='success'}==new_ids
    summary=read(OUTPUT/'summary.json');v=summary['validation']
    assert summary['status']=='complete' and summary['seed']==42
    assert summary['overall']['questions']==371
    assert summary['overall']['baseline_correct']==counts['baseline_question_only']
    assert summary['overall']['expanded_correct']==counts['master_answer_query']
    assert (v['format_errors'],v['empty_responses'],v['token_limit_records'],v['reasoning_tokens_reported'])==(errors,empty,length,reasoning)
    for name,s in summary['method_annotations'].items():
        assert len(s)==5 and s['5. 回答方式']['seed']==42
        assert s['5. 回答方式']['correct']==counts[name]
        assert s['3. 检索方式']['zscore'] is False
    proof=read(OUTPUT/'source_validation.json')
    assert proof['passed'] and proof['unweighted_correct']=={'baseline_question_only':236,'master_answer_query':241}
    result={'passed':True,'model':MODEL,'seed':42,'questions':371,'records':742,
        'fresh_requests':720,'same_run_shared_records':22,'native_image_records':sum(bool(r['query_images']) for r in records.values()),
        'api_requests_independently_reconstructed':742,'only_wire_change_from_historical':'seed=42',
        'all_frozen_sources_verified':True,'caption_api_provenance_verified':proof['caption_unique_images'],
        'probe_and_historical_response_ids_excluded':True,'format_errors':errors,'empty_responses':empty,
        'token_limit_records':length,'reasoning_tokens_logical_records':reasoning,
        'system_fingerprints':dict(Counter(str(r['system_fingerprint']) for r in fresh_rows)),
        'api_attempts':len(attempts),'transient_failed_attempts':sum(r['status']!='success' for r in attempts),
        'correct':dict(counts),'seed_determinism_not_guaranteed':True,
        'artifact_sha256':{n:sha(OUTPUT/n) for n in ['protocol.json','summary.json','responses.jsonl',
            'fresh_responses.jsonl','paired_results.json','paired_results.csv','api_request_manifest.jsonl',
            'api_attempts.jsonl','seed_support_probe.json','report.html']}}
    (OUTPUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
