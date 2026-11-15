"""Check seed support on frozen GPT-5.4 MemEye reader requests, without labels."""
from pathlib import Path
import json
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.openai_backbone_transport import build_request, load_config, request_sha256
from scripts.evaluation.openai_chat_transport import _NoRedirect

SOURCE = ROOT / 'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
OUTPUT = ROOT / 'results/memeye_gpt54_seed42_baselines_20261024'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / 'seed_support_probe.json'
    if result_path.exists():
        raise RuntimeError('Existing probe retained; inspect it before another API call')
    contract = json.loads((SOURCE / 'protocol.json').read_text())['contract']
    config = load_config(contract['api_env_file'], model=contract['model'])
    rows = [json.loads(line) for line in (SOURCE / 'reader/requests.jsonl').read_text().splitlines()]
    candidates = [next(r for r in rows if r['record_id'] == 'Personal_Health_Dashboard_Assistant/Q38::master_answer_query'),
                  next(r for r in rows if r['query_images'])]
    result = {'source': str(SOURCE), 'requested_model': config.model, 'seed': 42,
              'scope': 'Seed support and two-repeat smoke check only; excluded from scored full rerun',
              'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'status': 'running', 'attempts': []}
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    for row in candidates:
        body = build_request({k:row[k] for k in ('system','user_text','query_images')},
                             model=config.model, max_output_tokens=16, image_detail='high')
        assert request_sha256(body) == row['request_sha256'], 'Historical request has changed'
        body['seed'] = 42
        for repetition in range(2):
            meta = {'record_id': row['record_id'], 'repetition': repetition + 1,
                    'query_image_count': len(row['query_images']),
                    'historical_request_sha256': row['request_sha256'],
                    'request_sha256': request_sha256(body), 'request_keys': sorted(body),
                    'temperature': body['temperature'], 'reasoning_effort': body['reasoning_effort']}
            request = urllib.request.Request(config.base_url+'/chat/completions',
                data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
                headers={'Authorization':'Bearer '+config.api_key, 'Content-Type':'application/json'})
            try:
                with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
                    data = json.loads(response.read())
                    meta.update(http_status=response.status, request_id=response.headers.get('x-request-id'),
                                response=data, status='success')
            except urllib.error.HTTPError as error:
                try:
                    data = json.loads(error.read(65536))
                finally:
                    error.close()
                detail = data.get('error', {})
                # Only structured provider error fields, never headers or credentials.
                safe = {k:detail.get(k) for k in ('type','code','param','message')}
                meta.update(http_status=error.code, status='api_error', error=safe)
                result['status'] = ('seed_rejected' if safe.get('param') == 'seed'
                                    or 'seed' in str(safe.get('message')) else 'api_error')
            except (OSError, ValueError):
                meta.update(status='transport_or_protocol_error')
                result['status'] = 'transport_or_protocol_error'
            result['attempts'].append(meta)
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({k:v for k,v in meta.items() if k != 'response'}, ensure_ascii=False),flush=True)
            if meta['status'] != 'success':
                return
            assert data['model'] == config.model
    result['status'] = 'seed_accepted'
    result['repeat_pairs'] = [{'record_id':a['record_id'],
        'identical_request':a['request_sha256']==b['request_sha256'],
        'same_answer':a['response']['choices'][0]['message']['content']==b['response']['choices'][0]['message']['content']}
        for a,b in zip(result['attempts'][::2],result['attempts'][1::2])]
    result['caveat'] = 'Acceptance does not establish that the seed is effective or guarantee determinism.'
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'repeat_pairs':result['repeat_pairs']}),flush=True)


if __name__ == '__main__':
    main()
