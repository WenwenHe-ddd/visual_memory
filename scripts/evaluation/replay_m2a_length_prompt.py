"""Replay three saved token-limit failures with only the new length instruction.

No tool is executed and no memory database is changed. Image bytes, tool schemas,
messages, model and generation caps come from the saved requests.
"""
import argparse
import base64
import copy
import hashlib
import json
import mimetypes
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_m2a as base
from scripts.memory.m2a_length_prompt import LengthLimitedQwen, POLICY
from langchain_core.messages import messages_from_dict

CASES = [
    'memeye/Outdoor_Navigation_Route_Memory_Assistant',
    'memeye/Personal_Health_Dashboard_Assistant',
    'memgallery/Fashion_Personal_Care_Lifestyle_Shopping',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def restore(record, source):
    saved = copy.deepcopy(record['messages'])
    observations = json.loads((source/'observations.json').read_text())
    images_by_hash = {o['image_sha256']:o['image'] for o in observations if o['image']}
    images = iter(record['images'])
    used = 0
    for message in saved:
        if not isinstance(message['content'], list):
            continue
        for block in message['content']:
            if block['type'] not in ('image','image_url'):
                continue
            item = next(images)
            used += 1
            path = Path(item.get('path') or images_by_hash[item['bytes_sha256']])
            assert sha(path) == item['bytes_sha256']
            key = 'url' if 'url' in block else 'image_url'
            value = block[key]
            nested = isinstance(value, dict) and 'url' in value
            ref = value['url'] if nested else value
            if isinstance(ref, dict):
                mime = mimetypes.guess_type(str(path))[0] or 'image/jpeg'
                uri = 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()
                assert hashlib.sha256(uri.encode()).hexdigest() == ref['data_uri_sha256']
                assert len(uri) == ref['length']
                if nested:
                    block[key]['url'] = uri
                else:
                    block[key] = uri
            else:
                assert Path(ref) == path
    assert used == len(record['images'])
    assert base.safe_content(saved) == record['messages']
    messages = messages_from_dict([{'type':m['type'], 'data':m} for m in saved])
    assert base.safe_content([m.model_dump() for m in messages]) == record['messages']
    return messages


def run(source_root, out, device):
    out.mkdir(parents=True, exist_ok=False)
    selected = []
    for name in CASES:
        source = source_root/name/'attempt_001'
        with (source/'llm_calls.jsonl').open() as f:
            record = next(r for line in f if (r := json.loads(line)).get('hit_token_limit'))
        messages = restore(record, source)
        protocol = json.loads((source/'protocol.json').read_text())
        selected.append((source, record, messages, protocol))
    first = selected[0][3]
    provenance = dict(policy=POLICY, model=first['llm_path'], seed=42, do_sample=False,
        cases=[dict(source=str(s), call=r['call'], stage=r['stage'], sample_id=r['sample_id'],
                    original_input_sha256=base.digest(r), original_output_tokens=r['output_tokens'])
               for s,r,m,p in selected],
        implementation_sha256={str(p):sha(p) for p in [Path(__file__),
            ROOT/'scripts/memory/m2a_length_prompt.py', ROOT/'scripts/memory/m2a_local.py']})
    base.write(out/'protocol.json', provenance)
    llm = LengthLimitedQwen(first['llm_path'], out, device,
        image_max_pixels=first['image_max_pixels'])
    results = []
    for source, record, messages, protocol in selected:
        assert protocol['llm_path'] == first['llm_path']
        assert protocol['image_max_pixels'] == first['image_max_pixels']
        call = copy.copy(llm)
        call.stage, call.sample_id = record['stage'], record['sample_id']
        call.schemas = record['tools']
        call.max_new_tokens = 16 if call.stage == 'reader' else protocol['max_new_tokens']
        result = dict(source=str(source), source_call=record['call'], sample_id=call.sample_id,
                      max_new_tokens=call.max_new_tokens, original_output_tokens=record['output_tokens'])
        try:
            answer = call.invoke(messages)
            result.update(status='ok', output=answer.content, tool_calls=answer.tool_calls)
        except Exception as exc:
            result.update(status='error', error=repr(exc))
        latest = base.rows(out/'llm_calls.jsonl')[-1]
        result.update(output_tokens=latest['output_tokens'], hit_token_limit=latest['hit_token_limit'],
            elapsed_seconds=latest['elapsed_seconds'],
            word_count=len(latest['raw_response'].split()),
            original_messages_restored_exactly=True,
            mcq_format_valid=(latest['raw_response'] in 'ABCD' and len(latest['raw_response']) == 1)
                if call.stage == 'reader' else None)
        base.append_json(out/'replay_results.jsonl', result)
        results.append(result)
    summary = dict(cases=len(results), calls_completed=sum(r['status']=='ok' for r in results),
        token_limit_hits=sum(r['hit_token_limit'] for r in results),
        mcq_format_errors=sum(r['mcq_format_valid'] is False for r in results),
        results=results, scope='Saved-request replay only; no tool execution or benchmark rerun')
    base.write(out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'results/m2a_qwen_full_20261112')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.device)
