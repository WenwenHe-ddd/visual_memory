"""Replay the two observed extraction failures using the corrected local model."""
import json
from scripts.evaluation import mem_gallery_qwen_omni_v3 as run


def main():
    output = run.OUTPUT
    protocol = run.base.verify(output)
    model = run.LocalMultimodalQwen('cuda:0')
    broker = run.OmniBroker(model, output / 'memory', protocol)
    records = []
    cases = [
        run.PREVIOUS / 'memory/calls/3729953971fcf104b85c2d4bdf89a3fff5442c011e526dfb4a79248a1aaa8b6d.json',
        run.ROOT / 'results/mem_gallery_qwen_omni_simplemem030_20261104/memory/calls/c6cf28720adea0676308aeafdb83f3cb1aabe628b1ac82dacfa540f63ce2a513.json',
    ]
    try:
        for case in cases:
            old = run.base.read(case)
            messages = [m for m in old['request']['messages'] if m['role'] != 'system'
                        or 'expert knowledge extraction system' in m['content']]
            assert sum(m['role'] == 'system' for m in messages) == 1
            broker.local.owner = 'preflight:' + case.stem
            response = broker.client().create(messages=messages, model='local-qwen',
                response_format={'type': 'json_object'}, temperature=0)
            value = json.loads(response.choices[0].message.content)
            broker.check()
            records.append(dict(original_call=str(case), original_sha256=run.base.sha(case),
                                entities=len(value['entities']), relations=len(value['relations']), status='passed'))
            print(json.dumps(records[-1]), flush=True)
    finally:
        broker.close()
    run.base.dump(output / 'preflight/failed_cases_replayed.json', dict(status='passed', cases=records,
        validator_source_sha256=run.base.sha(__file__), environment=model.environment))


if __name__ == '__main__':
    main()
