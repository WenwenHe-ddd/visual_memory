"""Verify actual rendered prompts and every imported cache record before scoring."""
import hashlib
from pathlib import Path


def audit(output):
    from scripts.evaluation import mem_gallery_qwen_omni as base
    from scripts.evaluation.omni_qwen_transport_v3 import MergedSystemProcessor
    from transformers import AutoProcessor
    p = base.verify(output)
    processor = AutoProcessor.from_pretrained(str(base.base.gallery.MODEL), local_files_only=True)
    merged = MergedSystemProcessor(processor)
    migration_path = output / 'cache_migration.json'
    migration = base.read(migration_path) if migration_path.exists() else dict(calls=[])
    imported = {}
    for item in migration['calls']:
        assert base.sha(item['old_path']) == item['old_sha256']
        assert base.sha(item['new_path']) == item['new_sha256']
        old, new = base.read(item['old_path']), base.read(item['new_path'])
        assert old['output'] == new['output']
        a, b = old['request'], new['request']
        assert a['response_format'] is None and a['kind'] == 'memory'
        assert not old['output']['hit_token_limit'] and old['output']['raw_response'].strip()
        assert {k: v for k, v in a.items() if k not in ('messages', 'experiment_signature')} == {
            k: v for k, v in b.items() if k not in ('messages', 'experiment_signature')}
        assert len(a['messages']) == len(b['messages'])
        for left, right in zip(a['messages'], b['messages']):
            assert left['role'] == right['role']
            if isinstance(left['content'], str):
                assert left == right
            else:
                assert len(left['content']) == len(right['content'])
                for x, y in zip(left['content'], right['content']):
                    if x['type'] == 'image':
                        assert {k: v for k, v in x.items() if k != 'path'} == {k: v for k, v in y.items() if k != 'path'}
                        assert base.sha(x['path']) == base.sha(y['path']) == x['sha256']
                    else:
                        assert x == y
        before = processor.apply_chat_template(a['messages'], tokenize=False, add_generation_prompt=True)
        after = merged.apply_chat_template(b['messages'], tokenize=False, add_generation_prompt=True)
        assert before == after, item
        imported[Path(item['new_path']).stem] = item
    calls = list((output / 'memory/calls').glob('*.json'))
    rendered_count = 0
    for path in calls:
        record = base.read(path)
        request = record['request']
        assert base.obj(request) == path.stem
        assert request['experiment_signature'] == p['experiment_signature']
        if path.stem in imported:
            continue
        rendered = merged.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True)
        assert hashlib.sha256(rendered.encode()).hexdigest() == record['output']['rendered_prompt_sha256']
        for message in request['messages']:
            if message['role'] == 'system':
                assert message['content'] in rendered
        rendered_count += 1
    base.dump(output / 'transport_audit.json', dict(status='passed',
        imported_calls_verified=len(imported), rendered_prompts_verified=rendered_count,
        all_system_instructions_visible=True, old_raw_outputs_preserved=True))
