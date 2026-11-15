"""Versioned Omni recovery entrypoint; frozen v2 and text code are unchanged."""
import argparse
import copy
from pathlib import Path
import shutil

from scripts.evaluation import mem_gallery_qwen_omni as base
from scripts.evaluation.omni_qwen_transport_v3 import LocalMultimodalQwen, OmniBroker
from scripts.analysis.audit_omni_v3_transport import audit as audit_transport

ROOT = base.ROOT
PREVIOUS = ROOT / 'results/mem_gallery_qwen_omni_simplemem030_v2_20261104'
OUTPUT = ROOT / 'results/mem_gallery_qwen_omni_simplemem030_v3_20261104'


def install():
    # Process-local injection into the frozen runner, never edits shared text code.
    base.LocalMultimodalQwen = LocalMultimodalQwen
    base.OmniBroker = OmniBroker


def prepare(output, pilot=False):
    if (output / 'protocol.json').exists():
        return base.verify(output)
    original_freeze = base.freeze

    def seal(path, value):
        if path == output / 'protocol.json':
            value['schema'] = 'omni_simplemem030_qwen_visual_v3'
            value['system_message_policy'] = 'Losslessly merge leading system messages before Qwen template; native prompts, schema, and retry instructions must all render.'
            value['json_policy'] = 'Require entity/relation object shapes and typed fields; arbitrary string entity types and ignored extra keys match native parser. Never invent or repair entity facts.'
            value['recovery_parent'] = str(PREVIOUS)
            value['cache_policy'] = 'Reuse only complete non-schema single-system memory calls with identical generation parameters and image bytes; all entity extraction is regenerated. Original raw records are preserved.'
            value['experiment_signature'] = base.obj({k: v for k, v in value.items() if k != 'experiment_signature'})
        original_freeze(path, value)

    base.freeze = seal
    try:
        return base.prepare(output, pilot)
    finally:
        base.freeze = original_freeze


def migrate_cache(output):
    """Copy only behavior-identical calls, with auditable old/new provenance."""
    manifest_path = output / 'cache_migration.json'
    if manifest_path.exists():
        for row in base.read(manifest_path)['calls']:
            assert base.sha(row['old_path']) == row['old_sha256']
            assert base.sha(row['new_path']) == row['new_sha256']
        return
    p = base.verify(output)
    prior = base.verify(PREVIOUS)
    assert prior['model'] == p['model'] and prior['seed'] == p['seed']
    rows = []
    for path in sorted((PREVIOUS / 'memory/calls').glob('*.json')):
        record = base.read(path)
        old_request, result = record['request'], record['output']
        assert base.obj(old_request) == path.stem
        assert old_request['experiment_signature'] == prior['experiment_signature']
        if (old_request['response_format'] or old_request['kind'] != 'memory'
                or sum(m['role'] == 'system' for m in old_request['messages']) > 1
                or result['hit_token_limit'] or not result['raw_response'].strip()):
            continue
        request = copy.deepcopy(old_request)
        request['experiment_signature'] = p['experiment_signature']
        for message in request['messages']:
            for part in message['content'] if isinstance(message['content'], list) else []:
                if part['type'] != 'image':
                    continue
                source = Path(part['path'])
                assert source.is_relative_to(PREVIOUS / 'memory/transport_images')
                assert base.sha(source) == part['sha256']
                target = output / source.relative_to(PREVIOUS)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert base.sha(target) == part['sha256']
                else:
                    shutil.copyfile(source, target)
                part['path'] = str(target)
        target = output / 'memory/calls' / (base.obj(request) + '.json')
        base.freeze(target, dict(request=request, output=result))
        rows.append(dict(old_path=str(path), old_sha256=base.sha(path),
                         new_path=str(target), new_sha256=base.sha(target)))
    base.freeze(manifest_path, dict(status='verified', parent_protocol_sha256=base.sha(PREVIOUS / 'protocol.json'),
        reused_calls=len(rows), excluded_schema_calls=True, calls=rows))
    print('Verified compatible cache calls:', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'analyze', 'all'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    install()
    if args.stage in ('prepare', 'all'):
        prepare(args.output, args.pilot)
        if not args.pilot:
            migrate_cache(args.output)
    if args.stage in ('run', 'all'):
        base.run(args.output, args.device)
    if args.stage in ('analyze', 'all'):
        from scripts.analysis.audit_mem_gallery_qwen_omni import analyze
        audit_transport(args.output)
        analyze(args.output)


if __name__ == '__main__':
    main()
