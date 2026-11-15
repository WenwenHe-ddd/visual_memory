"""Gallery-specific metadata adaptation; preserve dates and official image IDs.

The pinned MemEye GA/MemoryOS input wrappers omit fields required by Gallery's
visual-search and temporal tasks. Restore original metadata before ingestion;
leave every native storage/retrieval operation and LLM setting unchanged.
"""
import argparse
import json
from pathlib import Path
import threading

from scripts.evaluation import mem_gallery_qwen_text_methods as base
from scripts.evaluation.continue_mem_gallery_ga_format import reflection_lines

OUTPUT = base.ROOT / 'results/mem_gallery_qwen_seed42_text_methods_20261026_v2'
ORIGINAL_HELPERS = base.native.helpers
ORIGINAL_INSTALL = base.native.install_runtime


def image_lines(row):
    raw = row['raw']
    ids = raw.get('image_id') or []
    if isinstance(ids, str):
        ids = [ids]
    caps = raw.get('image_caption') or []
    assert len(ids) == len(caps)
    result = []
    for image_id, caption in zip(ids, caps):
        result.extend(['image:', 'image_id: ' + image_id, 'image_caption: ' + caption])
    return result


def metadata_lines(row):
    return ['Date: ' + row['timestamp'], 'Round: ' + row['id']]


def gallery_helpers(method):
    helper = ORIGINAL_HELPERS(method)
    if method == 'gen_agents':
        def text(row, speaker_a, speaker_b):
            return '\n'.join(metadata_lines(row) + [helper['_dialogue_text'](row, speaker_a, speaker_b)] + image_lines(row))
        helper['_build_round_text'] = text
    elif method == 'memoryos':
        def text(row):
            return '\n'.join(metadata_lines(row) + [str(row['user']).strip()] + image_lines(row))
        helper['_round_user_with_captions'] = text
    return helper


def prepare():
    if (OUTPUT / 'metadata_adaptation.json').exists():
        for method in base.METHODS:
            base.verify(OUTPUT, method)
        return
    base.prepare(OUTPUT)
    dependencies = [Path(__file__).resolve(), Path(base.ROOT / 'scripts/evaluation/continue_mem_gallery_ga_format.py')]
    for method in base.METHODS:
        path = OUTPUT / method / 'protocol.json'
        p = base.read(path)
        del p['experiment_signature']
        p['study'] = 'Mem-Gallery / metadata-preserving native text memories / matched local Qwen reader'
        p['input_sha256'].update({str(f): base.sha(f) for f in dependencies})
        p['annotations']['1_candidate_memory_structure'] += ' GA/MemoryOS explicitly preserve original date, round and image ID in ingested text; SimpleMem/A-Mem native fields already retain image IDs and timestamps.'
        p['gallery_metadata_adaptation'] = 'Restore original date, round and image_id in GA/MemoryOS input wrappers; no inferred metadata, no changes to dialogue/caption contents. Native storage/retrieval unchanged.'
        p['reflection_format_policy'] = 'GA tries original output first; on invalid item count, retry up to 3 times with the original required count and one item per line. Raw outputs and retries retained; no gold access.'
        p['differences_from_paper'] += [p['gallery_metadata_adaptation'], p['reflection_format_policy']]
        p['experiment_signature'] = base.obj(p)
        base.dump(path, p)
    snapshot = base.read(OUTPUT / 'source_snapshot/manifest.json')
    for path in dependencies:
        digest = base.sha(path)
        target = OUTPUT / 'source_snapshot' / (digest[:12] + '_' + path.name)
        target.write_bytes(path.read_bytes())
        snapshot[str(path)] = dict(path=str(target), sha256=digest)
    base.dump(OUTPUT / 'source_snapshot/manifest.json', snapshot)
    base.dump(OUTPUT / 'metadata_adaptation.json', dict(status='prepared',
        supersedes=str(base.OUTPUT), reason='Restore original source IDs and timestamps omitted by the MemEye GA/MemoryOS input wrappers.',
        changed_algorithms=False, regenerated_captions=False, reference_inputs_unchanged=True))


def install(broker, protocol):
    encoder = ORIGINAL_INSTALL(broker, protocol)
    if protocol['mode'] == 'gen_agents':
        from memengine.function.Reflector import GAReflector
        lock = threading.Lock()

        def record(event):
            with lock:
                with (OUTPUT / 'gen_agents/format_retry_events.jsonl').open('a') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')

        def questions(self, inputs):
            return reflection_lines(self, inputs, 'question', getattr(broker.local, 'owner', None), record)

        def insights(self, inputs):
            return reflection_lines(self, inputs, 'insight', getattr(broker.local, 'owner', None), record)

        GAReflector.self_ask, GAReflector.generate_insight = questions, insights
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'audit', 'export'])
    parser.add_argument('--method', choices=base.METHODS)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
        return
    base.native.helpers = gallery_helpers
    base.native.install_runtime = install
    if args.stage == 'run':
        assert args.method
        base.run(OUTPUT, args.method, args.device)
    else:
        from scripts.analysis import audit_mem_gallery_text_methods as audit
        audit.OUTPUT = OUTPUT
        if args.stage == 'audit':
            assert args.method
            audit.audit_method(args.method)
        else:
            audit.export()


if __name__ == '__main__':
    main()
