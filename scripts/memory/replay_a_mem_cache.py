"""Replay pinned A-Mem construction/query calls for a K-only retrieval experiment.

Answer calls are never copied. The ordinary pipeline still reconstructs the
memory and reruns retrieval, and every imported call keeps explicit provenance.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.memory.reflexion_memory import hash_object, sha, write_json


def seed_memory_calls(source, target):
    source, target = source.resolve(), target.resolve()
    if source == target or (target / 'calls').exists():
        raise ValueError('use a freshly prepared target without calls')
    old = json.loads((source / 'protocol.json').read_text())
    new = json.loads((target / 'protocol.json').read_text())
    if old['mode'] != 'a_mem' or new['mode'] != 'a_mem':
        raise ValueError('only A-Mem supports this replay')
    for field in ('reader_model', 'reader_model_path', 'reader_system_prompt', 'seed',
                  'do_sample', 'dtype', 'max_new_tokens', 'memory_max_new_tokens',
                  'embedding_model_path', 'upstream', 'memory_sampling', 'query_policy',
                  'answer_prompt_policy', 'query_image_policy', 'tasks', 'questions',
                  'prepared_questions_sha256', 'observations_sha256'):
        if old[field] != new[field]:
            raise ValueError('incompatible replay field: ' + field)
    old_cfg = {k: v for k, v in old['method_defaults'].items() if k != 'retrieve_k'}
    new_cfg = {k: v for k, v in new['method_defaults'].items() if k != 'retrieve_k'}
    if old_cfg != new_cfg:
        raise ValueError('only retrieve_k may differ in method settings')
    for name, key in [('questions.jsonl', 'prepared_questions_sha256'), ('observations.json', 'observations_sha256')]:
        for directory, protocol in [(source, old), (target, new)]:
            text = (directory / name).read_text()
            value = [json.loads(s) for s in text.splitlines() if s.strip()] if name.endswith('jsonl') else json.loads(text)
            if hash_object(value) != protocol[key]:
                raise ValueError('prepared source data changed: ' + str(directory / name))
    # Current non-code inputs must still match the source experiment. Moved
    # historical source code is preserved by the source's own snapshot manifest.
    for name, digest in old['input_hashes'].items():
        if Path(name).suffix != '.py' and sha(Path(name)) != digest:
            raise ValueError('historical non-code input changed: ' + name)
    calls = []
    for path in sorted((source / 'calls').glob('*.json')):
        call = json.loads(path.read_text())
        request = call['request']
        if hash_object(request) != path.stem or request['experiment_signature'] != old['experiment_signature']:
            raise ValueError('source call provenance mismatch: ' + str(path))
        if request['kind'] == 'memory':
            calls.append((path, call))
    if not calls:
        raise ValueError('no memory calls to replay')
    evidence = target / 'memory_source'
    evidence.mkdir()
    shutil.copyfile(source / 'protocol.json', evidence / 'source_protocol.json')
    for task in old['tasks']:
        path = source / task / 'final_memory.json'
        destination = evidence / task / path.name
        destination.parent.mkdir()
        shutil.copyfile(path, destination)
    if (source / 'source_snapshot').is_dir():
        shutil.copytree(source / 'source_snapshot', evidence / 'source_snapshot')
    manifest = {'source_protocol_sha256': sha(source / 'protocol.json'),
                'source_experiment_signature': old['experiment_signature'],
                'policy': 'replay exact memory construction and query-keyword requests; rerun retrieval and every answer',
                'calls': [{'source_key': path.stem, 'source_sha256': sha(path),
                           'output_sha256': hash_object(call['output'])} for path, call in calls]}
    write_json(evidence / 'replay_manifest.json', manifest)
    new['memory_replay'] = {k: v for k, v in manifest.items() if k != 'calls'}
    new['memory_replay']['imported_calls'] = len(calls)
    new['input_hashes'][str(Path(__file__).resolve())] = sha(Path(__file__))
    for path in evidence.rglob('*'):
        if path.is_file():
            new['input_hashes'][str(path)] = sha(path)
    new.pop('experiment_signature')
    new['experiment_signature'] = hash_object(new)
    write_json(target / 'protocol.json', new)
    (target / 'calls').mkdir()
    for path, call in calls:
        replay = deepcopy(call)
        replay['request']['experiment_signature'] = new['experiment_signature']
        replay['replay_provenance'] = {'source_key': path.stem, 'source_sha256': sha(path),
                                     'source_experiment_signature': old['experiment_signature']}
        write_json(target / 'calls' / (hash_object(replay['request']) + '.json'), replay)
    print(f'Imported {len(calls)} memory calls; no answer calls imported', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    seed_memory_calls(args.source, args.target)
