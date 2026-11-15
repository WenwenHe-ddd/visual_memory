"""Recover legacy SM(T) graph snapshots using exact cached LLM outputs only.

The original release gives every graph a shared save path. This script preserves
that final snapshot, proves it did not exist when the eight task instances began,
and reconstructs their graphs without generating or changing any predictions.
"""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.evaluation import mcq_reader
from scripts.memory.reflexion_memory import ROOT, read_rows, write_json, sha, hash_object
from scripts.memory.text_memory_baselines import VENDOR, verify_vendor
from scripts.memory.text_memory_transport import Broker, client_factory


def semantic_graph(path):
    entities = read_rows(path / 'entities.jsonl')
    relations = read_rows(path / 'relations.jsonl')
    names = {e['entity_id']: e['name'] for e in entities}
    normalized_entities = []
    for entity in entities:
        row = {k: v for k, v in entity.items() if k not in ('entity_id', 'first_seen', 'last_updated')}
        row['aliases'] = sorted(row['aliases'])
        row['source_mau_ids'] = sorted(row['source_mau_ids'])
        normalized_entities.append(row)
    normalized_relations = []
    for relation in relations:
        row = {k: v for k, v in relation.items() if k not in ('relation_id', 'first_seen')}
        row['subject_id'] = names[row['subject_id']]
        row['object_id'] = names[row['object_id']]
        row['source_mau_ids'] = sorted(row['source_mau_ids'])
        normalized_relations.append(row)
    return {'entities': sorted(normalized_entities, key=hash_object),
            'relations': sorted(normalized_relations, key=hash_object)}


def recover(root, legacy):
    verify_vendor()
    protocol = json.loads((root / 'protocol.json').read_text())
    assert len(protocol['tasks']) == protocol['task_workers'] == 8
    birth_times = {name: int(subprocess.check_output(['stat', '-c', '%W', str(legacy / name)]))
                   for name in ('entities.jsonl', 'relations.jsonl')}
    assert all(birth_times.values()), 'filesystem birth-time evidence required'
    first_call_times = {}
    for event in read_rows(root / 'call_events.jsonl'):
        if event['owner'] in protocol['tasks']:
            first_call_times.setdefault(event['owner'], (root / 'calls' / (event['key'] + '.json')).stat().st_mtime)
    assert set(first_call_times) == set(protocol['tasks'])
    assert max(first_call_times.values()) < min(birth_times.values()), 'cannot establish empty initial graph state'
    snapshot = root / 'legacy_graph_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in birth_times:
        target = snapshot / name
        if target.exists():
            assert sha(target) == sha(legacy / name)
        else:
            shutil.copyfile(legacy / name, target)
    transport_root = root / 'graph_recovery_transport'
    transport_root.mkdir(exist_ok=True)
    calls_link = transport_root / 'calls'
    if not calls_link.exists():
        calls_link.symlink_to((root / 'calls').resolve(), target_is_directory=True)
    assert calls_link.resolve() == (root / 'calls').resolve()

    class CacheOnly:
        missing = 0
        def generate(self, *args, **kwargs):
            self.missing += 1
            raise RuntimeError('Missing exact cached request; recovery cannot generate new outputs')
    model = CacheOnly()
    broker = Broker(model, transport_root, protocol)
    import openai
    openai.OpenAI = client_factory(broker)
    sys.path.insert(0, str(VENDOR))
    from omni_memory import OmniMemoryConfig
    from omni_memory.core.mau import MultimodalAtomicUnit
    from omni_memory.knowledge import EntityExtractor, KnowledgeGraph
    results = {}
    source_hashes = {}
    try:
        for task in protocol['tasks']:
            broker.local.owner = task
            directory = root / task / 'runtime'
            config_path = directory / 'resolved_config.json'
            config = OmniMemoryConfig.from_dict(json.loads(config_path.read_text()))
            source_hashes[str(config_path)] = sha(config_path)
            extractor = EntityExtractor(config)
            target = root / 'graph_recovery' / task
            if target.exists():
                raise ValueError('recovery output already exists; inspect it rather than rebuilding over it')
            graph = KnowledgeGraph(storage_path=str(target))
            # Keep original insertion order, with final copies of each unit.
            maus = {}
            for path in sorted(directory.rglob('mau_*.jsonl')):
                source_hashes[str(path)] = sha(path)
                for row in read_rows(path):
                    maus[row['id']] = row
            caught = []
            for row in maus.values():
                mau = MultimodalAtomicUnit.from_dict(row)
                entities, relations = extractor.extract(mau)
                assert model.missing == 0, 'a reconstruction request lacked an exact cache match'
                try:
                    for entity in entities:
                        graph.add_extracted_entity(entity)
                    for relation in relations:
                        graph.add_extracted_relation(relation)
                except Exception as exc:
                    caught.append({'mau_id': mau.id, 'error': str(exc)})
            graph.save()
            results[task] = {'memory_units': len(maus), **graph.get_stats(),
                             'caught_graph_update_errors': caught,
                             'snapshot_origin': 'reconstructed from original MAUs and exact cached entity-extraction outputs'}
        last_task = max(protocol['tasks'], key=lambda t: (root / t / 'complete.json').stat().st_mtime)
        assert semantic_graph(root / 'graph_recovery' / last_task) == semantic_graph(snapshot), 'last original snapshot does not match reconstruction'
        replay_events = read_rows(transport_root / 'call_events.jsonl')
        assert len(replay_events) == sum(r['memory_units'] for r in results.values())
        assert all(e['reused'] for e in replay_events)
        evidence = {'status': 'passed', 'global_graph_file_birth_times': birth_times,
            'task_first_model_call_times': first_call_times,
            'initial_graph_state': 'Files first created after all eight task constructors; independent in-memory graphs started empty.',
            'storage_bug': 'Upstream KnowledgeGraph() used a shared default save path; task close overwrote disk snapshots.',
            'effect_on_predictions': 'No shared-state reload after construction; original QA predictions unchanged.',
            'last_task': last_task, 'last_original_graph_semantically_matches_reconstruction': True,
            'replayed_calls': len(replay_events), 'new_llm_calls': model.missing,
            'replay_events_sha256': sha(transport_root / 'call_events.jsonl'),
            'recovered_graph_sha256': {str(p): sha(p) for p in (root / 'graph_recovery').rglob('*.jsonl')},
            'native_save_snapshot_sha256': {str(p): sha(p) for p in snapshot.glob('*.jsonl')},
            'source_hashes': source_hashes, 'tasks': results,
            'qa_response_sha256': sha(root / 'qa_responses.jsonl')}
        write_json(root / 'storage_isolation_audit.json', evidence)
        print(json.dumps({k: evidence[k] for k in ('status','last_task','replayed_calls','new_llm_calls')}))
    finally:
        broker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'results/visual_memory_pipeline/simplemem')
    parser.add_argument('--legacy-graph', type=Path, default=ROOT / 'omni_memory_data/knowledge_graph')
    args = parser.parse_args()
    recover(args.root, args.legacy_graph)
