"""Prepare DeepSeek V4.1 Flash SimpleMem provenance evaluation with the matched local evaluator.

Window/alias/job construction deliberately matches the frozen Qwen preparation.
Only the source audit format and evaluation-model configuration differ. Shared
run, review, freeze and score functions retain the original inference protocol.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil

from scripts.analysis import simplemem_source_attribution as original
from scripts.analysis import simplemem_attribution_recall as recall
from scripts.analysis.simplemem_source_attribution import read, verify_inputs, RULES
from scripts.evaluation import mcq_reader
from scripts.memory.reflexion_memory import hash_object, read_rows, sha, write_json, write_rows

def prepare(source, root, evaluator):
    if root.exists():
        raise ValueError('output exists; use run/collect/score to resume')
    p = read(source / 'protocol.json')
    audit = read(source / 'backbone_audit.json')
    if not (audit['passed'] and audit['complete']):
        raise ValueError('source run must pass its full backbone audit')
    if audit['experiment_signature'] != p['experiment_signature']:
        raise ValueError('source audit signature differs')
    evaluation = read(evaluator / 'protocol.json')
    verify_inputs(evaluation, evaluator)
    # Selection uses retrieved memory IDs only. No question, option, prediction,
    # or gold label is placed in any attribution job or model prompt.
    selected = defaultdict(set)
    for row in read_rows(source / 'qa_responses.jsonl'):
        selected[row['task']].update(row['retrieval_detail']['context_memory_ids'])
    observations = read(source / 'observations.json')
    inputs = {str(f.resolve()): sha(f) for f in [source / 'protocol.json', source / 'qa_responses.jsonl',
        source / 'observations.json', source / 'backbone_audit.json', Path(__file__),
        evaluator / 'protocol.json', Path(original.__file__), Path(recall.__file__),
        Path(__file__).parents[1] / 'memory/reflexion_memory.py', Path(mcq_reader.__file__)]}
    jobs, entry_index = [], {}
    for task in p['tasks']:
        paths = [source / task / 'final_memory.json', source / task / 'runtime/construction_windows.json',
                 source / task / 'runtime/dialogue_sources.json']
        inputs.update({str(f.resolve()): sha(f) for f in paths})
        entries = {e['entry_id']: e for e in read(paths[0])}
        windows, dialogue_sources = read(paths[1]), read(paths[2])
        rounds = {r['id']: r for r in observations[task]}
        previous, dependencies = [], {}
        for window in windows:
            for identity in window['entry_ids']:
                dependencies[identity] = list(previous)
            if window['entry_ids']:
                previous = window['entry_ids'][:3]
        required, pending = set(selected[task]), list(selected[task])
        while pending:
            for dep in dependencies[pending.pop()]:
                if dep not in required:
                    required.add(dep)
                    pending.append(dep)
        for wi, window in enumerate(windows):
            current = [i for i in window['entry_ids'] if i in required]
            if not current:
                continue
            source_records = []
            # Only expose turns present in this window, including the exact
            # user/assistant split and captions used at construction time.
            present = defaultdict(set)
            for did in window['dialogue_ids']:
                origin = dialogue_sources[str(did)]
                present[origin['round_id']].add(origin['speaker'])
            for ri, rid in enumerate(window['source_round_ids'], 1):
                row = rounds[rid]
                user_present = row['speaker'] in present[rid]
                source_records.append({'alias': f'R{ri}', 'kind': 'round', 'round_id': task + '/' + rid,
                    'timestamp': row['timestamp'], 'speaker': row['speaker'],
                    'user': row['user'] if user_present else '',
                    'caption': '\n'.join(row.get('raw', {}).get('image_caption', [])) if user_present else '',
                    'assistant': row['assistant'] if 'assistant' in present[rid] else ''})
            for pi, identity in enumerate(dependencies[current[0]], 1):
                source_records.append({'alias': f'P{pi}', 'kind': 'prior_entry', 'entry_id': identity,
                                       'memory': entries[identity]['lossless_restatement']})
            for start in range(0, len(current), 8):
                ids = current[start:start + 8]
                key = f'{task}__window{wi:02d}__group{start // 8:02d}'
                targets = [{'alias': f'M{i}', 'entry_id': identity, 'memory': entries[identity]['lossless_restatement'],
                            'selected': identity in selected[task]} for i, identity in enumerate(ids, 1)]
                job = {'id': key, 'task': task, 'window_index': wi, 'sources': source_records, 'targets': targets}
                jobs.append(job)
                for target in targets:
                    entry_index[target['entry_id']] = {'task': task, 'job_id': key, **target}
    root.mkdir(parents=True)
    write_rows(root / 'jobs.jsonl', jobs)
    write_json(root / 'entry_index.json', entry_index)
    config = {k: evaluation[k] for k in ('reader_model', 'reader_model_path', 'seed', 'dtype', 'context_budget')}
    config.update(reader_device_map={'': 0}, max_new_tokens=4096, max_padded_batch_tokens=131072,
                  generation_batch_size=8, source_root=str(source.resolve()),
                  source_experiment_signature=p['experiment_signature'], input_hashes=inputs,
                  evaluator_control_root=str(evaluator.resolve()),
                  evaluation_scope='Frozen DeepSeek V4.1 Flash contexts; same local Qwen attribution evaluator as control',
                  jobs_sha256=sha(root / 'jobs.jsonl'), entry_index_sha256=sha(root / 'entry_index.json'),
                  selected_entries=sum(len(v) for v in selected.values()), annotation_entries=len(entry_index),
                  method='Two independent ordering passes with local Qwen, literal-quote validation, '
                         'source-set agreement and explicit disagreement adjudication; reconstructed provenance',
                  gold_blind=True, rules=RULES)
    config['signature'] = hash_object(config)
    write_json(root / 'protocol.json', config)
    shutil.copyfile(__file__, root / 'source_snapshot.py')
    write_json(root / 'progress.json', {'status': 'prepared', 'jobs_per_pass': len(jobs),
        'annotation_entries': len(entry_index), 'selected_entries': config['selected_entries']})
    print(f'Prepared {len(jobs)} jobs/pass, {len(entry_index)} entries, {config["selected_entries"]} selected', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluator-control', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.root.resolve(), args.evaluator_control.resolve())
