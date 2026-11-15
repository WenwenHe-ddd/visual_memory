"""MemEye-only rerun with a configurable shared search limit and exact caches."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import m2a_deepseek_suite as suite
from scripts.memory.m2a_search_budget import nonnegative_int


def prepare(root, parent, workers, additional_rounds=2):
    previous = suite.read(parent / 'memeye/summary.json')
    if not previous['coverage_complete'] or previous['questions'] != 371:
        raise ValueError('A complete MemEye baseline is required')
    plan = suite.prepare(root, workers=workers, datasets=('memeye',))
    plan['configuration'].update(max_additional_search_rounds=additional_rounds,
                                 max_search_calls_per_question=1 + additional_rounds)
    imported = []
    for task in previous['tasks']:
        out = Path(task['output']).resolve()
        manifest = out / 'completion_manifest.json'
        if suite.sha(manifest) != task['completion_sha256']:
            raise ValueError('Baseline completion changed: ' + task['task'])
        for name, expected in suite.read(manifest)['artifacts'].items():
            if suite.sha(out / name) != expected:
                raise ValueError('Baseline artifact changed: ' + str(out / name))
        target = root / 'memeye' / task['task'] / 'api_cache'
        target.mkdir(parents=True)
        caches = {Path(c['cache_directory']).resolve()
                  for c in suite.rows(out / 'llm_calls.jsonl') if c['status'] == 'ok'}
        for source in sorted(caches):
            # Only complete, successful entries are linked; new requests stay local.
            response = source / 'response.json'
            request = source / 'request.json.gz'
            if not response.exists() or not request.exists():
                raise ValueError('Incomplete baseline cache: ' + str(source))
            (target / source.name).symlink_to(source, target_is_directory=True)
            imported.append(dict(task=task['task'], source=str(source),
                                 request_sha256=suite.sha(request), response_sha256=suite.sha(response)))
    suite.write(root / 'cache_import.json', imported)
    plan['comparison'] = dict(parent=str(parent),
        parent_summary_sha256=suite.sha(parent / 'memeye/summary.json'),
        variable=f'At most {additional_rounds} additional semantic searches after the first, shared across nested agents',
        cache_policy='Read completed exact-request baseline cache entries; store new requests only in this run',
        cache_import_sha256=suite.sha(root / 'cache_import.json'))
    path = Path(__file__).resolve()
    target = root / 'code_snapshot' / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    plan['implementation_sha256'][str(path)] = suite.sha(path)
    suite.write(root / 'plan.json', plan)


def compare(root, parent):
    limit = suite.read(root / 'plan.json')['configuration']['max_search_calls_per_question']
    old = suite.read(parent / 'memeye/summary.json')
    new = suite.read(root / 'memeye/summary.json')
    if not new['coverage_complete'] or not new['strict_audit_passed']:
        return
    before = {t['task']: t for t in old['tasks']}
    changes, distribution, calls, matching_memories = [], Counter(), Counter(), []
    for task in new['tasks']:
        out, source = Path(task['output']), Path(before[task['task']]['output'])
        if suite.read(out / 'protocol.json')['max_search_calls_per_question'] != limit:
            raise ValueError('Executed search cap differs from the suite plan')
        original = {r['id']: r for r in suite.rows(source / 'responses.jsonl')}
        for r in suite.rows(out / 'responses.jsonl'):
            distribution[r['search_calls']] += 1
            if r['predicted_choice'] != original[r['id']]['predicted_choice']:
                changes.append(dict(id=r['id'], before=original[r['id']]['predicted_choice'],
                                    after=r['predicted_choice'], correct_before=original[r['id']]['correct'],
                                    correct_after=r['correct']))
        for c in suite.rows(out / 'llm_calls.jsonl'):
            calls['cached' if c['cache_replayed'] else 'fresh'] += 1
        def contents(directory):
            return sorted((m['text'], m['image_caption'], m['image_path'], m['evidence_ids'])
                          for m in suite.read(directory / 'semantic_before_qa.json'))
        matching_memories.append(dict(task=task['task'], identical=contents(out) == contents(source)))
    suite.write(root / 'memeye/comparison.json', dict(
        questions=new['questions'], correct_before=old['correct'], correct_after=new['correct'],
        accuracy_before=old['accuracy'], accuracy_after=new['accuracy'],
        accuracy_delta=new['accuracy'] - old['accuracy'], search_distribution=dict(distribution),
        max_search_calls_per_question=limit,
        search_cap_respected=max(distribution) <= limit, calls=dict(calls),
        memory_content_comparison=matching_memories, changed_answers=changes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-additional-search-rounds', type=nonnegative_int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root, parent = args.output.resolve(), args.parent.resolve()
    if not (root / 'plan.json').exists():
        prepare(root, parent, args.workers, args.max_additional_search_rounds)
    plan = suite.verify(root)
    if plan['comparison']['parent'] != str(parent) or set(plan['datasets']) != {'memeye'}:
        raise ValueError('Rerun scope or baseline differs')
    if plan['configuration']['max_additional_search_rounds'] != args.max_additional_search_rounds:
        raise ValueError('Requested search cap differs from the frozen plan')
    if args.prepare_only:
        return
    original_command = suite.command

    def command(task, out, cache, env_file):
        return original_command(task, out, cache, env_file) + [
            '--max-additional-search-rounds', str(args.max_additional_search_rounds)]

    suite.command = command
    if not (root / 'probe.json').exists():
        suite.probe(root, suite.DEFAULT_ENV)
    suite.run(root, suite.DEFAULT_ENV)
    compare(root, parent)


if __name__ == '__main__':
    main()
