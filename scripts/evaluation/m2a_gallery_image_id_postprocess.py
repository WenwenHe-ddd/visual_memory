"""Rescore Gallery outputs after mapping internal image tokens to source image IDs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.m2a_inputs import gallery_reader, task_names
from scripts.evaluation.m2a_suite import read, write, sha, now

KEYS = ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')


def run(recovery, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = read(recovery/'recovery_summary.json')
    assert sorted(t['task'] for t in summary['tasks']) == task_names('memgallery')
    scorer = gallery_reader().score_answer
    totals = {'before': Counter(), 'after': Counter()}
    by_type = defaultdict(lambda: {'before': Counter(), 'after': Counter(), 'questions': 0})
    source_hashes, transformations, results, task_results = {}, [], [], []
    seen = set()
    for task in summary['tasks']:
        out = Path(task['output'])
        protocol = read(out/'protocol.json')
        source = ROOT/'data/Mem-Gallery/data/dialog'/(task['task']+'.json')
        assert sha(source) == protocol['sources'][str(source)]
        source_hashes[str(source)] = sha(source)
        path_to_id = {}
        for session in read(source)['multi_session_dialogues']:
            for turn in session['dialogues']:
                refs = turn.get('input_image') or []
                refs = [refs] if isinstance(refs, str) else refs
                ids = turn.get('image_id') or []
                assert len(refs) == len(ids)
                for ref, official_id in zip(refs, ids):
                    path = str((source.parent/ref).resolve())
                    assert re.fullmatch(r'D\d+:IMG_\d+', official_id)
                    assert path not in path_to_id or path_to_id[path] == official_id
                    path_to_id[path] = official_id
        manager = read(out/'image_manager.json')
        aliases = {f'<image{i}>':path_to_id[path] for i,path in enumerate(manager['image_paths'])
                   if path in path_to_id}
        for i,path in enumerate(manager['image_paths']):
            assert manager['image_path_to_id'][path] == i
        for name in ('responses.jsonl', 'image_manager.json', 'evaluation_labels.json', 'protocol.json', 'completion_manifest.json'):
            source_hashes[str(out/name)] = sha(out/name)
        labels = read(out/'evaluation_labels.json')
        answers = [json.loads(line) for line in (out/'responses.jsonl').read_text().splitlines()]
        assert [a['id'] for a in answers] == [q['id'] for q in labels]
        task_totals = {'before': Counter(), 'after': Counter()}
        for answer, label in zip(answers, labels):
            assert answer['id'] not in seen
            seen.add(answer['id'])
            original = answer['raw_answer']
            replacements = []

            def replace(match):
                token = match[0]
                if token not in aliases:
                    return token
                replacements.append(dict(token=token, image_path=manager['image_paths'][int(token[6:-1])],
                                         official_image_id=aliases[token]))
                return aliases[token]

            mapped = re.sub(r'<image\d+>', replace, original) if answer['point'] == 'VS' else original
            # Gold is used only for scoring, after the input-derived mapping.
            before, after = scorer(original, label['answer']), scorer(mapped, label['answer'])
            assert all(abs(before[k] - answer[k]) < 1e-12 for k in KEYS)
            if not replacements:
                assert mapped == original and before == after
            for phase, score in [('before', before), ('after', after)]:
                totals[phase].update(score)
                task_totals[phase].update(score)
                by_type[answer['point']][phase].update(score)
            by_type[answer['point']]['questions'] += 1
            result = dict(answer, raw_answer=mapped, correct=bool(after['em']), **after)
            result.update(original_raw_answer=original, original_correct=answer['correct'],
                original_metrics=before, image_id_replacements=replacements, source_output=str(out),
                postprocessing='historical image path to official ID; VS only; preserve order and other text')
            results.append(result)
            if replacements:
                transformations.append(dict(id=answer['id'], replacements=replacements,
                    before=original, after=mapped, before_em=before['em'], after_em=after['em']))
        task_results.append(dict(task=task['task'], questions=len(answers),
            metrics={phase:{k:counts[k]/len(answers) for k in KEYS} for phase,counts in task_totals.items()}))
    assert len(results) == summary['expected_questions'] == 1711
    for path, expected in source_hashes.items():
        assert sha(path) == expected, 'Source modified during postprocessing: ' + path
    for phase in totals:
        for k in KEYS:
            if phase == 'before':
                assert abs(totals[phase][k]/len(results) - summary['metrics'][k]) < 1e-12
    with (output/'responses.jsonl').open('w') as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False)+'\n')
    write(output/'mapping_audit.json', dict(passed=True, questions=len(results),
        source_files_unchanged=True, original_scores_recomputed=True, transformations=transformations,
        scope='Only VS answers; no gold-conditioned selection, no sorting, no extra model calls'))
    report = dict(created_at=now(), method='M2A single-pass adaptation plus deterministic image ID output mapping',
        questions=len(results), tasks=task_results, llm_calls_added=0, changed_answers=len(transformations),
        correct_before=int(totals['before']['em']), correct_after=int(totals['after']['em']),
        em_gains=sum(r['em'] > r['original_metrics']['em'] for r in results),
        em_losses=sum(r['em'] < r['original_metrics']['em'] for r in results),
        metrics={phase:{k:c[k]/len(results) for k in KEYS} for phase,c in totals.items()},
        by_type={p:dict(questions=c['questions'],
                    metrics={phase:{k:c[phase][k]/c['questions'] for k in KEYS} for phase in ('before','after')})
                 for p,c in sorted(by_type.items())},
        known_source_issues=summary['known_source_issues'],
        overall_strict_audit_passed=summary.get('strict_audit_passed',False),
        note='All 1711 original questions retained, including the known reversed question/answer record; no raw experiment outputs overwritten',
        source_sha256=source_hashes, recovery_summary_sha256=sha(recovery/'recovery_summary.json'))
    write(output/'summary.json',report)
    shutil.copy2(__file__, output/'postprocess_code.py')
    write(output/'completion_manifest.json',dict(status='completed',operation='deterministic postprocessing and rescoring',
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in report.items() if k not in ('tasks','by_type','source_sha256')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.recovery.resolve(), args.output.resolve())
