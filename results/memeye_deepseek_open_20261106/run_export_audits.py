"""Read-only per-method export verification after each Judge becomes complete."""
from pathlib import Path
import json
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0, str(ROOT))
from scripts.analysis import export_memeye_deepseek_open_llm_judge as export

run = export.DEFAULT_ROOT
directory = run / 'export_audits'
directory.mkdir(exist_ok=True)
questions = export.unique(export.judge.read_rows(export.CANONICAL / 'inputs/questions.jsonl'))
gold = export.unique(export.judge.read_rows(export.CANONICAL / 'inputs/gold.jsonl'))
manifest = export.read(run / 'methods_manifest.json')
done = set()
while len(done) < 10:
    for method in manifest['methods']:
        if method['id'] in done:
            continue
        path = Path(method['root']) / 'judge/summary.json'
        if not path.exists() or export.read(path).get('status') != 'complete':
            continue
        if method['id'] == 'hippomm' and not (Path(method['root']) / 'independent_output_audit.json').exists():
            continue
        summary, detail = export.verify_method(method, questions, gold)
        report = {'passed': True, 'method_id': method['id'], 'questions': len(detail),
            'score_sum': summary['score_sum'], 'llm_judge': summary['llm_judge'],
            'score_distribution': summary['score_distribution'],
            'original_answer_API_wire_and_raw_attempts_validated': True,
            'grade_raw_response_and_recursive_cache_provenance_validated': True,
            'format_recovery_ids': summary['format_recovery_ids'],
            'export_script_sha256': export.checked_hash(Path(export.__file__)),
            'judge_summary_sha256': summary['judge_summary_sha256']}
        export.judge.dump(directory / (method['id'] + '.json'), report)
        done.add(method['id'])
        print(json.dumps({'audited': method['id'], 'complete_methods': len(done),
                          'llm_judge': summary['llm_judge']}), flush=True)
    if len(done) < 10:
        time.sleep(15)
export.judge.dump(directory / 'completion.json', {'status': 'complete', 'methods': 10,
    'method_questions': 3710, 'manifest_sha256': export.checked_hash(run / 'methods_manifest.json'),
    'export_script_sha256': export.checked_hash(Path(export.__file__))})
