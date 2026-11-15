import json
from pathlib import Path
from datetime import datetime, timezone
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory import hippomm as base, hippomm_original as method
report = Path(__file__).resolve().parent
output = ROOT / 'results/hippomm_original_controlled_20261104'
checks = {}
lines = ['HippoMM representative-originals controlled comparison',
         'Qwen3-VL-8B-Instruct; GPU 0; identical retrieval and context',
         'All fast and empty-retrieval answers reused exactly.', '']
for ds, expected in [('memeye',371), ('mem_gallery',1711)]:
    root = output / ds
    method.code_manifest(root / 'online')
    oldroot = ROOT / 'results/hippomm_short_ids_20261104' / ds
    oldsummary = json.loads((oldroot / 'online/summary.json').read_text())
    summary = json.loads((root / 'online/summary.json').read_text())
    completion = json.loads((root / 'online/completion_manifest.json').read_text())
    answers = base.inputs.read_lines(root / 'online/qa_responses.jsonl')
    original = base.inputs.unique(base.inputs.read_lines(root / 'online/baseline_responses.jsonl'))
    questions = base.inputs.unique(base.inputs.read_lines(root / 'online/questions.jsonl'))
    scores = base.inputs.read_lines(root / 'online/per_question.jsonl')
    assert completion['status'] == 'complete' and completion['questions'] == expected
    assert len(answers) == len(scores) == len(questions) == expected
    assert set(base.inputs.unique(answers)) == set(questions) == set(original) == set(base.inputs.unique(scores))
    for row in answers:
        prior = original[row['id']]
        method.check_trace(row, prior)
        reuse = prior['route'] == 'fast' or not prior['selected_event_ids']
        assert row['answer_reused'] == reuse
        if reuse:
            assert row['answer'] == prior['answer'] and not row['llm_calls']
            assert row['historical_image_reads'] == 0
        else:
            assert row['historical_image_reads'] == len(prior['selected_event_ids'])
    for name in method.MEMORY_FILES:
        assert (root / 'offline' / name).read_bytes() == (oldroot / 'offline' / name).read_bytes()
    assert summary['context_source_micro_recall'] == oldsummary['context_source_micro_recall']
    names = ['accuracy'] if ds == 'memeye' else ['em','f1','bleu']
    for metric in names:
        assert abs(summary['baseline'][metric] - oldsummary[metric]) < 1e-12
        assert abs(sum(r[metric] for r in scores) / expected - summary[metric]) < 1e-12
    key = 'accuracy' if ds == 'memeye' else 'em'
    gained = sum(r[key] == 1 and r['baseline_metrics'][key] == 0 for r in scores)
    lost = sum(r[key] == 0 and r['baseline_metrics'][key] == 1 for r in scores)
    checks[ds] = {'status':'complete','questions':expected,'gained_correct':gained,'lost_correct':lost,
                  'net_correct':gained-lost,'summary':summary,
                  'checks':['full unique coverage','identical memory and frozen retrieval',
                            'fast and empty answers unchanged','baseline scores reproduced',
                            'metrics recomputed','completion requires exact image payload audit']}
    lines.append(f'{ds}: {expected}/{expected} complete')
    for metric in names:
        lines.append(f"{metric}: {summary['baseline'][metric]*100:.2f} -> {summary[metric]*100:.2f}; "
                     f"delta {summary['delta_percentage_points'][metric]:+.2f} points")
    lines.append(f"Memory: {summary['total_memory_bytes']/1e6:.2f} MB; "
                 f"{summary['representative_images_stored']} original representatives")
    lines.append(f'Correct answers gained/lost: {gained}/{lost}; net {gained-lost:+d}')
    lines.append('')
base.dump(report / 'final_verification.json', {'status':'complete','verified_at':datetime.now(timezone.utc).isoformat(),'datasets':checks})
(report / 'COMPARISON.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
