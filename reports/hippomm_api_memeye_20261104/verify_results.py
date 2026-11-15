"""Independent final coverage and fixed-visual-input audit; no model calls."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory import hippomm_inputs as inputs, hippomm_api as method

REPORT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/hippomm_api_memeye_20261104'


def main():
    result = {'dataset': 'memeye', 'questions_per_backbone': 371, 'backbones': {}}
    question_sets = []
    for name in ('gpt5', 'deepseek'):
        offline, online = OUTPUT / name / 'offline', OUTPUT / name / 'online'
        completion = json.loads((online / 'completion_manifest.json').read_text())
        assert completion['status'] == 'complete' and completion['questions'] == 371
        assert completion['provider_and_image_payload_audit'] == 'passed'
        memory, events, _, _, image_map = method.load_memory(offline, name)
        for filename in ('dialogues.jsonl', 'image_mapping.jsonl', 'visual_embeddings.npz', 'representative_images.jsonl'):
            assert (offline / filename).read_bytes() == (method.TEMPLATE / filename).read_bytes(), filename
        for item in image_map.values():
            assert (offline / item['file']).read_bytes() == (method.TEMPLATE / item['file']).read_bytes()
        questions = inputs.read_lines(online / 'questions.jsonl')
        responses = inputs.read_lines(online / 'qa_responses.jsonl')
        assert len(inputs.unique(questions)) == len(inputs.unique(responses)) == 371
        assert set(inputs.unique(questions)) == set(inputs.unique(responses))
        question_sets.append(questions)
        summary = json.loads((online / 'summary.json').read_text())
        assert summary['questions'] == 371 and summary['representative_images'] == 250
        assert sum(r['historical_image_reads'] > 0 for r in responses) <= summary['detailed_answers']
        summaries = list((offline / 'llm_cache').glob('*/call_*.json'))
        calls = list((online / 'llm_cache').glob('*/call_*.json'))
        roles = {}
        for path in summaries + calls:
            record = json.loads(path.read_text())
            assert record['request']['backbone'] == name
            assert record['response']['returned_model'] == method.transport.MODELS[name]
            role = record['request']['role']
            roles[role] = roles.get(role, 0) + 1
        summary.update(cached_call_roles=roles,
                       answers_with_historical_images=sum(r['historical_image_reads'] > 0 for r in responses),
                       query_format_recoveries=sum(bool(r['query_format_recovery']) for r in responses),
                       fixed_visual_inputs_audit='passed')
        result['backbones'][name] = summary
    assert question_sets[0] == question_sets[1], 'Provider question inputs differ'
    path = REPORT / 'final_verification.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
