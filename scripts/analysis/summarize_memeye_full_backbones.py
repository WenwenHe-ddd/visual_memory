"""Combine completed, independently audited MemEye full-backbone experiments.

All validation finishes before either comparison file is replaced. This module
does not import experiment runners, call APIs, or alter model-specific results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / 'results/memeye_full_backbone_20261021'
DEFAULT_MODELS = ('gpt-4o-mini-2024-07-18', 'gpt-5.4-2026-03-05',
                  'deepseek-v4.1-flash-1017_complete_captions')
MANIFESTS = ('source_memories.jsonl', 'source_questions.jsonl', 'evaluation.jsonl',
             'images.jsonl', 'api_requests_caption.jsonl')
CONDITIONS = {'baseline_question_only': 'baseline', 'master_answer_query': 'expanded'}
COLUMNS = ('model', 'scope', 'condition', 'questions', 'answer_correct',
           'answer_accuracy', 'gold_hits_at_5', 'gold_references',
           'recall_at_5_micro', 'recall_at_5_macro', 'provider',
           'caption_max_output_tokens', 'caption_budget_policy')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    require(Path(path).is_file(), f'Experiment is incomplete: missing {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path, key):
    require(Path(path).is_file(), f'Experiment is incomplete: missing {path}')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed = {row[key]: row for row in rows}
    require(len(indexed) == len(rows), f'Duplicate {key} in {path}')
    return indexed


def equal_number(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f'Metric differs from recorded questions: {label}')


def collect(root, models):
    """Validate every requested model and return a ready-to-write comparison."""
    root = Path(root).resolve()
    require(models and len(set(models)) == len(models), 'Models must be nonempty and unique')
    require(all(Path(model).name == model and model not in ('.', '..') for model in models),
            'Each model must be a directory name under the experiment root')
    rows, sources, reference_manifests, input_hashes = [], {}, None, {}
    caption_budgets, caption_policies, run_directories = {}, {}, {}
    fixed_components, reference_domains = None, None
    for run_directory in models:
        directory = root / run_directory
        protocol = read(directory / 'protocol.json')
        audit = read(directory / 'independent_audit.json')
        audit_hash = sha(directory / 'independent_audit.json')
        require(audit.get('audit_status') == 'passed' and audit.get('errors') == [],
                f'{run_directory}: independent audit has not passed')
        summary = read(directory / 'summary.json')
        retrieval = read(directory / 'retrieval_summary.json')
        contract = protocol['contract']
        model = contract['model']
        require(model not in run_directories, f'Duplicate backbone snapshot: {model}')
        run_directories[model] = str(directory)
        signature = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                               separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        require(signature == protocol['experiment_signature'], f'{model}: root signature differs')
        require(contract['scope'] == summary['scope'] == 'all_original_qwen_roles'
                and contract['model'] == summary['model'] == audit['model'] == model
                and protocol['run_id'] == summary['run_id'] == audit['run_id'],
                f'{model}: experiment identity or scope differs')
        require(summary['historical_qwen_caption_or_answer_reuse'] is False
                and summary['logical_reader_records'] == 742
                and summary['generation_stages']['caption_generation']['fresh_requests'] == 437
                and audit['inputs']['questions'] == 371 and audit['inputs']['domains'] == 8
                and audit['inputs']['memory_leaves'] == 438,
                f'{model}: full-backbone coverage differs')
        caption_stage = audit['caption_generation']
        require(caption_stage['fresh_requests'] == 437 and caption_stage['length_stops'] == 0
                and caption_stage['historical_response_reuse'] == 0,
                f'{model}: captions are incomplete or reused from another run')
        budget = contract['caption_max_output_tokens']
        require(type(budget) is int and budget > 0, f'{model}: invalid caption output budget')
        caption_budgets[model] = budget
        caption_policies[model] = (
            f'Fresh complete captions with a {budget}-token output budget; '
            'all accepted captions ended normally, without output truncation.')
        require(audit['retrieval']['metrics'] == retrieval == summary['recall_at_5']
                and audit['answers']['overall'] == summary['overall']
                and audit['answers']['by_domain'] == summary['by_domain'],
                f'{model}: summaries differ from the independent audit')
        # A passed audit must describe the current bytes, not an older summary.
        hashes = audit['artifact_sha256']
        required = ('protocol.json', 'summary.json', 'retrieval_summary.json',
                    'paired_results.jsonl', 'retrieval_results.jsonl',
                    *('inputs/' + name for name in MANIFESTS))
        require(all(str(directory / name) in hashes for name in required),
                f'{model}: audit does not bind all comparison inputs')
        for path, expected in hashes.items():
            require(Path(path).is_file() and sha(path) == expected,
                    f'{model}: artifact changed after audit: {path}')
        for name in ('protocol.json', 'summary.json', 'retrieval_summary.json'):
            sources[str(directory / name)] = hashes[str(directory / name)]
        sources[str(directory / 'independent_audit.json')] = audit_hash
        manifests = {name: (directory / 'inputs' / name).read_bytes() for name in MANIFESTS}
        if reference_manifests is None:
            reference_manifests = manifests
        require(manifests == reference_manifests,
                f'{model}: the five input manifests differ byte-for-byte across backbones')
        input_hashes[model] = {name: hashlib.sha256(data).hexdigest() for name, data in manifests.items()}
        for name, value in input_hashes[model].items():
            sources[str(directory / 'inputs' / name)] = value
        if fixed_components is None:
            fixed_components = contract['fixed_components']
        require(contract['fixed_components'] == fixed_components,
                f'{model}: fixed-component contract differs')
        questions = records(directory / 'inputs/source_questions.jsonl', 'id')
        evaluation = records(directory / 'inputs/evaluation.jsonl', 'id')
        results = records(directory / 'retrieval_results.jsonl', 'question_id')
        pairs = records(directory / 'paired_results.jsonl', 'question_id')
        require(len(questions) == 371 and set(questions) == set(evaluation) == set(results) == set(pairs),
                f'{model}: question identities or coverage differ')
        domains = sorted({question['task'] for question in questions.values()})
        require(len(domains) == 8 and set(domains) == set(summary['by_domain']) == set(retrieval['by_domain']),
                f'{model}: domain coverage differs')
        if reference_domains is None:
            reference_domains = domains
        require(domains == reference_domains, f'{model}: domains differ across backbones')
        provider = contract.get('provider')
        if provider is None:
            require(contract['api_config']['base_url'] == 'https://api.openai.com/v1',
                    f'{model}: unknown provider')
            provider = 'OpenAI'
        for scope in ['overall', *domains]:
            ids = [qid for qid, question in questions.items()
                   if scope == 'overall' or question['task'] == scope]
            answer_metrics = summary['overall'] if scope == 'overall' else summary['by_domain'][scope]
            recall_metrics = retrieval['overall'] if scope == 'overall' else retrieval['by_domain'][scope]
            gold_count = sum(len(set(evaluation[qid]['gold_ids'])) for qid in ids)
            require(answer_metrics['questions'] == recall_metrics['questions'] == len(ids)
                    and recall_metrics['gold_references'] == gold_count,
                    f'{model}/{scope}: denominators differ')
            if scope == 'overall':
                require(len(ids) == 371 and gold_count == 1041, f'{model}: expected 371 questions and 1041 gold references')
            for condition, field in CONDITIONS.items():
                hits, recalls, correct = 0, [], 0
                for qid in ids:
                    gold = set(evaluation[qid]['gold_ids'])
                    conditions = {row['name']: row for row in results[qid]['conditions']}
                    require(len(results[qid]['conditions']) == 2 and set(conditions) == set(CONDITIONS),
                            f'{model}/{qid}: retrieval conditions differ')
                    top = conditions[condition]['top5_memory_ids']
                    require(gold and len(top) == len(set(top)) == 5, f'{model}/{qid}: invalid gold or Top5')
                    hit = len(gold.intersection(top))
                    hits += hit
                    recalls.append(hit / len(gold))
                    answer = pairs[qid][field + '_answer']
                    require(answer in 'ABCD' and len(answer) == 1, f'{model}/{qid}: invalid reader choice')
                    is_correct = answer == evaluation[qid]['gold_choice']
                    require(pairs[qid][field + '_correct'] is is_correct,
                            f'{model}/{qid}: saved answer correctness differs')
                    correct += is_correct
                metric = recall_metrics[condition]
                require(answer_metrics[field + '_correct'] == correct and metric['gold_hits_top5'] == hits,
                        f'{model}/{scope}/{condition}: metric numerators differ')
                equal_number(answer_metrics[field + '_accuracy'], correct / len(ids), 'answer accuracy')
                equal_number(metric['gold_recall'], hits / gold_count, 'micro Recall@5')
                equal_number(metric['macro_recall_at_5'], sum(recalls) / len(ids), 'macro Recall@5')
                rows.append(dict(zip(COLUMNS, (model, scope, condition, len(ids), correct,
                    answer_metrics[field + '_accuracy'], hits, gold_count,
                    metric['gold_recall'], metric['macro_recall_at_5'], provider,
                    budget, caption_policies[model]))))
        for name in ('paired_results.jsonl', 'retrieval_results.jsonl'):
            sources[str(directory / name)] = hashes[str(directory / name)]
    require(len(rows) == len(models) * 18, 'Unexpected comparison row count')
    return {'scope': 'all_original_qwen_roles', 'source_sha256': sources, 'rows': rows,
            'models': len(models), 'caption_requests_per_model': 437, 'memory_leaves_per_model': 438,
            'reader_logical_conditions_per_model': 742,
            'retrieval_policy': 'Original BGE model-limit truncation with audit; original full text retained for readers.',
            'fixed_components': ['CLIP', 'BGE', 'Ward rule', 'external 2D bases',
                                 'assistant-authored option statements', 'reader prompt', 'question/option order'],
            'historical_qwen_caption_or_answer_reuse': False,
            'run_directories': run_directories,
            'caption_max_output_tokens_by_model': caption_budgets,
            'caption_budget_policy_by_model': caption_policies,
            'caption_output_budget_identical_across_models': len(set(caption_budgets.values())) == 1,
            'only_model_parameter_changed': False,
            'comparison_limitations': [
                'Caption output-token budgets and provider-specific generation controls are reported separately; this is not a model-parameter-only comparison.',
                'The initial DeepSeek 256-token run is excluded. Its replacement uses an independent fresh run with a 1024-token budget to retain complete captions.'
            ] if len(set(caption_budgets.values())) > 1 else [
                'Provider/model-specific generation controls may differ; see each frozen root protocol.'
            ],
            'identical_input_manifests_verified': list(MANIFESTS), 'input_manifest_sha256': input_hashes,
            'metric_definitions': {'recall_at_5_micro': 'Total recalled gold references / total gold references (1041).',
                                   'recall_at_5_macro': 'Mean of per-question recalled gold / per-question gold (371 questions).'}}


def summarize(root=DEFAULT_ROOT, models=DEFAULT_MODELS, output_dir=None):
    root = Path(root).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else root
    require(output != root or tuple(models) == DEFAULT_MODELS,
            'A model subset requires a separate --output-dir; the root comparison is reserved for all three models')
    comparison = collect(root, models)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(comparison['rows'])
    contents = {'comparison.csv': buffer.getvalue(),
                'comparison.json': json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + '\n'}
    for path, expected in comparison['source_sha256'].items():
        require(sha(path) == expected, f'Comparison source changed during validation: {path}')
    output.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for name, content in contents.items():
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                             dir=output, prefix='.' + name + '.', delete=False) as handle:
                handle.write(content)
                temporary.append((Path(handle.name), output / name))
        for source, target in temporary:
            source.replace(target)
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS),
                        help='Run directory names under --root; model labels come from audited protocols')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.root, args.models, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot write comparison: {error}\n')
    print(json.dumps({'models': result['models'], 'rows': len(result['rows']),
                      'output': str((args.output_dir or args.root).resolve())}))


if __name__ == '__main__':
    main()
