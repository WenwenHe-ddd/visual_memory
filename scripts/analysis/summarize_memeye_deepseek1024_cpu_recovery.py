"""Pure standard-library file audit of the four CPU Open-tail recoveries."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
DIAGNOSTIC = OUT / 'metrics/recovery_diagnostics'
METHODS = ('gen_agents', 'memoryos', 'a_mem', 'simplemem')
EXPECTED_IDS = {'Social_Chat_Memory_Test/Q' + str(i) for i in (1, 2, 4, 8, 24, 25, 37, 38)}
ADAPTERS = {
    'gen_agents': 'recover_memeye_deepseek1024_text_open.py',
    'memoryos': 'recover_memeye_deepseek1024_memoryos_open.py',
    'a_mem': 'recover_memeye_deepseek1024_text_open.py',
    'simplemem': 'recover_memeye_deepseek1024_simplemem_open.py',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def query(question):
    value = question['question']
    if question['query_image_captions']:
        value += "\nquestion's image:\nimage_caption: " + ' '.join(question['query_image_captions'])
    return value


def main():
    plan = read(OUT / 'plan.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, 'An original pinned source changed: ' + path
    native_source = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_text_methods.py'
    methods, shared_mismatches, tails = [], None, set()
    unchanged_files, verified_snapshot_files = 0, 0
    for method in METHODS:
        folder = DIAGNOSTIC / (method + '_open_failure_20261106')
        path = folder / 'recovery_evidence.json'
        evidence = read(path)
        target = OUT / 'open/text_methods' / method
        snapshot = Path(evidence['original_failure_snapshot'])
        assert evidence['status'] == 'cpu_recovered' and evidence['questions'] == 371
        assert evidence['api_calls'] == evidence['gpu_model_loads'] == 0
        assert evidence['no_native_retrieval_or_generation_rerun'] is True
        assert evidence['source_code_sha256'] == sha(native_source)
        adapter = ROOT / 'scripts/evaluation' / ADAPTERS[method]
        assert evidence['adapter_sha256'] == sha(adapter)
        for name, expected in evidence['preserved_input_sha256'].items():
            original = Path(name)
            assert sha(original) == expected
            assert sha(snapshot / original.relative_to(target)) == expected
            unchanged_files += 1
            verified_snapshot_files += 1
        for name, expected in evidence['output_sha256'].items():
            assert sha(name) == expected
        protocol = read(target / 'protocol.json')
        for name, expected in protocol['open_evaluation']['source_hashes'].items():
            assert sha(name) == expected, 'MCQ memory source changed: ' + name
        questions = rows(target / 'questions.jsonl')
        retrieved = rows(target / 'retrieval.jsonl')
        manifest = rows(target / 'reader_manifest.jsonl')
        assert len(questions) == len(retrieved) == len(manifest) == 371
        assert len({q['id'] for q in questions}) == 371
        qmap, rmap, mmap = ({r['id']: r for r in values} for values in (questions, retrieved, manifest))
        assert set(qmap) == set(rmap) == set(mmap)
        actual_mismatches = []
        for identity, q in qmap.items():
            row, reader = rmap[identity], mmap[identity]
            fields = [key for key, value in q.items() if row[key] != value]
            assert not fields or fields == ['retrieval_query']
            assert row['retrieval_query'] == query(q)
            assert row['context'] == row['retrieved_context']
            assert row['context_sha256'] == digest(row['context'])
            assert reader['context'] == row['context'] and reader['retrieval_detail'] == row['retrieval_detail']
            messages = [{'role': 'system', 'content': reader['system']},
                {'role': 'assistant', 'content': 'Retrieved memory context:\n' + row['context']},
                {'role': 'user', 'content': row['retrieval_query']}]
            assert reader['reader_messages'] == messages
            assert reader['payload_sha256'] == reader['reader_messages_sha256'] == digest(messages)
            assert reader['user_text'] == row['retrieval_query']
            assert reader['source_query_image_paths'] == q['query_image_paths'] and reader['query_image_paths'] == []
            if fields:
                actual_mismatches.append({'id': identity, 'fields': fields,
                    'preserved_frozen_query_sha256': digest(q['retrieval_query']),
                    'validated_actual_deepseek_query_sha256': digest(row['retrieval_query'])})
        assert {row['id'] for row in actual_mismatches} == EXPECTED_IDS
        assert actual_mismatches == evidence['mismatches']
        if shared_mismatches is None:
            shared_mismatches = actual_mismatches
        else:
            assert actual_mismatches == shared_mismatches
        marker = read(target / 'complete.json')
        assert marker == evidence['native_marker'] and marker['passed'] is True and marker['questions'] == 371
        assert marker['protocol_sha256'] == sha(target / 'protocol.json')
        assert marker['retrieval_sha256'] == sha(target / 'retrieval.jsonl')
        assert marker['reader_manifest_sha256'] == sha(target / 'reader_manifest.jsonl')
        tail_sha = sha(folder / 'adapted_native_tail.py')
        tails.add(tail_sha)
        methods.append({'method': method, 'passed': True, 'questions': 371,
            'recovery_evidence_path': str(path), 'recovery_evidence_sha256': sha(path),
            'cpu_adapter_path': str(adapter), 'cpu_adapter_sha256': sha(adapter),
            'base_recovery_adapter_sha256': evidence.get('base_recovery_adapter_sha256', sha(adapter)),
            'native_source_sha256': evidence['source_code_sha256'], 'adapted_native_tail_sha256': tail_sha,
            'preserved_input_sha256': evidence['preserved_input_sha256'],
            'original_failure_snapshot': str(snapshot), 'snapshot_bytes_match_every_preserved_input': True,
            'unchanged_mcq_memory_source_sha256': protocol['open_evaluation']['source_hashes'],
            'new_completion_output_sha256': evidence['output_sha256'],
            'native_retrieval_sha256_before': evidence['preserved_input_sha256'][str(target / 'retrieval.jsonl')],
            'native_retrieval_sha256_after': sha(target / 'retrieval.jsonl'),
            'native_contexts_preserved_in_all_reader_payloads': 371,
            'replayed_original_call_events': marker['replayed_call_events'],
            'preexisting_native_open_call_settings': marker['new_call_settings'],
            'api_calls_by_cpu_recovery': 0, 'gpu_model_loads_by_cpu_recovery': 0,
            'native_retrieval_or_generation_rerun': False,
            'held_parent_zombie_child': evidence.get('held_parent_zombie_child'),
            'same_method_active_processes_at_cpu_recovery': evidence['same_method_active_processes'],
            'independent_mcq_attribution_processes': evidence.get('independent_mcq_attribution_processes', [])})
    assert len(tails) == 1
    api_helpers = [ROOT / 'scripts/evaluation' / n for n in (
        'run_memeye_deepseek1024_text_open_recovery.py',
        'run_memeye_deepseek1024_simplemem_open_recovery.py')]
    builder_snapshot = DIAGNOSTIC / 'cpu_recovery_summary_builder_snapshot.py'
    shutil.copyfile(__file__, builder_snapshot)
    result = {'schema': 'memeye-deepseek-cpu-open-recovery-summary-v1', 'status': 'passed', 'passed': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'methods_verified': 4,
        'reader_records_verified': 1484, 'preserved_native_input_files_verified': unchanged_files,
        'failure_snapshot_files_verified': verified_snapshot_files,
        'questions_per_method': 371, 'shared_stale_derived_field_records_per_method': 8,
        'shared_mismatches': shared_mismatches, 'same_adapted_native_tail_for_all_four': True,
        'adapted_native_tail_sha256': next(iter(tails)), 'methods': methods,
        'api_wrapper_source_sha256': {str(p): sha(p) for p in api_helpers},
        'source_sha256': {str(p): sha(p) for p in (Path(__file__), builder_snapshot, OUT / 'plan.json',
            native_source, ROOT / 'scripts/memory/text_memory_baselines.py', ROOT / 'scripts/memory/reflexion_memory.py')},
        'api_calls_by_summary_builder': 0, 'gpu_model_loads_by_summary_builder': 0,
        'api_calls_by_cpu_recoveries': 0, 'gpu_model_loads_by_cpu_recoveries': 0,
        'native_results_regenerated': False, 'original_pinned_source_files_changed': False,
        'main_progress_written': False,
        'approved_boundary': 'Preserve original signed protocol and question bytes, including the eight stale Qwen-derived retrieval_query fields. Exclude only that field from frozen-field equality. Require every actual query to equal the original native derivation from the frozen question and fixed DeepSeek captions; preserve all native contexts, retrieval output bytes, source checks, context bounds, and original completion audits. Create only missing reader manifest and native completion outputs.',
        'historical_failures_remain': True,
        'api_call_scope_note': 'Zero counts refer only to CPU recovery and this audit. Preexisting native Open retrieval API calls are retained and separately recorded per method; subsequent answer/Judge processes are outside this audit.'}
    output = DIAGNOSTIC / 'cpu_recovery_summary.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps({'passed': True, 'methods': 4, 'reader_records': 1484,
        'unchanged_native_files': unchanged_files, 'snapshot_files': verified_snapshot_files,
        'summary_path': str(output), 'summary_sha256': sha(output)}))


if __name__ == '__main__':
    main()
