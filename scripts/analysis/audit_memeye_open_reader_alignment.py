"""CPU-only protocol/template audit of the three Open ours readers vs SemRAG."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation import memeye_open_method_reader as generic

BASELINE = ROOT / 'results/memeye_qwen_semrag_open_20261105'
EXPERIMENT = ROOT / 'results/memeye_qwen_other_open_20261105'
METHODS = ('ours_1', 'ours_2', 'ours_1_ours_2')


def read(path):
    return json.loads(Path(path).read_text())


def verify_protocol(path):
    value = read(path)
    assert common.digest(value['contract']) == value['experiment_signature']
    return value


def main():
    old = verify_protocol(BASELINE / 'reader/protocol.json')['contract']
    service_protocol = verify_protocol(EXPERIMENT / 'qwen_service/protocol.json')
    service = service_protocol['contract']
    assert old['model'] == service['model']
    for key in ('seed', 'dtype', 'attention_implementation', 'device', 'input_truncation'):
        assert old[key] == service[key]
    assert old['query_image_max_pixels'] == service['image_max_pixels'] == common.QUERY_IMAGE_MAX_PIXELS == 1638400
    assert old['batch_size'] == 1 and old['max_new_tokens'] == 512 and old['do_sample'] is False
    assert old['seed'] == 42
    for hashes in (old['source_sha256'], service['implementation_sha256']):
        for path, expected in hashes.items():
            assert common.sha(path) == expected, path
    old_env_paths = list((BASELINE / 'reader').glob('environment_*.json'))
    assert len(old_env_paths) == 1
    old_environment, service_environment = read(old_env_paths[0]), read(EXPERIMENT / 'qwen_service/environment.json')
    for key in ('torch', 'transformers', 'device', 'gpu', 'dtype', 'context_limit',
                'generation_config', 'query_image_max_pixels'):
        assert old_environment[key] == service_environment[key], key

    # AutoProcessor loads only the tokenizer and image processor, never model weights.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(common.MODEL), local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': common.QUERY_IMAGE_MAX_PIXELS}
    baseline = common.by_id(common.read_lines(BASELINE / 'reader_manifest.jsonl'))
    assert len(baseline) == 371
    checks = {}
    sources = [Path(__file__), Path(generic.__file__), Path(common.__file__),
        ROOT / 'scripts/evaluation/memeye_open_semrag_reader.py',
        ROOT / 'scripts/evaluation/omni_qwen_transport.py',
        BASELINE / 'reader/protocol.json', BASELINE / 'reader_manifest.jsonl',
        EXPERIMENT / 'qwen_service/protocol.json', EXPERIMENT / 'qwen_service/environment.json', *old_env_paths]
    for method in METHODS:
        root = EXPERIMENT / 'ours' / method
        protocol = verify_protocol(root / 'reader/protocol.json')
        contract = protocol['contract']
        assert contract['method_id'] == method
        assert contract['service_signature'] == service_protocol['experiment_signature']
        assert contract['answer_model'] == old['model']['repo']
        for key in ('max_new_tokens', 'seed', 'do_sample', 'batch_size'):
            assert contract[key] == old[key]
        assert contract['image_sha256'] == old['query_image_sha256']
        for path, expected in contract['source_sha256'].items():
            assert common.sha(path) == expected, path
        rows = common.read_lines(root / 'reader_manifest.jsonl')
        assert len(rows) == 371 and set(common.by_id(rows)) == set(baseline)
        equal_contexts = image_questions = 0
        for row in rows:
            control = baseline[row['id']]
            assert 'reader_messages' not in row
            assert row['system'] == control['system']
            assert row['question'] == control['question']
            assert row['query_image_paths'] == control['query_image_paths']
            assert len(row['top_memory_ids']) == 5
            assert row['user_text'] == row['context'] + '\n\nQuestion:\n' + row['question'] + '\n\nAnswer concisely.'
            assert row['context'].startswith('Retrieved dialogue and image descriptions (ranked by relevance):\n\n')
            if row['top_memory_ids'] == control['top_memory_ids']:
                assert row['context'] == control['context'] and row['user_text'] == control['user_text']
                assert row['payload_sha256'] == control['payload_sha256']
                equal_contexts += 1
            native_old = [{'role': 'system', 'content': row['system']},
                {'role': 'user', 'content': [{'type': 'image'} for _ in row['query_image_paths']]
                 + [{'type': 'text', 'text': row['user_text']}]}]
            native_new = copy.deepcopy(generic.messages(row))
            for message in native_new:
                if isinstance(message['content'], list):
                    for part in message['content']:
                        if part['type'] == 'image':
                            part.pop('path'); part.pop('sha256')
            assert native_new == native_old
            old_rendered = processor.apply_chat_template(native_old, tokenize=False, add_generation_prompt=True)
            new_rendered = processor.apply_chat_template(native_new, tokenize=False, add_generation_prompt=True)
            assert new_rendered == old_rendered
            image_questions += bool(row['query_image_paths'])
        sources += [root / 'reader/protocol.json', root / 'reader_manifest.jsonl']
        checks[method] = {'questions': len(rows), 'identical_native_and_rendered_templates': len(rows),
            'same_top5_as_baseline_has_identical_context_and_payload': equal_contexts,
            'question_image_questions': image_questions, 'top_k': 5}
    report = {'passed': True, 'audit_scope': 'Frozen protocol, runtime environment and CPU-rendered chat templates; no new answers or API calls.',
        'methods': checks, 'same_model_files_and_revision': True,
        'same_runtime_package_versions': True, 'same_generation_config': True,
        'same_query_image_paths_and_hashes': True, 'same_system_and_context_format': True,
        'same_native_message_roles_and_rendered_tokens_input': True,
        'model': old['model']['repo'], 'model_revision': old['model']['revision'],
        'seed': 42, 'do_sample': False, 'max_new_tokens': 512, 'batch_size': 1,
        'dtype': old['dtype'], 'attention_implementation': old['attention_implementation'],
        'image_processor_size': processor.image_processor.size, 'query_image_max_pixels': 1638400,
        'input_truncation': False, 'same_load_runtime_function': str(Path(common.__file__)),
        'raw_answer_handling': 'Both paths strip decoded text, preserve empty/truncated answers, and use the same EOS IDs and token-limit rule.',
        'remaining_difference': 'Per-method retrieved Top5 evidence; shared service transport and request caching preserve identical wire inputs.',
        'source_sha256': {str(p.resolve()): common.sha(p) for p in sources}}
    target = EXPERIMENT / 'audits/reader_alignment_20261105.json'
    common.dump(target, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
