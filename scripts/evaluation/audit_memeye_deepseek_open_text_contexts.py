"""Check every saved Open reader context against its frozen native memory."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from scripts.evaluation import memeye_deepseek_open_text_retrieval as run


def native_file(name):
    path = run.base.VENDOR / 'memengine' / name
    spec = importlib.util.spec_from_file_location('open_context_audit_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(method):
    root = run.OUT / method
    protocol = run.read(root / 'protocol.json')
    source = Path(protocol['open_evaluation']['source_root'])
    rows = run.base.read_rows(root / 'retrieval.jsonl')
    manifests = run.base.read_rows(root / 'reader_manifest.jsonl')
    assert len(rows) == len(manifests) == 371
    original = {task: run.read(source / task / 'final_memory.json')
                for task in protocol['tasks']} if method != 'memoryos' else None
    if method == 'gen_agents':
        # Import the three pure formatter files directly, avoiding unrelated
        # package initialization of LLM clients and retrieval backends.
        ArgConfig = native_file('config/Config.py').ArgConfig
        ConcateUtilization = native_file('function/Utilization.py').ConcateUtilization
        LMTruncation = native_file('function/Truncation.py').LMTruncation
        cfg, model = run.base.method_config(protocol)
        config = run.base.helpers(method)['_build_ga_config'](
            SimpleNamespace(_empty_memory_text='No relevant memory retrieved.'), cfg, model)
        recall = ArgConfig(config).recall
        utilize, truncate = ConcateUtilization(recall.utilization), LMTruncation(recall.truncation)
    if method == 'simplemem':
        run.official.bootstrap()
        from simplemem.core.answer_generator import AnswerGenerator
        from simplemem.core.models.memory_entry import MemoryEntry
        formatter = object.__new__(AnswerGenerator)
        original = {task: {e['entry_id']: e for e in entries} for task, entries in original.items()}
    changed = 0
    for row, manifest in zip(rows, manifests):
        assert row['id'] == manifest['id']
        detail = row['retrieval_detail']
        if method == 'gen_agents':
            entries = [original[row['task']][i] for i in detail['context_memory_indices']]
            assert entries == detail['context_memories']
            context = truncate(utilize([entry['text'] for entry in entries]))
        elif method == 'memoryos':
            context = run.base.helpers(method)['_format_retrieval_context'](detail)
        elif method == 'a_mem':
            entries = [original[row['task']][i] for i in detail['context_memory_ids']]
            context = ''.join('talk start time:' + note['timestamp'] + 'memory content: ' + note['content']
                              + 'memory context: ' + note['context'] + 'memory keywords: ' + str(note['keywords'])
                              + 'memory tags: ' + str(note['tags']) + '\n' for note in entries)
        else:
            entries = [original[row['task']][i] for i in detail['context_memory_ids']]
            assert detail['items'] == [dict(entry, id=entry['entry_id']) for entry in entries]
            context = formatter._format_contexts([MemoryEntry(**entry) for entry in entries])
        assert context == row['context'] == row['retrieved_context'] == manifest['context']
        assert manifest['reader_messages'][1] == {
            'role': 'assistant', 'content': 'Retrieved memory context:\n' + context}
        assert manifest['reader_messages'][2] == {
            'role': 'user', 'content': run.base.retrieval_query(row)}
        assert manifest['query_image_paths'] == []
        changed += bool(row['query_image_paths'])
    report = {'status': 'passed', 'method': method, 'questions': 371,
              'all_contexts_reconstructed_from_native_memory': True,
              'all_reader_roles_and_question_image_caption_queries_verified': True,
              'question_image_questions': changed,
              'source_protocol_sha256': run.base.sha(source / 'protocol.json'),
              'retrieval_sha256': run.base.sha(root / 'retrieval.jsonl'),
              'reader_manifest_sha256': run.base.sha(root / 'reader_manifest.jsonl'),
              'audit_script_sha256': run.base.sha(Path(__file__))}
    run.base.write_json(root / 'independent_context_audit.json', report)
    print(method, '371 contexts reconstructed from frozen native memory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(run.SOURCES), required=True)
    audit(parser.parse_args().method)
