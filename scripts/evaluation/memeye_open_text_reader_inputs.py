"""Preserve the native text-baseline reader roles and caption-only query images."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from scripts.evaluation.memeye_open_text_retrieval import ROOT, OUT, SOURCES, read
from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM
from scripts.memory import text_memory_baselines as base


def prepare(method):
    target = OUT / method
    source = ROOT / SOURCES[method][0]
    p = read(source / 'protocol.json')
    assert p['query_image_policy'] == 'cached Qwen captions only, no native images'
    old = {r['id']: r for r in base.read_rows(source / 'qa_responses.jsonl')}
    rows = base.read_rows(target / 'retrieval.jsonl')
    assert len(rows) == 371 and len({r['id'] for r in rows}) == 371
    evidence, result = {}, []
    for r in rows:
        qold = old[r['id']]
        path = source / 'calls' / (qold['call_key'] + '.json')
        request = read(path)['request']
        assert base.hash_object(request) == path.stem
        expected = base.reader_messages(qold, qold['retrieved_context'], p)
        assert request['messages'] == expected, r['id']
        evidence[str(path)] = base.sha(path)
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'assistant', 'content': 'Retrieved memory context:\n' + r['context']},
                    {'role': 'user', 'content': r['retrieval_query']}]
        result.append(dict(r, system=SYSTEM, user_text=r['retrieval_query'], reader_messages=messages,
            source_query_image_paths=r['query_image_paths'], query_image_paths=[],
            query_image_policy='Cached Qwen query-image captions in user text; no native query image, same as original text baseline.',
            reader_messages_sha256=base.hash_object(messages), payload_sha256=base.hash_object(messages)))
    base.write_rows(target / 'reader_manifest.jsonl', result)
    base.write_json(target / 'reader_input_audit.json', {'questions': len(result), 'method': method,
        'original_mcq_correct': SOURCES[method][1], 'all_original_reader_messages_verified': True,
        'reader_roles': ['system', 'assistant', 'user'], 'system': SYSTEM,
        'original_question_image_questions': sum(bool(r['source_query_image_paths']) for r in result),
        'native_reader_images': 0, 'caption_only_question_images_preserved': True,
        'source_hashes': evidence, 'retrieval_sha256': base.sha(target / 'retrieval.jsonl'),
        'manifest_sha256': base.sha(target / 'reader_manifest.jsonl'),
        'script_sha256': base.sha(Path(__file__))})
    print(method, len(result), 'native text reader inputs verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(SOURCES), required=True)
    prepare(parser.parse_args().method)
