"""Audit frozen sources, Open inputs, native bounds and retrieval-only LLM calls."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from pathlib import Path
from scripts.evaluation.memeye_open_text_retrieval import ROOT, OUT, SOURCES, QUESTIONS, read
from scripts.evaluation.simplemem_reflection_ablation import operation
from scripts.memory import text_memory_baselines as base


def audit(method):
    dest = OUT / method
    p = read(dest / 'protocol.json')
    done = read(dest / 'complete.json')
    assert done['questions'] == 371
    assert done['retrieval_sha256'] == base.sha(dest / 'retrieval.jsonl')
    assert done['protocol_sha256'] == base.sha(dest / 'protocol.json')
    for path, digest in p['open_evaluation']['source_hashes'].items():
        assert base.sha(Path(path)) == digest, path
    for path, digest in read(dest / 'retrieval_implementation.json')['code_sha256'].items():
        assert base.sha(Path(path)) == digest, path
    source = ROOT / SOURCES[method][0]
    old = {r['id']: r for r in base.read_rows(source / 'qa_responses.jsonl')}
    questions = {r['id']: r for r in base.read_rows(QUESTIONS)}
    rows = base.read_rows(dest / 'retrieval.jsonl')
    assert {r['id'] for r in rows} == set(questions)
    counts = Counter()
    for row in rows:
        q = questions[row['id']]
        assert row['question'] == q['question']
        assert row['query_image_captions'] == q['query_image_captions']
        assert row['retrieval_query'] == base.retrieval_query(q)
        assert row['context_sha256'] == base.hash_object(row['context'])
        assert not any(k in row for k in ('options', 'gold_choice', 'gold_ids', 'ground_truth'))
        detail = row['retrieval_detail']
        if method in ('a_mem', 'simplemem', 'gen_agents'):
            count = detail['context_memory_count']
            counts[count] += 1
            if method == 'gen_agents':
                assert count == 10
            elif method == 'a_mem':
                assert count == 5 and len(set(detail['context_memory_ids'])) == 5
            else:
                assert count == 5 or (count == 0 and detail['source_empty_preserved'])
                assert len(set(detail['context_memory_ids'])) == count
        else:
            counts[len(detail['retrieved_pages'])] += 1
            assert len(detail['retrieved_pages']) <= 10
            assert len(detail['retrieved_user_knowledge']) <= 20
            assert len(detail['retrieved_assistant_knowledge']) <= 20
    types = Counter()
    for path in (dest / 'calls').glob('*.json'):
        call = read(path)
        assert base.hash_object(call['request']) == path.stem
        req = call['request']
        assert req['kind'] == 'memory' and req['seed'] == 42
        if method == 'a_mem':
            users = [m['content'] for m in req['messages'] if m['role'] == 'user']
            assert len(users) == 1 and users[0].startswith('Given the following question, generate several keywords')
            assert req['max_new_tokens'] == 1000
            types['query_keywords'] += 1
        elif method == 'simplemem':
            name = operation(req)
            assert name in ('information_plan', 'query_plan', 'query_analysis', 'reflection_assessment', 'reflection_queries')
            assert req['max_new_tokens'] == 16384
            types[name] += 1
        else:
            raise AssertionError('unexpected new LLM call')
    replayed = base.read_rows(dest / 'replayed_calls.jsonl') if (dest / 'replayed_calls.jsonl').exists() else []
    for row in replayed:
        assert base.sha(Path(row['source_path'])) == row['source_sha256']
    reader = base.read_rows(dest / 'reader_manifest.jsonl')
    assert len(reader) == 371
    for row in reader:
        assert [m['role'] for m in row['reader_messages']] == ['system', 'assistant', 'user']
        assert not row['query_image_paths']
        assert row['source_query_image_paths'] == questions[row['id']]['query_image_paths']
        assert row['reader_messages'][1]['content'] == 'Retrieved memory context:\n' + row['context']
        assert row['reader_messages'][2]['content'] == row['retrieval_query']
    report = {'status': 'complete', 'method': method, 'questions': 371,
        'source_mcq_correct': SOURCES[method][1], 'frozen_sources_verified': True,
        'memory_construction_new_llm_calls': 0, 'new_llm_calls': dict(types),
        'replayed_call_events': len(replayed), 'context_count_distribution': dict(counts),
        'open_queries_changed': sum(r['retrieval_query'] != old[r['id']]['retrieval_query'] for r in rows),
        'native_reader_query_image_policy': 'cached captions only',
        'original_question_image_questions': sum(bool(r['query_image_paths']) for r in rows),
        'reader_manifest_sha256': base.sha(dest / 'reader_manifest.jsonl'),
        'audit_script_sha256': base.sha(Path(__file__))}
    base.write_json(dest / 'validation_final.json', report)
    print(method, '371/371 validated', dict(types))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(SOURCES), required=True)
    audit(parser.parse_args().method)
