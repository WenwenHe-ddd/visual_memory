"""Driver for the GPT text Open evaluation with an explicit derived-field audit.

The shared canonical input carries a Qwen-derived retrieval_query cache. The
GPT run correctly reconstructs the query from question + GPT image captions.
Only the finalizer's comparison of that derived metadata needs normalization.
No frozen protocol, question file, retrieval or native call is altered.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
from pathlib import Path
from scripts.evaluation import memeye_gpt5_open_text_retrieval as experiment

native_finalize = experiment.finalize


def finalize(method, target, source, protocol, questions):
    changes, normalized = [], copy.deepcopy(questions)
    for q in normalized:
        expected = experiment.base.retrieval_query(q)
        if q['retrieval_query'] != expected:
            changes.append({'id': q['id'], 'original_cached_query_sha256': experiment.base.hash_object(q['retrieval_query']),
                            'gpt_derived_query_sha256': experiment.base.hash_object(expected)})
        q['retrieval_query'] = expected
    experiment.freeze(target / 'derived_query_audit_amendment.json', {
        'policy': 'Normalize only the finalizer comparison of canonical Qwen-derived query metadata to question plus original GPT-5 image captions; actual native retrieval already uses this exact reconstructed GPT query.',
        'affected_ids': changes, 'retrieval_recomputed': False, 'source_or_protocol_changed': False,
        'question_file_sha256': experiment.base.sha(target / 'questions.jsonl'),
        'protocol_sha256': experiment.base.sha(target / 'protocol.json'),
        'retrieval_sha256': experiment.base.sha(target / 'retrieval.jsonl'),
        'driver_sha256': experiment.base.sha(Path(__file__)),
        'native_script_sha256': experiment.base.sha(Path(experiment.__file__))})
    return native_finalize(method, target, source, protocol, normalized)


if __name__ == '__main__':
    experiment.finalize = finalize
    experiment.main()
