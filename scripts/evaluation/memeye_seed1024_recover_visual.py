"""Bounded HippoMM seed-1024 rewrite-format recovery and exact native resume.

Keep all completed answers, original malformed raw outputs and native three-
attempt validation. Only after an explicit rewrite word-count failure, request
2-5 individual words in a JSON array, join them, and apply the unchanged native
query parser. The model chooses every word from the original question; no gold,
choices, answer hints, hand-written replacement queries or relaxed parser enter
the adapter. Three distinct format-adapter requests are the fixed maximum.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_seed1024_visual as native

ERROR = 'rewrite: invalid output after 3 attempts: Visual query must contain 2-5 words'
MEMORY_FILES = ('memory.json', 'events.jsonl', 'dialogues.jsonl',
                'image_mapping.jsonl', 'visual_embeddings.npz')
sha, read, dump, digest = native.sha, native.read, native.dump, native.digest


def locations(args):
    baseline = args.output_root / 'hippomm_baseline/memeye'
    recovery = args.output_root / 'hippomm/recovery'
    return baseline, recovery


def copy_frozen(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise ValueError('Frozen recovery snapshot differs: ' + str(target))
    else:
        shutil.copyfile(source, target)


def prepare(args):
    native.verify_preflight(args)
    baseline, recovery = locations(args)
    path = recovery / 'protocol.json'
    if path.exists():
        return verify(args)
    questions = native.hip.inputs.unique(read(baseline / 'online/questions.jsonl'))
    responses = native.hip.inputs.unique(read(baseline / 'online/qa_responses.jsonl'))
    errors = read(baseline / 'online/question_errors.jsonl')
    latest = {r['id']: r for r in errors}
    missing = set(questions) - set(responses)
    if len(questions) != 371 or not missing or set(responses) - set(questions):
        raise ValueError('Expected a complete attempted cohort with missing native answers')
    if missing != set(latest) or any(latest[qid]['error'] != ERROR for qid in missing):
        raise ValueError('Only original native rewrite word-count failures are eligible')
    cfg = native.settings(args)
    cfg.memory_dir = baseline / 'offline'
    cfg.output = baseline / 'online'
    native.hip.load_run(cfg)
    if json.loads((cfg.output / 'protocol.json').read_text())['contract']['seed'] != 1024:
        raise ValueError('Recovery cannot import another seed')
    copy_frozen(cfg.output / 'qa_responses.jsonl', recovery / 'original_qa_responses.jsonl')
    copy_frozen(cfg.output / 'question_errors.jsonl', recovery / 'original_question_errors.jsonl')
    original_files = [Path(__file__), Path(native.__file__),
        recovery / 'original_qa_responses.jsonl', recovery / 'original_question_errors.jsonl',
        baseline / 'online/protocol.json', baseline / 'online/questions.jsonl',
        baseline / 'experiment_manifest.json',
        args.output_root / 'hippomm/stability_protocol.json']
    original_files += [baseline / 'offline' / name for name in MEMORY_FILES]
    cache_manifest = []
    for p in sorted((baseline / 'online/llm_cache').glob('call_*.json')):
        cache_manifest.append(dict(path=str(p), sha256=sha(p)))
    native.write(recovery / 'original_cache_manifest.jsonl', cache_manifest)
    original_files.append(recovery / 'original_cache_manifest.jsonl')
    for qid in sorted(missing):
        failure = latest[qid]
        for cid in failure['llm_calls']:
            call_path = baseline / 'online/llm_cache' / (cid + '.json')
            call = json.loads(call_path.read_text())
            if call['request']['role'] == 'rewrite':
                # Validate exact failure, without inventing a replacement query.
                try:
                    native.hip.core.parse_query(call['response']['raw_response'])
                except (ValueError, TypeError) as exc:
                    if '2-5 words' not in str(exc):
                        raise ValueError('Failure is not solely the original word-count constraint') from exc
                else:
                    raise ValueError('Recovery requested for a valid native rewrite')
    contract = dict(kind='seed1024_hippomm_word_array_format_recovery', seed=1024,
        evaluator_seed=42, method='hippomm', original_answer_count=len(responses),
        original_cached_calls=len(cache_manifest), original_missing_ids=sorted(missing),
        native_rewrite_attempts=3, max_new_adapter_requests_per_failed_query=3,
        eligibility='Only a native rewrite that exhausted three attempts with the exact whitespace word-count error.',
        adapter='Model returns 2-5 one-word strings in words array; join using one space, then unchanged native parse_query.',
        gold_options_in_adapter=False, old_answers_replaced=False,
        original_memory_rebuilt=False, original_seed42_caches_imported=False,
        open_policy='Apply the same bounded adapter only if a fresh Open query exhausts the unchanged three native attempts.',
        unchanged=['original question and memory', 'CLIP', 'native parsers', 'all thresholds',
                   'Top5', 'fast/final readers', 'representative images', 'seed1024', '1024 output-token cap'],
        input_sha256={str(p.resolve()): sha(p) for p in original_files})
    native.frozen_json(path, dict(contract=contract, experiment_signature=digest(contract)))
    print(json.dumps(dict(phase='prepared_recovery', original_answers=len(responses),
        missing_ids=sorted(missing), original_cached_calls=len(cache_manifest))), flush=True)
    return verify(args)


def verify(args):
    native.verify_preflight(args)
    baseline, recovery = locations(args)
    protocol = json.loads((recovery / 'protocol.json').read_text())
    contract = protocol['contract']
    if digest(contract) != protocol['experiment_signature'] or contract['seed'] != 1024:
        raise ValueError('Recovery protocol signature/seed changed')
    for name, expected in contract['input_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Recovery source changed: ' + name)
    for item in read(recovery / 'original_cache_manifest.jsonl'):
        if sha(item['path']) != item['sha256']:
            raise ValueError('Original native cached output changed')
    snapshot = (recovery / 'original_qa_responses.jsonl').read_bytes()
    if not (baseline / 'online/qa_responses.jsonl').read_bytes().startswith(snapshot):
        raise ValueError('Previously completed MCQ answers were changed')
    return protocol


def parse_words(raw):
    value = native.hip.core.parse_object(raw)
    if set(value) != {'words'} or not isinstance(value['words'], list) or not 2 <= len(value['words']) <= 5:
        raise ValueError('Expected only words: an array of 2-5 individual words')
    words = value['words']
    if any(not isinstance(w, str) or len(w.split()) != 1 or w != w.strip() for w in words):
        raise ValueError('Each words item must be exactly one nonempty whitespace-separated word')
    query = ' '.join(words)
    parsed = native.hip.core.parse_query(json.dumps({'visual_query': query}))
    return words, parsed


class Adapter:
    def __init__(self, args, protocol, stage):
        self.args, self.protocol, self.stage = args, protocol, stage
        self.baseline, self.recovery = locations(args)
        qs = (read(self.baseline / 'online/questions.jsonl') if stage == 'mcq'
              else native.open_questions())
        self.question_ids = {}
        for q in qs:
            self.question_ids.setdefault(q['question'], []).append(q['id'])
        self.original_structured = native.hip.core.structured_call
        self.used_ids = set()

    def __call__(self, llm, role, prompt, parse, attempts=3):
        try:
            return self.original_structured(llm, role, prompt, parse, attempts)
        except ValueError as exc:
            if role != 'rewrite' or str(exc) != ERROR:
                raise
            failure = str(exc)
        marker = '\nQuestion: '
        if marker not in prompt:
            raise ValueError('Native rewrite question boundary missing')
        question = prompt.split(marker, 1)[1]
        ids = self.question_ids.get(question)
        if not ids:
            raise ValueError('Rewrite question is outside the frozen cohort')
        if self.stage == 'mcq' and not set(ids) <= set(self.protocol['contract']['original_missing_ids']):
            raise ValueError('A completed MCQ answer is ineligible for recovery')
        log_path = self.recovery / (self.stage + '_adapter_attempts.jsonl')
        old = read(log_path) if log_path.exists() else []
        seen = {r['request_sha256']: r for r in old}
        previous, error = None, failure
        for attempt in (1, 2, 3):
            word_count = (3, 4, 2)[attempt - 1]
            repaired = ('Extract ONLY the concrete visual search subject from the following question. '
                'Do not answer the question, guess the requested answer, or invent visual attributes. '
                f'Return exactly one JSON object with only the key words and exactly {word_count} '
                'individual English word strings in its array. Each array item must contain exactly '
                'ONE nonempty whitespace-separated word, without spaces inside an item. '
                'Select the words yourself from the visual search subject. '
                'No proposed answer, explanation, or additional key is allowed.\n'
                f'FORMAT ADAPTER ATTEMPT {attempt} OF 3.\nQuestion: ' + question)
            if previous is not None:
                repaired += '\nPrevious invalid array: ' + previous + '\nValidation error: ' + error
            key = digest(dict(stage=self.stage, role=role, prompt=repaired, seed=1024, max_new_tokens=1024))
            raw = llm(role, repaired)
            try:
                words, query = parse_words(raw)
                error = None
            except (ValueError, TypeError) as exc:
                words, query, error = None, None, str(exc)
            record = dict(question_ids=ids, stage=self.stage, attempt=attempt,
                request_sha256=key, original_native_failure=failure, raw_response=raw,
                valid=error is None, validation_error=error, words=words, adapted_query=query,
                recovery_protocol_sha256=sha(self.recovery / 'protocol.json'))
            if key in seen:
                if seen[key] != record:
                    raise ValueError('Recovered cached adapter output differs')
            else:
                native.append(log_path, record)
            if error is None:
                self.used_ids.update(ids)
                return parse(json.dumps({'visual_query': query}))
            previous = raw
        raise ValueError('Bounded rewrite word-array adapter failed after three distinct requests: ' + error)


def recover_mcq(args, protocol):
    baseline, recovery = locations(args)
    adapter = Adapter(args, protocol, 'mcq')
    original_append = native.hip.append
    expected_output = baseline / 'online/qa_responses.jsonl'

    def annotated_append(path, row):
        if Path(path) == expected_output and row.get('id') in adapter.used_ids:
            row = {**row, 'format_recovery_protocol_sha256': sha(recovery / 'protocol.json'),
                   'format_recovery_kind': 'bounded_word_array_after_native_three_attempts'}
        return original_append(path, row)

    native.hip.core.structured_call = adapter
    native.hip.append = annotated_append
    try:
        cfg = native.settings(args)
        cfg.memory_dir, cfg.output = baseline / 'offline', baseline / 'online'
        native.hip.run(cfg)
        native.hip.analyze(cfg)
    finally:
        native.hip.core.structured_call = adapter.original_structured
        native.hip.append = original_append
    verify(args)
    # The original paired representative-image procedure consumes the now-
    # completed seed-1024 baseline; all calls and answers stay in this seed.
    cfg.output = args.output_root / 'hippomm/offline'
    native.hip_original.build(cfg)
    cfg.memory_dir, cfg.output = cfg.output, args.output_root / 'hippomm/online'
    native.hip_original.prepare(cfg)
    native.hip_original.run(cfg)
    native.hip_original.analyze(cfg)
    native.analyze(args)
    dump(recovery / 'mcq_completion.json', dict(status='complete', questions=371,
        original_answers_preserved=protocol['contract']['original_answer_count'],
        recovered_ids=sorted(adapter.used_ids), seed=1024,
        baseline_responses_sha256=sha(baseline / 'online/qa_responses.jsonl'),
        final_responses_sha256=sha(cfg.output / 'qa_responses.jsonl'),
        recovery_protocol_sha256=sha(recovery / 'protocol.json')))


def recover_open(args, protocol):
    _, recovery = locations(args)
    native.open_prepare(args)
    adapter = Adapter(args, protocol, 'open')
    native.hip.core.structured_call = adapter
    try:
        native.open_run(args)
    finally:
        native.hip.core.structured_call = adapter.original_structured
    dump(recovery / 'open_completion.json', dict(status='complete', questions=371, seed=1024,
        recovered_ids=sorted(adapter.used_ids), recovery_protocol_sha256=sha(recovery / 'protocol.json'),
        responses_sha256=sha(args.output_root / 'hippomm/open/qa_responses.jsonl')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'mcq', 'open', 'all'), required=True)
    parser.add_argument('--output-root', type=Path, default=native.DEFAULT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output_root, args.method = args.output_root.resolve(), 'hippomm'
    native.patch_runtime_seed()
    with (args.output_root / 'hippomm.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(args)
        if args.phase in ('mcq', 'all'):
            recover_mcq(args, protocol)
        if args.phase in ('open', 'all'):
            recover_open(args, protocol)


if __name__ == '__main__':
    main()
