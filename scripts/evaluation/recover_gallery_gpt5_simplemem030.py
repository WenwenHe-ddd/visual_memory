"""Resume one failed builder window with a hash-bound JSON quote repair.

The original GPT-5 response and every field value remain archived. Only one
single-quoted keyword literal is serialized with JSON double quotes. There is
no extra model generation, prompt change, or change to retrieval parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluation import mem_gallery_gpt5_simplemem030 as run

CALL = '45e8bca66166e50a934fb41e08b380df1ada14a0eccda4d370de259b5f42e7e9'
RESPONSE_SHA = '2dfe0e40b32a3efb81f0ab04e46bba293a77dd4deae28581659ce77123817239'
BEFORE = "'textured gray open-front coat'"
AFTER = '"textured gray open-front coat"'
OriginalMemory = run.SimpleMem


def repair(raw):
    assert hashlib.sha256(raw.encode()).hexdigest() == RESPONSE_SHA
    assert raw.count(BEFORE) == 1
    fixed = raw.replace(BEFORE, AFTER)
    entries = json.loads(fixed)
    assert isinstance(entries, list) and len(entries) == 10
    return fixed


class RepairedMemory(OriginalMemory):
    def __init__(self, protocol, workdir):
        super().__init__(protocol, workdir)
        original_parse = self.core.memory_builder._parse_llm_response

        def parse(response, ids):
            if hashlib.sha256(response.encode()).hexdigest() == RESPONSE_SHA:
                fixed = repair(response)
                result = original_parse(fixed, ids)
                run.dump(workdir/'json_quote_repair.json', dict(original_call=CALL,
                    original_response_sha256=RESPONSE_SHA, corrected_response=fixed,
                    before=BEFORE, after=AFTER, replacements=1, dialogue_ids=ids,
                    entry_ids=[e.entry_id for e in result]))
                return result
            return original_parse(response, ids)

        self.core.memory_builder._parse_llm_response = parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=run.OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    p = run.verify(output)
    # Recovery is only permitted after the original worker pool has stopped.
    assert run.read(output/'execution.json')['status'] == 'failed'
    parallel = output/'parallel_format_recovery.json'
    merge = None
    if parallel.exists():
        import time
        while run.read(parallel)['status'] == 'running':
            time.sleep(5)
        done = run.read(parallel)
        assert done['status'] == 'complete'
        assert done['repair_source_sha256'] == run.sha(__file__)
        assert done['source_sha256'] == run.sha(run.ROOT/'scripts/evaluation/recover_gallery_gpt5_simplemem030_parallel.py')
        assert done['concurrency_bound'] <= p['task_workers']
        assert done['complete_manifest_sha256'] == run.sha(output/'memory/Baking_Dessert_Daily_Life_Skill/complete.json')
        for name, digest in done['stream_sha256'].items():
            assert run.sha(name) == digest
        marker = output/'parallel_api_merge.json'
        if marker.exists():
            merge = run.read(marker)
            for name, entry in merge.items():
                assert run.sha(name) == entry['merged_sha256']
        else:
            merge = {}
            for name in ('api_attempts.jsonl', 'api_responses.jsonl'):
                target = output/'memory'/name
                source = output/'memory/recovery_api'/name
                prefix, addition = target.read_bytes(), source.read_bytes()
                # Validate the separate append-only streams before merging.
                before = [json.loads(s) for s in prefix.splitlines()]
                added = [json.loads(s) for s in addition.splitlines()]
                if name == 'api_responses.jsonl':
                    ids = [r['output']['response_id'] for r in before+added]
                    assert len(ids) == len(set(ids))
                target.write_bytes(prefix+addition)
                merge[str(target)] = dict(prefix_bytes=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                    source=str(source), source_sha256=run.sha(source), appended_records=len(added), merged_sha256=run.sha(target))
            run.freeze(marker, merge)
    call = output/'memory/calls'/(CALL+'.json')
    saved = run.read(call)
    fixed = repair(saved['output']['raw_response'])
    assert saved['output']['finish_reason'] == 'stop'
    retained = {t: run.sha(output/'memory'/t/'complete.json') for t in p['tasks']
                if (output/'memory'/t/'complete.json').exists()}
    proof = dict(kind='single_keyword_quote_serialization', source_sha256=run.sha(__file__),
        original_call=CALL, original_call_sha256=run.sha(call), original_response_sha256=RESPONSE_SHA,
        corrected_response_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
        before=BEFORE, after=AFTER, replacements=1, retained_complete_tasks=retained,
        policy='One hash-bound quote repair only. Preserve field values, original outputs, model, prompts, caps and retrieval settings.')
    if parallel.exists():
        proof.update(parallel_recovery_sha256=run.sha(parallel), api_merge_sha256=run.sha(output/'parallel_api_merge.json'))
    run.freeze(output/'json_quote_recovery_amendment.json', proof)
    errors = output/'memory/run_errors.json'
    if errors.exists():
        run.freeze(output/'initial_memory_errors.json', run.read(errors))
        errors.unlink()
    run.SimpleMem = RepairedMemory
    run.memory_stage(output, run.ga.DEFAULT_ENV)
    for task, digest in retained.items():
        assert run.sha(output/'memory'/task/'complete.json') == digest
    run.dump(output/'json_quote_recovery_complete.json', dict(status='complete',
        amendment_sha256=run.sha(output/'json_quote_recovery_amendment.json'),
        retained_complete_tasks=len(retained), original_call_unchanged=run.sha(call)==proof['original_call_sha256']))


if __name__ == '__main__':
    main()
