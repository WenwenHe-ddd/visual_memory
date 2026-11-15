"""Resume Gallery GA with bounded retries for invalid reflection line counts.

The original native prompt and cached response are always tried first. A retry
adds only the already-required count and line format, never answers or labels.
Valid original outputs remain untouched. Raw failed and corrected calls persist.
"""
import argparse
import json
from pathlib import Path
import threading

from scripts.evaluation import mem_gallery_qwen_text_methods as run


def reflection_lines(reflector, input_dict, kind, owner, record):
    from langchain.prompts import PromptTemplate
    config = getattr(reflector.config, kind + '_prompt')
    count = getattr(reflector.config, kind + '_number')
    prompt = PromptTemplate(input_variables=config.input_variables, template=config.template).format(**input_dict)
    response = reflector.llm.fast_run(prompt)
    lines = [line for line in response.splitlines() if line.strip()]
    if len(lines) == count:
        return lines
    original = response
    for attempt in range(1, 4):
        instruction = (f'\n\nOutput format requirement: return exactly {count} {kind}s, '
            f'one complete {kind} on each of exactly {count} nonempty lines. '
            'No preamble, headings, Markdown fences, or explanation. Do not combine items into one line.')
        if attempt > 1:
            instruction += f' This is format retry {attempt}; the previous response had an invalid line count.'
        response = reflector.llm.fast_run(prompt + instruction)
        lines = [line for line in response.splitlines() if line.strip()]
        record(dict(owner=owner, kind=kind, attempt=attempt, expected=count,
                    original_response=original, retry_response=response, parsed_lines=len(lines), valid=len(lines) == count))
        if len(lines) == count:
            return lines
    raise ValueError('Reflection output still violates the required line count after three format retries')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    root = run.OUTPUT / 'gen_agents'
    original_install = run.native.install_runtime
    lock = threading.Lock()
    run.freeze(root / 'format_continuation_protocol.json', dict(
        parent_experiment_signature=run.read(root / 'protocol.json')['experiment_signature'],
        source_sha256={str(Path(__file__).resolve()): run.sha(Path(__file__))},
        policy='Try original native prompt first; preserve all valid outputs. On wrong line count only, retry up to 3 times with explicit original count/line instructions. No scoring or gold used.',
        original_errors_sha256=run.sha(root / 'run_errors.json')))

    def install(broker, protocol):
        encoder = original_install(broker, protocol)
        from memengine.function.Reflector import GAReflector

        def record(event):
            with lock:
                with (root / 'format_retry_events.jsonl').open('a') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')

        def questions(self, inputs):
            return reflection_lines(self, inputs, 'question', getattr(broker.local, 'owner', None), record)

        def insights(self, inputs):
            return reflection_lines(self, inputs, 'insight', getattr(broker.local, 'owner', None), record)

        GAReflector.self_ask, GAReflector.generate_insight = questions, insights
        return encoder

    run.native.install_runtime = install
    run.run(run.OUTPUT, 'gen_agents', args.device)
    run.dump(root / 'recovery.json', dict(status='complete', all_scenarios_completed=True,
        format_events_sha256=run.sha(root / 'format_retry_events.jsonl'),
        reader_complete_sha256=run.sha(root / 'reader/complete.json')))


if __name__ == '__main__':
    main()
