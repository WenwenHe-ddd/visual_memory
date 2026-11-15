"""Continue unblocked scenarios; never resubmit the provider-rejected scenario.

This produces partial memory artifacts only. A full score is not available
while a scenario remains blocked, and no missing answer is manufactured.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from scripts.evaluation import recover_mem_gallery_amem_api as recovery

run = recovery.run
BLOCKED = {'Astronomy_Physics_Scientific_Experiments_Cosmology'}


class AvailableExecutor(ThreadPoolExecutor):
    def submit(self, fn, name):
        if name in BLOCKED:
            future = Future()
            future.set_exception(RuntimeError('Scenario blocked by provider data_inspection_failed; not resubmitted'))
            return future
        return super().submit(fn, name)


def main():
    root = run.OUTPUT / 'deepseek'
    run.gpt.verify(root)
    errors = run.read(root / 'memory/run_errors.json')
    assert any(e['task'] in BLOCKED and 'API request was rejected' in e['traceback'] for e in errors)
    attempts = run.rows(root / 'memory/api_attempts.jsonl')
    rejected = [r for r in attempts if r.get('provider_error_code') == 'data_inspection_failed']
    assert len(rejected) == 1
    run.freeze(root / 'available_tasks_amendment.json', dict(
        script_sha256=run.sha(__file__), base_protocol_sha256=run.sha(root / 'protocol.json'),
        recovery_script_sha256=run.sha(recovery.__file__),
        blocked_tasks=sorted(BLOCKED), rejection=rejected[0],
        prior_errors_sha256=run.sha(root / 'memory/run_errors.json'),
        policy='Continue other scenarios using original cache and algorithm; blocked scenario never resubmitted; no full-dataset score.'))
    run.runner.ThreadPoolExecutor = AvailableExecutor
    try:
        recovery.main()
    finally:
        complete = [f.parent.name for f in (root / 'memory').glob('*/complete.json')]
        run.dump(root / 'partial_execution_status.json', dict(status='partial',
            completed_tasks=sorted(complete), blocked_tasks=sorted(BLOCKED), full_dataset_score_available=False))


if __name__ == '__main__':
    main()
