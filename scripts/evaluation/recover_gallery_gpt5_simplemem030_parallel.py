"""Use an idle task slot for the diagnosed Baking-only format recovery.

Only Baking state is mutated. API records use a separate stream until the
original pool stops; final merge and sealing are performed by the recovery
controller. All original inference settings and request cache keys are retained.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.evaluation import recover_gallery_gpt5_simplemem030 as recovery

run = recovery.run
TASK = 'Baking_Dessert_Daily_Life_Skill'
OriginalModel = run.mos.Model
original_dump = run.dump


class Model(OriginalModel):
    def __init__(self, root, env_file, expected_model):
        super().__init__(Path(root)/'recovery_api', env_file, expected_model)


def scoped_dump(path, value):
    path = Path(path)
    if path.parent == run.OUTPUT/'memory' and path.name in ('run_errors.json', 'generation_environment.json'):
        path = path.parent/'recovery_api'/path.name
    return original_dump(path, value)


def main():
    output = run.OUTPUT
    p = run.verify(output)
    assert run.read(output/'execution.json')['status'] == 'running'
    errors = run.read(output/'memory/run_errors.json')
    assert {e['task'] for e in errors} == {TASK}
    assert not (output/'memory'/TASK/'complete.json').exists()
    active = []
    for f in (output/'memory').glob('*/progress.json'):
        if f.parent.name != TASK and run.read(f)['phase'] != 'complete':
            active.append(f.parent.name)
    assert len(active) < p['task_workers']
    stream = output/'memory/recovery_api'
    stream.mkdir(exist_ok=True)
    proof = dict(status='running', task=TASK, other_active_tasks=active,
        concurrency_bound=len(active)+1, protocol_task_workers=p['task_workers'],
        source_sha256=run.sha(__file__), repair_source_sha256=run.sha(recovery.__file__),
        experiment_signature=p['experiment_signature'],
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        policy='Use one vacant task slot. Preserve all inference settings; Baking-only state; separate API log stream; exact request cache.')
    original_dump(output/'parallel_format_recovery.json', proof)
    run.mos.Model = Model
    run.SimpleMem = recovery.RepairedMemory
    run.dump = scoped_dump
    try:
        run.memory_stage(output, run.ga.DEFAULT_ENV, TASK)
    except BaseException:
        proof.update(status='failed', ended_at_utc=datetime.now(timezone.utc).isoformat())
        original_dump(output/'parallel_format_recovery.json', proof)
        raise
    proof.update(status='complete', ended_at_utc=datetime.now(timezone.utc).isoformat(),
        complete_manifest_sha256=run.sha(output/'memory'/TASK/'complete.json'),
        stream_sha256={str(f):run.sha(f) for f in stream.glob('*') if f.is_file()})
    original_dump(output/'parallel_format_recovery.json', proof)
    print('Baking recovery complete; original pool and log streams remain separate', flush=True)


if __name__ == '__main__':
    main()
