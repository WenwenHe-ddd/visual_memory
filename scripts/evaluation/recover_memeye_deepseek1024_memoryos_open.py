"""MemoryOS-only CPU recovery, allowing its absent optional replay log.

MemoryOS copies its native runtime and makes no construction replay events.
The executed general recovery source is preserved; only its evidence path
list is adapted here.  All native source/query/context checks stay intact.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import types

from scripts.evaluation import recover_memeye_deepseek1024_text_open as core

CORE_SHA256 = '55f1bf5799068ec9e11e4fabcc3d53bfcb5228392c2b514fb8e243e77c596569'


def recover():
    if core.suite.sha(core.__file__) != CORE_SHA256:
        raise ValueError('Executed base recovery source changed')
    tree = ast.parse(Path(core.__file__).read_text())
    audit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    index = next(i for i, n in enumerate(audit.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'paths' for t in n.targets))
    audit.body[index + 1:index + 1] = ast.parse(
        "paths = [path for path in paths if path.name != 'replayed_calls.jsonl' or path.is_file()]"
    ).body
    namespace = dict(vars(core), __file__=__file__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[audit], type_ignores=[])), __file__, 'exec'), namespace)
    function = types.FunctionType(core.recover.__code__, namespace, 'memoryos_recover')
    value = function('memoryos')
    value.update(base_recovery_adapter_sha256=CORE_SHA256,
        optional_replay_log_policy='Absent MemoryOS construction replay log is not fabricated or required. The original native completion audit already treats it as optional.')
    core.suite.dump(core.OUT / 'metrics/recovery_diagnostics/memoryos_open_failure_20261106/recovery_evidence.json', value)
    return value


if __name__ == '__main__':
    result = recover()
    print(json.dumps({k: result[k] for k in ('method', 'questions', 'different_frozen_fields', 'api_calls', 'gpu_model_loads')}))
