import collections
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parent
observations = json.loads((root / 'observations.json').read_text())
progress = {p.parent.name: json.loads(p.read_text()) for p in (root / 'memoryos').glob('*/progress.json')
            if p.parent.name in observations}
result = dict(status=json.loads((root / 'execution_status.json').read_text()),
    rounds=sum(len(observations[t]) if p['phase'] == 'complete' else p['done'] for t, p in progress.items()),
    total_rounds=3962, completed_tasks=sum(p['phase'] == 'complete' for p in progress.values()),
    memory_calls=len(list((root / 'memoryos/calls').glob('*.json'))),
    answers=len(list((root / 'memoryos/reader/responses').glob('*.json'))))
for stage in ('memoryos', 'memoryos/reader'):
    path = root / stage / 'api_attempts.jsonl'
    if path.exists():
        result[stage + '_attempts'] = dict(collections.Counter(json.loads(s)['status'] for s in path.read_text().splitlines()))
for name in ('memoryos/run_errors.json', 'memoryos/summary_normalization_events.jsonl'):
    path = root / name
    if path.exists():
        result[name] = len(path.read_text().splitlines())
result['last_progress_age_seconds'] = round(time.time() - max(p.stat().st_mtime for p in (root / 'memoryos').glob('*/progress.json')), 1)
print(json.dumps(result))
