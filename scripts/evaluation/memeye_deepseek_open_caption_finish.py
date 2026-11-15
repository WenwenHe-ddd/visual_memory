"""Finish candidate-dependent DeepSeek Open retrieval and its independent audit."""
from pathlib import Path
import json
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_open_caption_inputs as src
root = src.DEFAULT
paths = [Path(__file__), Path(src.__file__), ROOT / 'scripts/analysis/audit_memeye_deepseek_open_caption_inputs_cuda.py']
src.freeze(root / 'dependent_stages_protocol.json', {'source_sha256': {str(p): src.sha(p) for p in paths},
    'device': 'cuda:1', 'candidate_marker': str(root / 'candidate_validation.json'),
    'stage_order': ['candidate validation', 'ours retrieval', 'independent CUDA audit']})
while not (root / 'candidate_validation.json').exists():
    time.sleep(5)
for log, command in [
    ('retrieval_ours.log', [sys.executable, '-u', 'scripts/evaluation/memeye_deepseek_open_caption_inputs.py', '--phase', 'retrieve', '--family', 'ours', '--device', 'cuda:1']),
    ('audit_ours.log', [sys.executable, '-u', 'scripts/analysis/audit_memeye_deepseek_open_caption_inputs_cuda.py', '--family', 'ours'])]:
    with (root / log).open('x') as out:
        subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
src.freeze(root / 'all_inputs_complete.json', {'passed': True, 'methods': list(src.SPECS), 'questions_per_method': 371,
    'artifact_sha256': {str(p): src.sha(p) for p in [root / 'independent_base_audit.json', root / 'independent_ours_audit.json', *[root / m / 'ready_for_reader.json' for m in src.SPECS]]}})
print(json.dumps(src.read(root / 'all_inputs_complete.json')), flush=True)
