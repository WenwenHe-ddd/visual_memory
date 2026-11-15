"""Preserve incomplete CPU output, then reproduce source on its original CUDA device."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_open_caption_inputs as src
root = src.DEFAULT
marker = root / 'encoding_device_recovery.json'
archive = root / 'cpu_incomplete_embeddings'
if marker.exists():
    p = src.read(marker)
    src.native.verify_hashes(p['archived_artifact_sha256'])
else:
    original = root / 'embeddings'
    assert not (root / 'encoding_validation.json').exists()
    assert original.exists() and not archive.exists()
    log = root / 'encoding_run.log'
    assert "AssertionError: ('Cartoon_Entertainment_Companion/COMIC_Q35', 'baseline_question_only_zscore')" in log.read_text()
    original.rename(archive)
    p = {'reason': 'CPU re-encoding reproduced raw ranks but a source full-domain zscore ranking failed at Cartoon COMIC_Q35. Preserve partial CPU vectors and re-encode all tasks on the original source CUDA device; require exact full-rank reproduction for every condition.',
         'original_source_device': 'cuda:1', 'recovery_device': 'cuda:1',
         'source_script_sha256': src.sha(src.__file__), 'recovery_script_sha256': src.sha(__file__),
         'archived_artifact_sha256': {str(p): src.sha(p) for p in [log, *archive.glob('*.npz')]}}
    src.freeze(marker, p)
src.encode(root, 'cuda:1')
src.freeze(root / 'encoding_device_recovery_completion.json', {'passed': True,
    'recovery_protocol_sha256': src.sha(marker), 'encoding_validation_sha256': src.sha(root / 'encoding_validation.json'),
    'exact_source_MCq_rankings_required': 1484})
