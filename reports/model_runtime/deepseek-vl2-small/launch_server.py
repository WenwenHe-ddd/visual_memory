"""Run the verified local VL2 checkpoint on the specifically authorized GPU."""
import os
from pathlib import Path
import subprocess

ROOT = Path('/storage/hewenwen/VisualMemory')
HERE = Path(__file__).resolve().parent
PYTHON = '/storage/hewenwen/NTU Decentralized Agent/.venv-vllm/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-bafd4aa2-2d61-3085-6b3b-89d320c2982c',
           PYTHONPATH=str(ROOT / '.vendor/vl2_runtime'), HF_HUB_OFFLINE='1',
           HF_HUB_DISABLE_TELEMETRY='1', VLLM_USE_FLASHINFER_SAMPLER='0')
args = [PYTHON, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(ROOT / 'models/deepseek-vl2-small'),
        '--served-model-name', 'deepseek-vl2-small', '--host', '127.0.0.1',
        '--port', '8312', '--dtype', 'bfloat16', '--max-model-len', '4096',
        '--gpu-memory-utilization', '0.65', '--max-num-seqs', '1',
        '--max-num-batched-tokens', '4096', '--limit-mm-per-prompt', '{"image":1}',
        '--enforce-eager', '--seed', '42', '--chat-template', str(HERE / 'chat_template.jinja'),
        '--chat-template-content-format', 'string', '--generation-config', 'vllm',
        '--hf-overrides', (HERE / 'hf_overrides.json').read_text()]
if __name__ == '__main__':
    raise SystemExit(subprocess.call(args, env=env, cwd=ROOT))
