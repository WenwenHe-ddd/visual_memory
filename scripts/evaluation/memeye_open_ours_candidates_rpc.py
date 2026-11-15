"""Run unchanged MemGallery candidate generation through shared local Qwen."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation.memeye_open_ours_inputs import DEFAULT, prepare
from scripts.evaluation.memeye_open_qwen_service import RPCQwen, DEFAULT_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    prepare(args.root)
    client = RPCQwen(args.url)
    assert client.environment['device'] == 'cuda:0'
    original_ensure = common.ensure_protocol
    paths = [Path(__file__), ROOT / 'scripts/evaluation/memeye_open_qwen_service.py',
             ROOT / 'scripts/evaluation/omni_qwen_transport.py',
             args.root.parent / 'qwen_service/protocol.json']
    transport = {'kind': 'serialized_shared_local_qwen', 'url': args.url,
                 'service_signature': client.service_signature,
                 'source_sha256': {str(p): common.sha(p) for p in paths}}

    def ensure(path, contract):
        return original_ensure(path, {**contract, 'transport': transport})

    def load(device):
        assert device == client.environment['device']
        return client, None, {**client.environment, 'rpc_transport': transport}

    def generate(runtime, rows, cap):
        assert runtime[0] is client
        assert not any(row.get('query_image_paths') for row in rows)
        conversations = [[{'role': 'system', 'content': row['system']},
                          {'role': 'user', 'content': [{'type': 'text', 'text': row['user_text']}]}]
                         for row in rows]
        started = time.perf_counter()
        outputs = client.generate(conversations, cap, sampling_temperature=0.)
        elapsed = time.perf_counter() - started
        return [{**row, 'batch_elapsed_seconds': elapsed} for row in outputs]

    common.ensure_protocol, common.load_runtime, common.generate_batch = ensure, load, generate
    report = common.generate_candidates(args.root, device='cuda:0', batch_size=args.batch_size, max_attempts=3)
    print(report, flush=True)


if __name__ == '__main__':
    main()
