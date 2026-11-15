"""Download an official, pinned VL2 Small snapshot without loading model code."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path('/storage/hewenwen/VisualMemory')
DEST = ROOT / 'models/deepseek-vl2-small'
REPORT = ROOT / 'reports/model_downloads/deepseek-vl2-small'
REPO = 'deepseek-ai/deepseek-vl2-small'
ENDPOINT = 'https://huggingface.co'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(name, value):
    path = REPORT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT / 'source_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['repo_id'] == REPO and manifest['endpoint'] == ENDPOINT
    else:
        info = HfApi(endpoint=ENDPOINT, token=False).model_info(REPO, files_metadata=True)
        manifest = {'repo_id': REPO, 'endpoint': ENDPOINT, 'revision': info.sha,
                    'resolved_at_utc': now(), 'local_dir': str(DEST), 'files': []}
        for sibling in info.siblings:
            entry = {'path': sibling.rfilename, 'size': sibling.size,
                     'git_blob_sha1': sibling.blob_id}
            if sibling.lfs is not None:
                entry['lfs_sha256'] = sibling.lfs.sha256
            manifest['files'].append(entry)
        manifest['expected_bytes'] = sum(item['size'] for item in manifest['files'])
        write('source_manifest.json', manifest)
    status = {'repo_id': REPO, 'revision': manifest['revision'], 'local_dir': str(DEST),
              'started_at_utc': now(), 'status': 'downloading', 'gpu_used': False,
              'expected_bytes': manifest['expected_bytes']}
    write('status.json', status)
    print(json.dumps(status), flush=True)
    disable_progress_bars()
    snapshot_download(repo_id=REPO, revision=manifest['revision'], endpoint=ENDPOINT,
                      local_dir=DEST, cache_dir=DEST / '.cache/hub', token=False,
                      max_workers=4, etag_timeout=30)
    status['status'] = 'verifying_hashes'
    write('status.json', status)
    verified = []
    for entry in manifest['files']:
        path = DEST / entry['path']
        assert path.is_file() and path.stat().st_size == entry['size'], entry['path']
        print('Verifying ' + entry['path'], flush=True)
        digest = hashlib.sha256()
        git_digest = hashlib.sha1(b'blob ' + str(entry['size']).encode() + b'\0')
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
                if 'lfs_sha256' not in entry:
                    git_digest.update(chunk)
        actual = digest.hexdigest()
        if 'lfs_sha256' in entry:
            assert actual == entry['lfs_sha256'], 'SHA-256 mismatch: ' + entry['path']
        else:
            assert git_digest.hexdigest() == entry['git_blob_sha1'], 'Git hash mismatch: ' + entry['path']
        verified.append({'path': entry['path'], 'size': entry['size'],
                         'sha256': actual, 'official_hash_match': True})
        write('verification.json', {'complete': False, 'files': verified})
    write('verification.json', {'complete': True, 'repo_id': REPO,
          'revision': manifest['revision'], 'verified_at_utc': now(),
          'total_bytes': sum(item['size'] for item in verified), 'files': verified})
    status.update(status='complete', completed_at_utc=now(), files_verified=len(verified))
    write('status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        description = re.sub(r'https?://\S+', '[url]', str(error))[:500]
        failure = {'status': 'failed', 'at_utc': now(),
                   'error_type': type(error).__name__, 'description': description}
        REPORT.mkdir(parents=True, exist_ok=True)
        write('last_error.json', failure)
        print(json.dumps(failure), flush=True)
        sys.exit(1)
