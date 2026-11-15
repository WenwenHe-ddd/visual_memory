"""Provider-size adapter: keep original bytes except oversized image inputs.

Oversized files are JPEG encoded at the original resolution, starting at Q95
and decreasing in five-point steps only as needed to meet the Data URI limit.
Original files and CLIP inputs are never modified. Every substitution is hashed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageFile
from scripts.evaluation import gallery_deepseek_transport as api

ROOT = api.common.ROOT
OUTPUT = ROOT/'results/mem_gallery_deepseek_caption_visual_master_20261030_image_limits'
CACHE = OUTPUT/'api_image_inputs'
MAX_URI_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = (MAX_URI_BYTES-128)//4*3


def jpeg_to_limit(data, max_bytes=MAX_IMAGE_BYTES):
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert('RGB')
        # Pillow's estimate is too small for optimized 4:4:4 high-entropy JPEGs.
        # Reserving the full pixel buffer affects allocation, not encoded bytes.
        ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 4*image.width*image.height)
        for quality in range(95, 4, -5):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True, subsampling=0)
            encoded = buffer.getvalue()
            if len(encoded) <= max_bytes:
                return encoded, dict(quality=quality, subsampling=0, dimensions=list(image.size),
                                     resized=False, cropped=False, original_mode=source.mode)
    raise ValueError('Image could not fit without changing dimensions')


def prepare():
    source = api.common.DEFAULT_OUTPUT
    jobs = api.common.rows(source/'caption_jobs.jsonl')
    qs = api.common.rows(source/'inputs/questions.jsonl')
    paths = sorted({j['path'] for j in jobs} | {path for q in qs for path in q['query_image_paths']})
    CACHE.mkdir(parents=True, exist_ok=True)
    entries, oversized = {}, []
    for path in paths:
        data = Path(path).read_bytes()
        if len(data) <= MAX_IMAGE_BYTES: continue
        original_sha = hashlib.sha256(data).hexdigest()
        oversized.append(path)
        if original_sha in entries:
            entries[original_sha]['original_paths'].append(path)
            continue
        encoded, details = jpeg_to_limit(data)
        dest = CACHE/(original_sha+'.jpg')
        if dest.exists(): assert dest.read_bytes() == encoded
        else: dest.write_bytes(encoded)
        entries[original_sha] = dict(original_paths=[path], original_sha256=original_sha,
            original_bytes=len(data), path=str(dest), sha256=hashlib.sha256(encoded).hexdigest(),
            encoded_bytes=len(encoded), data_uri_bytes=23+4*((len(encoded)+2)//3), **details)
        assert entries[original_sha]['data_uri_bytes'] < MAX_URI_BYTES
        print('Prepared oversized image', len(entries), details, flush=True)
    value = dict(policy='Oversized only; same dimensions; JPEG Q95 then steps of five; no crop or resize; CLIP sees original bytes.',
        max_data_uri_bytes=MAX_URI_BYTES, max_image_bytes=MAX_IMAGE_BYTES,
        image_files=len(paths), oversized_file_count=len(oversized), unique_oversized_images=len(entries), entries=entries)
    api.common.freeze(CACHE/'manifest.json', value)
    return value


def wire_for(payload, cap=2048):
    manifest = api.common.read(CACHE/'manifest.json')
    mapped = []
    for image in payload.get('query_images', []):
        assert api.common.sha(image['path']) == image['sha256']
        entry = manifest['entries'].get(image['sha256'])
        if entry:
            assert image['path'] in entry['original_paths']
            mapped.append(dict(path=entry['path'], sha256=entry['sha256']))
        else:
            assert Path(image['path']).stat().st_size <= manifest['max_image_bytes']
            mapped.append(image)
    return api.wire_for(dict(payload, query_images=mapped), cap)


def audit():
    manifest = api.common.read(CACHE/'manifest.json')
    for key, entry in manifest['entries'].items():
        assert key == entry['original_sha256']
        for path in entry['original_paths']: assert api.common.sha(path) == key
        assert api.common.sha(entry['path']) == entry['sha256']
        data, details = jpeg_to_limit(Path(entry['original_paths'][0]).read_bytes())
        assert data == Path(entry['path']).read_bytes()
        assert all(entry[k] == v for k,v in details.items())
        with Image.open(entry['original_paths'][0]) as original, Image.open(entry['path']) as encoded:
            assert original.size == encoded.size and list(original.size) == entry['dimensions']
        assert entry['data_uri_bytes'] < MAX_URI_BYTES
    return dict(status='passed', unique_reencoded_images=len(manifest['entries']),
        reencoded_file_count=manifest['oversized_file_count'], dimensions_preserved=True,
        unchanged_files_keep_original_bytes=True, original_files_and_clip_inputs_unchanged=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','audit'))
    args=parser.parse_args()
    print(prepare() if args.stage=='prepare' else audit())
