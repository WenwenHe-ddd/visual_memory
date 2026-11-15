"""Resume Gallery GPT-5 GA with byte-preserving WebP/static-GIF MIME support.

Some upstream files have .jpg extensions but contain other supported formats.
JPEG/PNG requests remain byte-for-byte identical to the initial run, allowing
only its own verified caption cache to be reused. No image is transcoded.
"""
import base64
import hashlib
from pathlib import Path
import sys

from scripts.evaluation import mem_gallery_gpt5_gen_agents as run

ORIGINAL_WIRE = run.wire_for
ORIGINAL_VERIFY = run.verify


def wire_for(payload, cap=2048):
    images = payload.get('query_images', [])
    parts = []
    for image in images:
        data = Path(image['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == image['sha256']
        if data.startswith(b'\xff\xd8\xff'):
            mime = 'image/jpeg'
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            mime = 'image/webp'
        elif data.startswith((b'GIF87a', b'GIF89a')):
            from PIL import Image
            with Image.open(image['path']) as decoded:
                assert getattr(decoded, 'n_frames', 1) == 1, 'Animated GIF requires an explicit frame policy'
            mime = 'image/gif'
        else:
            raise ValueError('Unsupported image signature: ' + image['path'])
        parts.append(dict(type='image_url', image_url=dict(
            url='data:' + mime + ';base64,' + base64.b64encode(data).decode(), detail='high')))
    wire = ORIGINAL_WIRE(dict(payload, query_images=[]), cap)
    wire['messages'][-1]['content'] = parts + wire['messages'][-1]['content']
    return wire


def verify(output):
    p = ORIGINAL_VERIFY(output)
    amendment = dict(kind='byte_preserving_image_mime_support',
        parent_experiment_signature=p['experiment_signature'],
        adapter_sha256=run.sha(Path(__file__)),
        image_policy='Sniff original bytes; accept JPEG, PNG, WebP and single-frame GIF. No transcoding.',
        reuse_policy='Only same-run cached outputs whose reconstructed request hashes match exactly.',
        jpeg_png_wire_requests_unchanged=True, captions_regenerated_for_all_unique_history_images=True)
    run.freeze(output / 'image_format_amendment.json', amendment)
    snapshot = output / 'source_snapshot' / (amendment['adapter_sha256'][:16] + '_' + Path(__file__).name)
    if snapshot.exists():
        assert run.sha(snapshot) == amendment['adapter_sha256']
    else:
        snapshot.write_bytes(Path(__file__).read_bytes())
    return p


if __name__ == '__main__':
    run.wire_for = wire_for
    run.verify = verify
    run.main()
