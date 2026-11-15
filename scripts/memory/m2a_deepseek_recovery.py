"""Explicit image compatibility amendment for failed DeepSeek M2A tasks."""
import base64
import hashlib
import io
from pathlib import Path

from PIL import Image
from scripts.memory import m2a_deepseek as original

ORIGINAL_IMAGE_PART = original.image_part
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_EDGE = 2048
POLICY = ('Unchanged bytes for supported static images <=8 MiB; larger images '
          'use RGB JPEG quality95, subsampling0, longest edge<=2048; MPO uses '
          'primary frame0 as lossless PNG, applying the same large-image rule '
          'if that PNG exceeds8 MiB. Other animations remain rejected. '
          'Original files and retrieval embeddings remain unchanged; '
          'source and transmitted bytes, dimensions and transformations logged.')


def image_part(block):
    ref = block.get('url') or block.get('image_url') or block.get('image')
    if isinstance(ref, dict):
        ref = ref['url']
    data = (base64.b64decode(ref.split(',', 1)[1], validate=True)
            if ref.startswith('data:') else Path(ref).read_bytes())
    with Image.open(io.BytesIO(data)) as im:
        if im.format != 'MPO' and len(data) <= MAX_SOURCE_BYTES:
            return ORIGINAL_IMAGE_PART(block)
        if im.format != 'MPO' and getattr(im, 'n_frames', 1) != 1:
            raise ValueError('Animations are not covered by this recovery policy')
        metadata = dict(source_bytes_sha256=hashlib.sha256(data).hexdigest(),
                        source_bytes=len(data), source_format=im.format,
                        source_dimensions=list(im.size),
                        source_frames=getattr(im, 'n_frames', 1))
        if not ref.startswith('data:'):
            metadata['path'] = ref
        changes = []
        if im.format == 'MPO':
            im.seek(0)
            frame = im.convert('RGB')
            buf = io.BytesIO()
            frame.save(buf, format='PNG')
            data, mime = buf.getvalue(), 'image/png'
            changes.append('MPO primary frame0 decoded to RGB PNG')
        else:
            frame = im.convert('RGB')
        if len(data) > MAX_SOURCE_BYTES:
            frame.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            frame.save(buf, format='JPEG', quality=95, subsampling=0, optimize=True)
            data, mime = buf.getvalue(), 'image/jpeg'
            changes.append('RGB JPEG quality95 subsampling0 longest edge<=2048')
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError('Image exceeds recovery payload limit after fixed preprocessing')
        metadata.update(bytes_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                        mime_type=mime, transmitted_dimensions=list(frame.size),
                        transformations=changes, image_policy=POLICY)
    return dict(type='image_url', image_url=dict(
        url='data:'+mime+';base64,'+base64.b64encode(data).decode(), detail='high')), metadata
