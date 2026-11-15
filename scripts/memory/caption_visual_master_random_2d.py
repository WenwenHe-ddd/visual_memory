"""Random-2D entry for the existing MemEye caption/visual master builder.

Select through visual_memory_pipeline.py offline --memory caption_visual_master
--visual-basis random-2d --basis-seed 0. This module never reads a PCA basis or
calibration images. The shared builder still encodes only the arriving image.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_random_basis(visual_dim: int, seed: int) -> np.ndarray:
    """Draw a fixed two-dimensional subspace with a local PCG64 RNG."""
    if isinstance(visual_dim, bool) or not isinstance(visual_dim, (int, np.integer)) or visual_dim < 2:
        raise ValueError('visual_dim must be an integer >= 2')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    gaussian = rng.standard_normal((int(visual_dim), 2), dtype=np.float64)
    basis, triangular = np.linalg.qr(gaussian, mode='reduced')
    # Fix QR column signs for an explicit, reproducible convention.
    basis *= np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    if not np.allclose(basis.T @ basis, np.eye(2), rtol=0.0, atol=1e-12):
        raise ValueError('Random basis failed its orthogonality check')
    return basis


def offline(args):
    """Build fresh Random-2D state, without changing PCA defaults or artifacts."""
    if getattr(args, 'dataset', 'memeye') != 'memeye':
        raise ValueError('Random-2D currently supports the MemEye adapter only')
    if getattr(args, 'memory_source', None) is not None:
        raise ValueError('Random-2D requires rebuilding arrivals, not memory-source export')
    seed = getattr(args, 'basis_seed', None)
    seed = 0 if seed is None else seed
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Random-2D requires a fresh --output; existing memory will not be overwritten.')
    protocol = json.loads(Path(args.models_protocol).read_text())
    basis = make_random_basis(protocol['image_encoder']['embedding_dim'], seed)
    metadata = {
        'basis_kind': 'random-2d',
        'basis_source': 'generated:gaussian_qr',
        'basis_seed': int(seed),
        'basis_recalibrated': False,
        'external_calibration_images': 0,
        'basis_generation': {
            'generator': 'numpy.random.Generator(PCG64)',
            'distribution': 'independent standard normal',
            'orthogonalization': 'reduced QR; columns signed for nonnegative diagonal of R',
            'shape': list(basis.shape),
            'dtype': str(basis.dtype),
            'numpy_version': np.__version__,
            'fixed_throughout_arrivals': True,
        },
    }
    from scripts.memory import caption_visual_master
    # Reserve the destination before model loading, including when two debug
    # sessions simultaneously choose the same initially empty directory.
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.random_2d.lock'
    try:
        with lock.open('x') as handle:
            handle.write('Random-2D construction is in progress.\n')
    except FileExistsError:
        raise FileExistsError('Random-2D construction already owns this --output.') from None
    try:
        if any(path != lock for path in output.iterdir()):
            raise FileExistsError('Random-2D requires a fresh --output; existing memory will not be overwritten.')
        return caption_visual_master.offline(args, basis_override=basis, basis_metadata=metadata)
    finally:
        lock.unlink()
