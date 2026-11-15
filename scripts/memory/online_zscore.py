"""Transient score correction, independent of memory and query construction."""
from dataclasses import dataclass
import numpy as np
from scripts.memory.ridge_whitening import fit, score, standardize


@dataclass(frozen=True)
class ZScoreWeighting:
    ridge: float = .1
    alpha: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('ridge must be finite and positive')
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')

    def apply(self, query, embeddings):
        # BREAKPOINT: all members of the supplied group, before Top-k selection.
        components, eigenvalues = fit(embeddings)
        raw = embeddings @ query
        white = score(query, embeddings, components, eigenvalues, self.ridge)
        z, z_status = standardize(white)
        _, raw_status = standardize(raw)
        raw_std = float(raw.std())
        correction = np.zeros(len(raw))
        if len(raw) > 1 and raw_status != 'zero_variance' and z_status != 'zero_variance':
            correction = self.alpha * raw_std * np.tanh(z)
        # BREAKPOINT: original cosine plus a bounded correction, never z alone.
        return {'raw': raw, 'white': white, 'z': z, 'correction': correction,
                'weighted': raw + correction, 'raw_std': raw_std,
                'white_std': float(white.std()), 'raw_status': raw_status, 'z_status': z_status}


def score_master(query, embeddings, ridge, alpha):
    """Compatibility entrypoint; also accepts a whole natural memory."""
    return ZScoreWeighting(ridge, alpha).apply(query, embeddings)
