"""Independent positive-variance weighting: add the SVD term, then +tanh.

This is not whitening and does not negate the existing z-score correction.
The fitted directions come from centered, L2-normalized leaf embeddings; the
matching scores use the original uncentered query and leaves, as in z_score.
"""
from dataclasses import dataclass
import numpy as np
from scripts.memory.ridge_whitening import fit, standardize


@dataclass(frozen=True)
class PositiveVarianceWeighting:
    ridge: float = .1
    alpha: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('ridge must be finite and positive')
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')

    def apply(self, query, embeddings):
        components, eigenvalues = fit(embeddings)
        raw = embeddings @ query
        direction_weights = eigenvalues / (eigenvalues + self.ridge)
        enhanced = raw + (embeddings @ components.T) @ (direction_weights * (components @ query))
        z, z_status = standardize(enhanced)
        _, raw_status = standardize(raw)
        raw_std = float(raw.std())
        correction = np.zeros(len(raw))
        if len(raw) > 1 and raw_status != 'zero_variance' and z_status != 'zero_variance':
            correction = self.alpha * raw_std * np.tanh(z)
        return {'raw': raw, 'positive_variance': enhanced, 'z': z,
                'correction': correction, 'weighted': raw + correction,
                'raw_std': raw_std, 'positive_variance_std': float(enhanced.std()),
                'raw_status': raw_status, 'z_status': z_status}
