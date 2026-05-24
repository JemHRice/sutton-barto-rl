"""
Tile coding for 2-D continuous state spaces (Chapter 10, Appendix B).

Implements the asymmetric offset scheme from Sutton & Barto so that overlapping
tilings do not all share the same tile boundaries.  Uses direct indexing
(no hash collisions) for the Mountain Car state space.

Mountain Car bounds:
  position ∈ [–1.2,  0.6]
  velocity ∈ [–0.07, 0.07]
"""

import numpy as np

POS_MIN, POS_MAX =  -1.2,  0.6
VEL_MIN, VEL_MAX = -0.07, 0.07


class TileCoder:
    """
    Tile coding for a 2-D state (position, velocity).

    Parameters
    ----------
    n_tilings   : number of overlapping tilings (default 8)
    n_tiles     : tiles per dimension per tiling (default 8)
    state_bounds: [(min, max), (min, max)] for each dimension
    """

    def __init__(
        self,
        n_tilings: int = 8,
        n_tiles:   int = 8,
        state_bounds: list[tuple[float, float]] | None = None,
    ):
        if state_bounds is None:
            state_bounds = [(POS_MIN, POS_MAX), (VEL_MIN, VEL_MAX)]
        self.n_tilings    = n_tilings
        self.n_tiles      = n_tiles
        self.state_bounds = state_bounds
        # Asymmetric offsets: coprime strides (1, 3) so tilings spread well in 2-D.
        self._offsets = np.array([[i * 1, i * 3], [0, 0]], dtype=float)
        # Each tiling: (n_tiles+1)^2 cells (the +1 covers boundary overshoot after offset).
        self.n_cells_per_tiling = (n_tiles + 1) ** 2
        self.n_features         = n_tilings * self.n_cells_per_tiling

    def _scale(self, state: np.ndarray) -> np.ndarray:
        """Map state to [0, n_tiles] range."""
        lo = np.array([b[0] for b in self.state_bounds])
        hi = np.array([b[1] for b in self.state_bounds])
        return (state - lo) / (hi - lo) * self.n_tiles

    def active_tiles(self, state: np.ndarray) -> np.ndarray:
        """
        Return array of n_tilings active feature indices for the given state.
        Each index is in [0, n_features).
        """
        scaled  = self._scale(np.asarray(state, dtype=float))
        indices = np.empty(self.n_tilings, dtype=int)
        for i in range(self.n_tilings):
            offset  = np.array([i * 1, i * 3]) / self.n_tilings
            coords  = np.clip(
                (scaled + offset).astype(int),
                0, self.n_tiles,
            )
            cell = coords[0] * (self.n_tiles + 1) + coords[1]
            indices[i] = i * self.n_cells_per_tiling + cell
        return indices

    def value(self, weights: np.ndarray, state: np.ndarray) -> float:
        """Q(s) ≈ sum of weights at active tiles."""
        return float(weights[self.active_tiles(state)].sum())

    def update(
        self,
        weights: np.ndarray,
        state:   np.ndarray,
        target:  float,
        alpha:   float,
    ) -> None:
        """
        Semi-gradient update: w[active] += α * (target – Q(s)) / n_tilings.
        Dividing by n_tilings keeps the effective step size equal to α
        regardless of how many tiles fire.
        """
        tiles     = self.active_tiles(state)
        error     = target - weights[tiles].sum()
        weights[tiles] += (alpha / self.n_tilings) * error
