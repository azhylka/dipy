"""Vectorized INR-based tractography.

:func:`vectorized_inr_tracking` is the answer to the question *"how can
INRPmfGen be used with LocalTracking?"*

The problem
-----------
:class:`~dipy.tracking.local_tracking.LocalTracking` calls
``get_pmf_c()`` at the C level with the GIL released (``nogil``).  A
Python method cannot intercept that call, so
:class:`~dipy.direction.inr_direction_getter.INRPmfGen` cannot be used
as a drop-in replacement for :class:`~dipy.direction.pmf.SHCoeffPmfGen`
inside the Cython tracking loop.

The solution: vectorized_inr_tracking
--------------------------------------
Instead of stepping one streamline at a time (the Cython inner loop),
this function steps *all active streamlines simultaneously*:

1. Collect current positions for all active streamlines  → ``(N, 3)``
2. Call ``pmf_gen.get_pmf_batch(positions)``             → ``(N, n_verts)``
3. Apply angle constraint (vectorised ``numpy`` dot product)
4. Sample next direction for every streamline (vectorised cumsum trick)
5. Advance all positions, check stopping criterion
6. Repeat until all streamlines are finished

The neural network forward pass in step 2 runs once per tracking step
for the entire active pool, maximising GPU utilisation.

Coordinate convention
---------------------
Seeds and output streamlines are in **world space (RASMM)**.
PMF queries are performed in **voxel space** (passed to
``pmf_gen.get_pmf_batch`` as voxel coordinates).  The stopping
criterion is a binary mask in voxel space.
"""

import numpy as np

from dipy.tracking.utils import apply_affine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _world_to_vox(pts_world: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    return apply_affine(inv_affine, pts_world)


def _vox_to_world(pts_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return apply_affine(affine, pts_vox)


def _in_mask(pts_vox: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vectorised binary mask lookup. Returns bool array of shape (N,)."""
    ijk = np.round(pts_vox).astype(int)
    inside = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < mask.shape[0])
        & (ijk[:, 1] >= 0) & (ijk[:, 1] < mask.shape[1])
        & (ijk[:, 2] >= 0) & (ijk[:, 2] < mask.shape[2])
    )
    result = np.zeros(len(pts_vox), dtype=bool)
    i = ijk[inside, 0]
    j = ijk[inside, 1]
    k = ijk[inside, 2]
    result[inside] = mask[i, j, k]
    return result


def _sample_directions(pmfs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one direction index per row using the vectorised cumsum trick.

    Parameters
    ----------
    pmfs : (N, n_verts) non-negative, rows that sum to 0 are invalid.

    Returns
    -------
    chosen : (N,) int  — index into sphere.vertices for each streamline.
             -1 for rows with zero sum (stopping streamlines).
    """
    row_sums = pmfs.sum(axis=1)
    valid = row_sums > 0
    chosen = np.full(len(pmfs), -1, dtype=int)
    if valid.any():
        normed = np.zeros_like(pmfs)
        normed[valid] = pmfs[valid] / row_sums[valid, None]
        cumsum = np.cumsum(normed[valid], axis=1)  # (n_valid, n_verts)
        u = rng.random(valid.sum())[:, None]       # (n_valid, 1)
        chosen[valid] = (cumsum < u).sum(axis=1)   # index of first entry ≥ u
        chosen[valid] = np.clip(chosen[valid], 0, pmfs.shape[1] - 1)
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def vectorized_inr_tracking(
    seeds: np.ndarray,
    pmf_gen,
    mask: np.ndarray,
    affine: np.ndarray,
    step_size: float = 0.5,
    max_angle: float = 30.0,
    min_length: float = 10.0,
    max_length: float = 200.0,
    pmf_threshold: float = 0.1,
    random_seed: int = 42,
) -> list:
    """Probabilistic tractography with per-step INR queries.

    All active streamlines are advanced simultaneously each iteration,
    so the neural network is called once per step for the entire pool
    rather than once per streamline per step.

    Parameters
    ----------
    seeds : ndarray, shape (N, 3)
        Seed points in **world space (RASMM)**.
    pmf_gen : INRPmfGen
        PMF generator with a ``get_pmf_batch(points_vox)`` method.
    mask : bool ndarray, shape (X, Y, Z)
        Binary stopping criterion in **voxel space**.  Tracking stops
        when a streamline leaves the mask.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine of the FOD volume.
    step_size : float
        Step size in mm.  Default 0.5.
    max_angle : float
        Maximum turning angle in degrees.  Default 30.
    min_length : float
        Minimum streamline length in mm.  Default 10.
    max_length : float
        Maximum streamline length in mm.  Default 200.
    pmf_threshold : float
        PMF values below this are treated as zero.  Default 0.1.
    random_seed : int
        Seed for the NumPy RNG.

    Returns
    -------
    streamlines : list of ndarray
        Variable-length streamlines in **world space (RASMM)**.
        Only streamlines with length ≥ *min_length* are returned.

    Notes
    -----
    Tracking is **bidirectional**: for each seed the function tracks
    forward (initial direction from PMF peak) and backward (antipodal
    initial direction), then concatenates the two half-streamlines.
    """
    rng = np.random.default_rng(random_seed)
    inv_affine = np.linalg.inv(affine)
    cos_max = np.cos(np.deg2rad(max_angle))
    max_steps = int(max_length / step_size)
    min_steps = int(min_length / step_size)
    vertices = pmf_gen.sphere.vertices  # (n_verts, 3)

    seeds = np.asarray(seeds, dtype=np.float64)
    seeds_vox = _world_to_vox(seeds, inv_affine)  # (N, 3)
    n = len(seeds)

    # ── Initialise directions from PMF at seeds ───────────────────────────────
    init_pmfs = pmf_gen.get_pmf_batch(seeds_vox)  # (N, n_verts)
    init_pmfs = np.clip(init_pmfs, 0.0, None)
    init_pmfs[init_pmfs < pmf_threshold] = 0.0
    init_dirs_idx = _sample_directions(init_pmfs, rng)  # (N,) — -1 if invalid

    # ── Track in both directions from each seed ───────────────────────────────
    # half_streamlines[i] = list of world-space points for seed i, half s
    forward_halves = [None] * n
    backward_halves = [None] * n

    for sign_idx, (target, sign) in enumerate(
        [(forward_halves, 1), (backward_halves, -1)]
    ):
        # Current positions (vox) and directions (world unit vectors)
        pos_vox = seeds_vox.copy()                                   # (N, 3)
        dirs = np.where(
            (init_dirs_idx >= 0)[:, None],
            sign * vertices[np.maximum(init_dirs_idx, 0)],           # (N, 3)
            0.0,
        )
        active = (init_dirs_idx >= 0) & _in_mask(pos_vox, mask)     # (N,) bool

        # Accumulated points per streamline (world space)
        streams = [
            [_vox_to_world(pos_vox[i:i+1], affine)[0]] if active[i] else []
            for i in range(n)
        ]

        for _step in range(max_steps):
            if not active.any():
                break

            idx = np.where(active)[0]
            cur_pos_vox = pos_vox[idx]       # (n_active, 3)
            cur_dirs = dirs[idx]             # (n_active, 3)

            # ── Batch PMF query ───────────────────────────────────────────
            pmfs = pmf_gen.get_pmf_batch(cur_pos_vox)  # (n_active, n_verts)
            pmfs = np.clip(pmfs, 0.0, None)
            pmfs[pmfs < pmf_threshold] = 0.0

            # ── Angle constraint (vectorised) ─────────────────────────────
            # cos_sim[i, j] = dot(cur_dir[i], vertices[j])
            cos_sim = cur_dirs @ vertices.T   # (n_active, n_verts)
            # Keep only forward hemisphere relative to current direction
            pmfs[cos_sim < cos_max] = 0.0

            # ── Sample next direction ─────────────────────────────────────
            chosen = _sample_directions(pmfs, rng)  # (n_active,) — -1 if stuck
            stuck = chosen < 0
            active[idx[stuck]] = False

            still = ~stuck
            if not still.any():
                continue

            still_idx = idx[still]
            new_dirs = vertices[chosen[still]]   # (n_still, 3)

            # ── Step forward ──────────────────────────────────────────────
            # Step in world space (mm), then convert to voxel for mask check
            old_world = _vox_to_world(pos_vox[still_idx], affine)
            new_world = old_world + step_size * new_dirs
            new_vox = _world_to_vox(new_world, inv_affine)

            # ── Stopping criterion ────────────────────────────────────────
            in_m = _in_mask(new_vox, mask)
            active[still_idx[~in_m]] = False

            keep = still_idx[in_m]
            pos_vox[keep] = new_vox[in_m]
            dirs[keep] = new_dirs[in_m]
            for gi, world_pt in zip(keep, new_world[in_m]):
                streams[gi].append(world_pt)

        for i in range(n):
            target[i] = np.array(streams[i]) if streams[i] else None

    # ── Merge forward + backward half-streamlines per seed ───────────────────
    streamlines = []
    for i in range(n):
        fwd = forward_halves[i]
        bwd = backward_halves[i]

        if fwd is not None and bwd is not None:
            # Reverse backward half (it grew away from seed) and prepend
            full = np.concatenate([bwd[::-1], fwd[1:]], axis=0)
        elif fwd is not None:
            full = fwd
        elif bwd is not None:
            full = bwd
        else:
            continue

        if len(full) >= min_steps:
            streamlines.append(full)

    return streamlines
