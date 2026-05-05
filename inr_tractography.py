#!/usr/bin/env python3
"""INR-based whole-brain tractography on BIDS wmfod files.

Pipeline per subject / session / FOD variant
--------------------------------------------
1. Train an ImageINR on the wmfod volume  (skipped when checkpoint exists).
2. Export the checkpoint as a TorchScript ``.ts`` file (skipped when it
   already exists alongside the checkpoint).
3. Run :func:`~dipy.tracking.tracker.deterministic_tracking` with
   ``inr_model`` pointing at the ``.ts`` file.
   :class:`~dipy.direction.pmf.INRPmfGen` is selected automatically; it
   runs the INR forward pass via libtorch with no GIL held, so
   multi-threaded tracking works without subprocess overhead.
4. Save tractogram as ``.trk`` (RASMM space).

Parallelism (--nbr-threads)
----------------------------
Pass ``--nbr-threads N`` to let the tracker use N threads.  Because
``INRPmfGen`` never holds the GIL, all threads run concurrently.  The
default (0) uses all available cores.

Usage
-----
python inr_tractography.py \\
    --bids /media/andrey/data/FaceTract/data/bids \\
    [--output-dir derivatives/inr_tractography]   \\
    [--subjects sub-FacTrac001]                   \\
    [--sessions EGON Holland Klinische]            \\
    [--fod-variants tournier dhollander]           \\
    [--epochs 200]                                 \\
    [--seed-density 2]                             \\
    [--max-angle 30] [--step-size 0.5]             \\
    [--min-length 20] [--max-length 200]           \\
    [--nbr-threads 4]                              \\
    [--dry-run]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from msmt_csd_inr import ImageINR, TrainConfig, train as _train_fn

SESSIONS = ["EGON", "Holland", "Klinische"]


# ─────────────────────────────────────────────────────────────────────────────
# INR model helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_canonical(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI volume and reorient it to RAS+ canonical orientation.

    ``nib.as_closest_canonical`` flips/transposes axes so the data array is in
    RAS+ order *and* updates the affine consistently. This is required because
    the INR is trained on coordinates in [-1, 1]^3 mapped from data-array voxel
    indices — the tracker must therefore query voxel coordinates from an affine
    matching the same voxel layout the INR saw at training time.
    """
    return nib.as_closest_canonical(nib.load(str(path)))


def _check_pmf(fod_path: Path) -> None:
    """Run one PMF query at the volume centre to verify the FOD data."""
    from dipy.data import get_sphere
    from dipy.direction.pmf import SHCoeffPmfGen

    fod_data = np.asarray(_load_canonical(fod_path).dataobj, dtype=np.float64)
    sphere = get_sphere(name="repulsion724")
    pmf_gen = SHCoeffPmfGen(fod_data, sphere, "tournier07", legacy=True)
    spatial_shape = fod_data.shape[:3]
    centre = np.array([s / 2.0 for s in spatial_shape], dtype=np.float64)
    pmf = np.asarray(pmf_gen.get_pmf(centre))
    pmf_max = float(np.max(pmf))
    pmf_pos = int(np.sum(pmf > 0))
    print(f"  [check] PMF at centre — max={pmf_max:.4f}  positive={pmf_pos}/{len(pmf)}")
    if pmf_max == 0.0:
        raise RuntimeError(
            "PMF is all-zero at the volume centre — check FOD data and basis type."
        )


def _train_inr(fod_path: Path, job_dir: Path, ckpt_name: str,
               epochs: int, device: Optional[str], num_workers: int = 0) -> Path:
    cfg = TrainConfig(
        image=str(fod_path),
        output=str(job_dir),
        epochs=epochs,
        batch_size=100000,
        checkpoint_name=ckpt_name,
        device=device,
        num_workers=num_workers,
        amp=False,
    )
    return _train_fn(cfg)


def _export_torchscript(ckpt_path: Path, ts_path: Path) -> Path:
    """Load a ``.pt`` checkpoint and save it as a TorchScript ``.ts`` file.

    The exported model accepts ``(1, 3)`` float32 normalised coordinates
    in ``[-1, 1]^3`` and returns ``(1, n_coeffs)`` float32 SH coefficients,
    which is the exact interface expected by ``INRPmfGen`` / ``inr_torch_infer``.

    Always traced on CPU because ``inr_torch_helper.cpp`` loads the module on
    CPU and feeds it CPU tensors.  Tracing on CUDA would bake ``device=cuda:0``
    into the serialised graph, causing a device mismatch at inference time.

    Parameters
    ----------
    ckpt_path : Path
        Path to the ``.pt`` checkpoint written by ``_train_inr``.
    ts_path : Path
        Destination for the ``.ts`` TorchScript file.
    device : str or None
        Unused — kept for API compatibility.  Tracing always runs on CPU.

    Returns
    -------
    ts_path : Path
    """
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = ImageINR(**ckpt["model_config"]).to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))
    print(f"  [export] TorchScript saved → {ts_path.name}")
    return ts_path


# ─────────────────────────────────────────────────────────────────────────────
# Job discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover_jobs(
    bids_dir: Path,
    subjects: Optional[list] = None,
    sessions: Optional[list] = None,
    fod_variants: Optional[list] = None,
) -> list:
    if sessions is None:
        sessions = SESSIONS
    if fod_variants is None:
        fod_variants = ["tournier"]

    jobs = []
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name
        if subjects and sub not in subjects:
            continue
        for ses in sessions:
            ses_dir = sub_dir / f"ses-{ses}"
            dwi_dir = ses_dir / "dwi"
            if not dwi_dir.is_dir():
                continue
            mask = dwi_dir / f"{sub}_mask.nii.gz"
            if not mask.exists():
                print(f"  [warn] mask not found: {mask}")
                continue
            for variant in fod_variants:
                fod = dwi_dir / f"{sub}_wmfod_{variant}.nii.gz"
                if fod.exists():
                    jobs.append((sub, ses, fod, mask))
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Single-job pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_job(
    sub, ses, fod_path, mask_path, output_dir, *,
    epochs, seed_density, max_angle, step_size,
    min_length, max_length, pmf_threshold,
    device, skip_training, nbr_threads, num_workers=0, seed_mask_path=None,
    force_retrain=False, force_retract=False, target_mask_path=None,
    exclude_mask_path=None,
):
    import torch
    from dipy.data import get_sphere
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    from dipy.reconst import shm
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.tracker import deterministic_tracking
    from dipy.tracking.utils import seeds_from_mask

    stem = fod_path.name.replace(".nii.gz", "")
    job_dir = output_dir / sub / f"ses-{ses}"
    job_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = job_dir / f"{stem}_inr.pt"
    ts_path = job_dir / f"{stem}_inr.ts"
    tract_path = job_dir / f"{stem}_inr_tractogram.tck"

    if tract_path.exists() and not force_retract:
        print(f"  [skip] {tract_path.name}")
        return tract_path

    # ── 1. Train INR ──────────────────────────────────────────────────────────
    if force_retrain and ckpt_path.exists():
        ckpt_path.unlink()
        ts_path.unlink(missing_ok=True)
    if ckpt_path.exists():
        print(f"  [load] {ckpt_path.name}")
    else:
        if skip_training:
            raise FileNotFoundError(f"No checkpoint: {ckpt_path}")
        print(f"  [train] {stem}  ({epochs} epochs) …")
        t0 = time.time()
        _train_inr(fod_path, job_dir, ckpt_path.name, epochs, device, num_workers)
        print(f"  [train] {(time.time()-t0)/60:.1f} min")

    # ── 2. Export to TorchScript (INRPmfGen requires a .ts file) ─────────────
    if not ts_path.exists():
        _export_torchscript(ckpt_path, ts_path)

    # ── 3. Read volume metadata from the checkpoint (avoids loading the FOD) ─
    ckpt_meta = torch.load(str(ckpt_path), map_location="cpu")
    spatial_shape = tuple(ckpt_meta["image_shape"])
    sh_order = shm.order_from_ncoef(ckpt_meta["n_channels"])

    # ── sanity-check: one PMF query at the volume centre ─────────────────────
    _check_pmf(fod_path)

    # ── 4. Load affine and masks (canonical RAS+ to match training) ──────────
    fod_img = _load_canonical(fod_path)
    affine = fod_img.affine
    mask_data = np.asarray(_load_canonical(mask_path).dataobj, dtype=bool)

    # ── 5. Build seed mask ────────────────────────────────────────────────────
    if seed_mask_path is not None:
        seed_mask = (
            np.asarray(_load_canonical(seed_mask_path).dataobj, dtype=bool) & mask_data
        )
        print(f"  [seed]  custom mask {seed_mask_path.name}: {seed_mask.sum():,} voxels")
    else:
        fa_path = mask_path.parent / mask_path.name.replace("_mask", "_FA")
        if fa_path.exists():
            fa = np.asarray(_load_canonical(fa_path).dataobj)
            seed_mask = (fa > 0.2) & mask_data
            print(f"  [seed]  FA>0.2 WM mask: {seed_mask.sum():,} voxels")
        else:
            seed_mask = mask_data
            print(f"  [seed]  using full brain mask ({seed_mask.sum():,} voxels)")

    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)
    print(f"  [track] {len(seeds):,} seeds  density={seed_density}")

    # ── 6. Track ──────────────────────────────────────────────────────────────
    sphere = get_sphere(name="repulsion724")
    sc = BinaryStoppingCriterion(mask_data)

    t0 = time.time()
    tractogram = deterministic_tracking(
        seeds,
        sc,
        affine,
        inr_model=str(ts_path),
        inr_spatial_shape=spatial_shape,
        inr_sh_order=sh_order,
        sphere=sphere,
        basis_type="tournier07",
        legacy=True,
        min_len=min_length,
        max_len=max_length,
        step_size=step_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        nbr_threads=nbr_threads,
        return_all=False,
    )
    streamlines = list(tractogram)
    print(f"  [track] {len(streamlines):,} streamlines  {time.time()-t0:.1f}s")

    # ── 6b. Target / exclusion mask filtering ─────────────────────────────────
    if target_mask_path is not None:
        from dipy.tracking.utils import target
        target_mask = np.asarray(nib.as_closest_canonical(nib.load(target_mask_path)).dataobj, dtype=bool)
        streamlines = list(target(streamlines, affine, target_mask, include=True))
        print(f"  [target]  {len(streamlines):,} streamlines after target mask")

    if exclude_mask_path is not None:
        from dipy.tracking.utils import target
        exclude_mask = np.asarray(nib.as_closest_canonical(nib.load(exclude_mask_path)).dataobj, dtype=bool)
        streamlines = list(target(streamlines, affine, exclude_mask, include=False))
        print(f"  [exclude] {len(streamlines):,} streamlines after exclusion mask")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    from dipy.tracking.streamline import Streamlines
    sft = StatefulTractogram(Streamlines(streamlines), fod_img, Space.RASMM)
    save_tractogram(sft, str(tract_path), bbox_valid_check=False)
    print(f"  [save]  {tract_path}")
    return tract_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bids", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: {bids}/derivatives/inr_tractography")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sessions", nargs="+", default=None, choices=SESSIONS)
    p.add_argument("--fod-variants", nargs="+", default=["tournier"])

    g = p.add_argument_group("INR training")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes for INR training (default: 0).")
    g.add_argument("--skip-training", action="store_true")
    g.add_argument("--device", default=None)

    g = p.add_argument_group("tractography")
    g.add_argument("--seed-density", type=int, default=2)
    g.add_argument("--max-angle", type=float, default=45.0)
    g.add_argument("--step-size", type=float, default=0.5)
    g.add_argument("--min-length", type=float, default=20.0)
    g.add_argument("--max-length", type=float, default=200.0)
    g.add_argument("--pmf-threshold", type=float, default=0.1)
    g.add_argument("--seed-mask", type=Path, default=None,
                   help="NIfTI seed mask (ANDed with brain mask). "
                        "Default: FA>0.2 if available, else brain mask.")
    g.add_argument("--target-mask", type=Path, default=None,
                   help="NIfTI target ROI mask. Only streamlines passing "
                        "through this mask are kept.")
    g.add_argument("--exclude-mask", type=Path, default=None,
                   help="NIfTI exclusion mask. Streamlines passing through "
                        "this mask are discarded.")
    g.add_argument("--nbr-threads", type=int, default=0,
                   help="Tracker threads (0 = all cores). INRPmfGen holds no "
                        "GIL so all threads run concurrently.")

    p.add_argument("--force-retrain", action="store_true",
                   help="Delete existing checkpoint and retrain from scratch.")
    p.add_argument("--force-retract", action="store_true",
                   help="Rerun tractography even if a tractogram already exists.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    bids = args.bids.resolve()
    if not bids.is_dir():
        sys.exit(f"BIDS directory not found: {bids}")

    out_dir = (args.output_dir or bids / "derivatives" / "inr_tractography").resolve()
    jobs = discover_jobs(bids, args.subjects, args.sessions, args.fod_variants)
    print(f"Found {len(jobs)} job(s) → {out_dir}")

    if args.dry_run:
        for sub, ses, fod, _ in jobs:
            print(f"  {sub}  ses-{ses}  {fod.name}")
        return

    errors = []
    for i, (sub, ses, fod, mask) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {sub}  ses-{ses}  {fod.name}")
        t0 = time.time()
        try:
            run_job(
                sub, ses, fod, mask, out_dir,
                epochs=args.epochs,
                seed_density=args.seed_density,
                max_angle=args.max_angle,
                step_size=args.step_size,
                min_length=args.min_length,
                max_length=args.max_length,
                pmf_threshold=args.pmf_threshold,
                device=args.device,
                skip_training=args.skip_training,
                nbr_threads=args.nbr_threads,
                num_workers=args.num_workers,
                seed_mask_path=args.seed_mask,
                force_retrain=args.force_retrain,
                force_retract=args.force_retract,
                target_mask_path=args.target_mask,
                exclude_mask_path=args.exclude_mask,
            )
            print(f"  [done] {(time.time()-t0)/60:.1f} min")
        except Exception as exc:
            msg = f"{sub} ses-{ses} {fod.name}: {exc}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)

    print(f"\n{'─'*60}")
    print(f"Completed {len(jobs)-len(errors)}/{len(jobs)} jobs")
    if errors:
        for e in errors:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
