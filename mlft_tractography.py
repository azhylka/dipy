#!/usr/bin/env python3
"""MLFT whole-brain tractography on BIDS wmfod files.

Pipeline per subject / session / FOD variant
--------------------------------------------
1. Load the wmfod NIfTI volume.
2. Run :func:`~dipy.tracking.tracker.mlft_tracking` with ``sh=fod_data``,
   ``seed_mask`` (where to start) and ``target_mask`` (where streamlines
   must reach for branching to terminate).
3. Save tractogram as ``.tck`` (RASMM space).

Seed and target masks (required) and exclusion mask (optional) may be:
- A single NIfTI file used for every subject (``--seed-mask``, ``--target-mask``,
  ``--exclude-mask``)
- Per-subject files discovered automatically using a filename template
  (``--seed-template`` / ``--target-template`` / ``--exclude-template``) where
  ``{sub}`` and ``{ses}`` are substituted (e.g. ``{sub}_internal_capsule.nii.gz``).

Streamlines passing through the exclusion mask are removed as a post-tracking
filter (matching the paper's approach of applying identical exclusion regions
across methods).

Usage
-----
python mlft_tractography.py \\
    --bids /media/andrey/data/FaceTract/data/bids \\
    --target-mask /path/to/cortex.nii.gz                         \\
    [--seed-mask /path/to/internal_capsule.nii.gz]               \\
    [--seed-template "{sub}_seed_region.nii.gz"]                 \\
    [--target-template "{sub}_target_region.nii.gz"]             \\
    [--exclude-mask /path/to/exclude.nii.gz]                     \\
    [--exclude-template "{sub}_csf.nii.gz"]                      \\
    [--output-dir derivatives/mlft_tractography]                 \\
    [--subjects sub-FacTrac001]                                  \\
    [--sessions EGON Holland Klinische]                          \\
    [--fod-variants tournier]                                    \\
    [--seed-density 2]                                           \\
    [--max-angle 45] [--step-size 0.5]                           \\
    [--min-length 20] [--max-length 200]                         \\
    [--pmf-threshold 0.1] [--max-levels 2]                       \\
    [--nbr-threads 0]                                            \\
    [--dry-run]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

SESSIONS = ["EGON", "Holland", "Klinische"]


# ─────────────────────────────────────────────────────────────────────────────
# Mask resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_mask(
    sub: str,
    ses: str,
    dwi_dir: Path,
    explicit: Optional[Path],
    template: Optional[str],
    label: str,
    required: bool = True,
) -> Optional[Path]:
    """Resolve a per-job mask path from explicit file or template.

    If ``required`` is False, returns None when neither is provided.
    """
    if explicit is not None:
        return explicit
    if template is not None:
        path = dwi_dir / template.format(sub=sub, ses=ses)
        if not path.exists():
            raise FileNotFoundError(
                f"{label} mask from template not found: {path}"
            )
        return path
    if required:
        raise ValueError(
            f"No {label} mask provided. Pass --{label}-mask <path> or "
            f"--{label}-template <pattern>."
        )
    return None


def resolve_inr(
    sub: str,
    ses: str,
    output_dir: Path,
    inr_ts: Optional[Path],
    inr_ckpt: Optional[Path],
    inr_template: Optional[str],
) -> tuple:
    """Resolve INR (.ts, .pt) paths from explicit args or per-subject template.

    Returns (None, None) when INR is not requested.
    """
    if inr_ts is not None:
        if inr_ckpt is None:
            raise ValueError("--inr-ts requires --inr-checkpoint")
        if not inr_ts.exists():
            raise FileNotFoundError(f"INR TorchScript not found: {inr_ts}")
        if not inr_ckpt.exists():
            raise FileNotFoundError(f"INR checkpoint not found: {inr_ckpt}")
        return inr_ts, inr_ckpt

    if inr_template is not None:
        stem = inr_template.format(sub=sub, ses=ses)
        job_dir = output_dir / sub / f"ses-{ses}"
        ts_path = job_dir / f"{stem}.ts"
        ckpt_path = job_dir / f"{stem}.pt"
        if not ts_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"INR files from template not found: {ts_path} / {ckpt_path}"
            )
        return ts_path, ckpt_path

    return None, None


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
    seed_mask_path, target_mask_path, exclude_mask_path=None,
    seed_density, max_angle, step_size,
    min_length, max_length, pmf_threshold,
    max_levels, relative_peak_threshold, min_separation_angle,
    nbr_threads, force_retract=False,
    inr_ts_path=None, inr_ckpt_path=None,
):
    from dipy.core.sphere import Sphere
    from dipy.data import default_sphere
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.tracker import mlft_tracking
    from dipy.tracking.utils import seeds_from_mask, target as target_filter

    stem = fod_path.name.replace(".nii.gz", "")
    job_dir = output_dir / sub / f"ses-{ses}"
    job_dir.mkdir(parents=True, exist_ok=True)

    tract_path = job_dir / f"{stem}_mlft_tractogram.tck"
    if tract_path.exists() and not force_retract:
        print(f"  [skip] {tract_path.name}")
        return tract_path

    # ── 1. Load FOD (or INR), brain mask, seed mask, target mask ──────────────
    use_inr = inr_ts_path is not None

    fod_img = nib.load(fod_path)
    affine = fod_img.affine
    brain_mask = np.asarray(nib.load(mask_path).dataobj, dtype=bool)

    fod_data = None
    inr_spatial_shape = None
    inr_sh_order = None
    if use_inr:
        import torch
        from dipy.reconst import shm

        ckpt_meta = torch.load(str(inr_ckpt_path), map_location="cpu")
        inr_spatial_shape = tuple(ckpt_meta["image_shape"])
        inr_sh_order = shm.order_from_ncoef(ckpt_meta["n_channels"])
        print(f"  [inr]     ts={inr_ts_path.name} shape={inr_spatial_shape} "
              f"sh_order={inr_sh_order}")
    else:
        fod_data = np.asarray(fod_img.dataobj, dtype=np.float64)
        fod_data = np.flip(fod_data, axis=0)  # match PTT script convention

    seed_mask = np.asarray(nib.load(seed_mask_path).dataobj, dtype=bool) & brain_mask
    target_mask = np.asarray(nib.load(target_mask_path).dataobj, dtype=bool)
    print(f"  [seed]    {seed_mask_path.name}: {seed_mask.sum():,} voxels")
    print(f"  [target]  {target_mask_path.name}: {target_mask.sum():,} voxels")

    exclude_mask = None
    if exclude_mask_path is not None:
        exclude_mask = np.asarray(nib.load(exclude_mask_path).dataobj, dtype=bool)
        print(f"  [exclude] {exclude_mask_path.name}: {exclude_mask.sum():,} voxels")

    # ── 2. Build seeds ────────────────────────────────────────────────────────
    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)
    print(f"  [track]   {len(seeds):,} seeds  density={seed_density}")

    # ── 3. Track with MLFT ────────────────────────────────────────────────────
    sc = BinaryStoppingCriterion(brain_mask)
    sphere = default_sphere
    sphere = Sphere(x=sphere.x, y=sphere.y, z=sphere.z)

    inr_kwargs = {}
    if use_inr:
        inr_kwargs = dict(
            inr_model=str(inr_ts_path),
            inr_spatial_shape=inr_spatial_shape,
            inr_sh_order=inr_sh_order,
        )

    t0 = time.time()
    streamlines = mlft_tracking(
        seeds, sc, affine, target_mask,
        sh=None if use_inr else fod_data,
        sphere=sphere,
        basis_type="tournier07",
        legacy=False if not use_inr else True,
        max_angle=max_angle,
        step_size=step_size,
        min_len=min_length,
        max_len=max_length,
        pmf_threshold=pmf_threshold,
        max_levels=max_levels,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        nbr_threads=nbr_threads,
        return_all=False,  # only target-reaching streamlines
        **inr_kwargs,
    )
    streamlines = Streamlines(streamlines)
    print(f"  [track]   {len(streamlines):,} streamlines  {time.time()-t0:.1f}s")

    # ── 3b. Exclusion mask filtering ──────────────────────────────────────────
    if exclude_mask is not None:
        streamlines = Streamlines(
            target_filter(streamlines, affine, exclude_mask, include=False)
        )
        print(f"  [exclude] {len(streamlines):,} streamlines after exclusion")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    sft = StatefulTractogram(streamlines, fod_img, Space.RASMM)
    save_tractogram(sft, str(tract_path), bbox_valid_check=False)
    print(f"  [save]    {tract_path}")
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
                   help="Default: {bids}/derivatives/mlft_tractography")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sessions", nargs="+", default=None, choices=SESSIONS)
    p.add_argument("--fod-variants", nargs="+", default=["tournier"])

    g_masks = p.add_argument_group("seed/target masks (required for MLFT)")
    g_masks.add_argument("--seed-mask", type=Path, default=None,
                         help="NIfTI seed region (single file used for all subjects). "
                              "Mutually exclusive with --seed-template.")
    g_masks.add_argument("--seed-template", type=str, default=None,
                         help="Filename pattern relative to subject's dwi/ dir, "
                              "e.g. '{sub}_internal_capsule.nii.gz'. "
                              "{sub} and {ses} are substituted.")
    g_masks.add_argument("--target-mask", type=Path, default=None,
                         help="NIfTI target region (single file for all subjects).")
    g_masks.add_argument("--target-template", type=str, default=None,
                         help="Filename pattern relative to subject's dwi/ dir, "
                              "e.g. '{sub}_motor_cortex.nii.gz'.")
    g_masks.add_argument("--exclude-mask", type=Path, default=None,
                         help="NIfTI exclusion region (single file for all subjects). "
                              "Streamlines passing through this mask are discarded "
                              "as a post-tracking filter. Optional.")
    g_masks.add_argument("--exclude-template", type=str, default=None,
                         help="Filename pattern for per-subject exclusion mask, "
                              "e.g. '{sub}_csf.nii.gz'. Optional.")

    g = p.add_argument_group("MLFT tractography parameters")
    g.add_argument("--seed-density", type=int, default=2)
    g.add_argument("--max-angle", type=float, default=45.0,
                   help="Paper default: 45°")
    g.add_argument("--step-size", type=float, default=0.5)
    g.add_argument("--min-length", type=float, default=20.0)
    g.add_argument("--max-length", type=float, default=200.0)
    g.add_argument("--pmf-threshold", type=float, default=0.1,
                   help="Paper default: 0.1")
    g.add_argument("--max-levels", type=int, default=2,
                   help="Paper recommends 2; 3 adds no improvement.")
    g.add_argument("--relative-peak-threshold", type=float, default=0.5,
                   help="For unused-peak detection during branching.")
    g.add_argument("--min-separation-angle", type=float, default=25.0,
                   help="Min angular separation between distinct peaks (degrees).")
    g.add_argument("--nbr-threads", type=int, default=0,
                   help="Number of threads (0 = all available).")

    g_inr = p.add_argument_group(
        "INR model (optional; replaces SH-based FOD evaluation)",
    )
    g_inr.add_argument("--inr-ts", type=Path, default=None,
                       help="TorchScript (.ts) file produced by inr_tractography.py. "
                            "When set, MLFT evaluates the FOD via the INR.")
    g_inr.add_argument("--inr-checkpoint", type=Path, default=None,
                       help="Companion .pt checkpoint with image_shape + n_channels "
                            "metadata. Required with --inr-ts.")
    g_inr.add_argument("--inr-template", type=str, default=None,
                       help="Filename pattern for per-subject INR files (looked up "
                            "under <output-dir>/{sub}/ses-{ses}/), with {sub} and "
                            "{ses} substituted. Both .ts and .pt files are expected "
                            "to share this stem.")

    p.add_argument("--force-retract", action="store_true",
                   help="Rerun tractography even if a tractogram already exists.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    bids = args.bids.resolve()
    if not bids.is_dir():
        sys.exit(f"BIDS directory not found: {bids}")

    if args.seed_mask is None and args.seed_template is None:
        sys.exit("Provide --seed-mask or --seed-template")
    if args.target_mask is None and args.target_template is None:
        sys.exit("Provide --target-mask or --target-template")

    out_dir = (args.output_dir or bids / "derivatives" / "mlft_tractography").resolve()
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
            seed_mask_path = resolve_mask(
                sub, ses, fod.parent, args.seed_mask, args.seed_template, "seed",
            )
            target_mask_path = resolve_mask(
                sub, ses, fod.parent, args.target_mask, args.target_template, "target",
            )
            exclude_mask_path = resolve_mask(
                sub, ses, fod.parent, args.exclude_mask, args.exclude_template,
                "exclude", required=False,
            )
            inr_ts_path, inr_ckpt_path = resolve_inr(
                sub, ses, out_dir, args.inr_ts, args.inr_checkpoint,
                args.inr_template,
            )
            run_job(
                sub, ses, fod, mask, out_dir,
                seed_mask_path=seed_mask_path,
                target_mask_path=target_mask_path,
                exclude_mask_path=exclude_mask_path,
                inr_ts_path=inr_ts_path,
                inr_ckpt_path=inr_ckpt_path,
                seed_density=args.seed_density,
                max_angle=args.max_angle,
                step_size=args.step_size,
                min_length=args.min_length,
                max_length=args.max_length,
                pmf_threshold=args.pmf_threshold,
                max_levels=args.max_levels,
                relative_peak_threshold=args.relative_peak_threshold,
                min_separation_angle=args.min_separation_angle,
                nbr_threads=args.nbr_threads,
                force_retract=args.force_retract,
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
