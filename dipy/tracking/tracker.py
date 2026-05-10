from nibabel.affines import voxel_sizes
import numpy as np

from dipy.data import default_sphere
from dipy.direction import (
    BootDirectionGetter,
    ClosestPeakDirectionGetter,
    ProbabilisticDirectionGetter,
)
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import INRPmfGen, SHCoeffPmfGen, SimplePeakGen, SimplePmfGen
from dipy.tracking.local_tracking import LocalTracking, ParticleFilteringTracking
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.tractogen import generate_tractogram, generate_tractogram_with_dirs
from dipy.tracking.utils import seeds_directions_pairs


def generic_tracking(
    seed_positions,
    seed_directions,
    sc,
    params,
    *,
    affine=None,
    sh=None,
    pam=None,
    sf=None,
    inr_model=None,
    sphere=None,
    basis_type=None,
    legacy=True,
    max_cross=None,
    nbr_threads=0,
    seed_buffer_fraction=1.0,
    save_seeds=False,
):
    affine = affine if affine is not None else np.eye(4)

    pmf_type = [
        {"name": "sh", "value": sh, "cls": SHCoeffPmfGen},
        {"name": "pam", "value": pam, "cls": SimplePeakGen},
        {"name": "sf", "value": sf, "cls": SimplePmfGen},
        {"name": "inr_model", "value": inr_model, "cls": INRPmfGen},
    ]

    initialized_pmf = [
        d_selected for d_selected in pmf_type if d_selected["value"] is not None
    ]
    if len(initialized_pmf) > 1:
        selected_pmf = ", ".join([p["name"] for p in initialized_pmf])
        raise ValueError(
            "Only one pmf type should be initialized. "
            f"Variables initialized: {', '.join(selected_pmf)}"
        )
    if len(initialized_pmf) == 0:
        available_pmf = ", ".join([d["name"] for d in pmf_type])
        raise ValueError(
            f"No PMF found. One of this variable ({available_pmf}) should be"
            " initialized."
        )

    selected_pmf = initialized_pmf[0]

    if selected_pmf["name"] == "sf" and sphere is None:
        raise ValueError("A sphere should be defined when using SF (an ODF).")

    sphere = sphere or default_sphere

    if selected_pmf["name"] == "pam":
        peak_data = selected_pmf["value"]
        if not hasattr(peak_data, "peak_indices") or not hasattr(
            peak_data, "peak_values"
        ):
            raise ValueError(
                "pam must be a PeaksAndMetrics object with "
                "peak_indices and peak_values attributes"
            )

        if hasattr(peak_data, "odf_vertices") and peak_data.odf_vertices is not None:
            odf_vertices = peak_data.odf_vertices
        else:
            odf_vertices = sphere.vertices

        pmf_gen = selected_pmf["cls"](
            # Peak indices are integer sphere vertex ids.
            np.asarray(peak_data.peak_indices, dtype=np.int32, order="C"),
            np.asarray(peak_data.peak_values, dtype=float, order="C"),
            np.asarray(odf_vertices, dtype=float, order="C"),
            sphere,
        )
    elif selected_pmf["name"] == "sh":
        pmf_gen = selected_pmf["cls"](
            np.asarray(selected_pmf["value"], dtype=float),
            sphere,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif selected_pmf["name"] == "inr_model":
        if INRPmfGen is None:
            raise RuntimeError(
                "INRPmfGen is not available. "
                "Rebuild dipy with libtorch support to use inr_model."
            )
        if params.inr is None:
            raise ValueError(
                "inr_model requires INR parameters in the tracker params. "
                "Pass inr_spatial_shape and inr_sh_order to "
                "generate_tracking_parameters."
            )
        pmf_gen = selected_pmf["cls"](
            selected_pmf["value"],
            params.inr.spatial_shape,
            sphere,
            params.inr.sh_order,
            basis_type=basis_type,
            legacy=legacy,
        )
    else:
        pmf_gen = selected_pmf["cls"](
            np.asarray(selected_pmf["value"], dtype=float), sphere
        )

    if seed_directions is not None:
        if not isinstance(seed_directions, (np.ndarray, list)):
            raise ValueError("seed_directions should be a numpy array or a list.")
        elif isinstance(seed_directions, list):
            seed_directions = np.array(seed_directions)

        if not np.array_equal(seed_directions.shape, seed_positions.shape):
            raise ValueError(
                "seed_directions and seed_positions should have the same shape."
            )
    else:
        if selected_pmf["name"] == "pam":
            # Compute seed voxel coordinates
            inv_affine = np.linalg.inv(affine)
            seed_voxels = np.dot(seed_positions, inv_affine[:3, :3].T)
            seed_voxels += inv_affine[:3, 3]
            seed_voxels = np.round(seed_voxels).astype(int)

            seed_voxels[:, 0] = np.clip(
                seed_voxels[:, 0], 0, peak_data.peak_indices.shape[0] - 1
            )
            seed_voxels[:, 1] = np.clip(
                seed_voxels[:, 1], 0, peak_data.peak_indices.shape[1] - 1
            )
            seed_voxels[:, 2] = np.clip(
                seed_voxels[:, 2], 0, peak_data.peak_indices.shape[2] - 1
            )

            seed_peak_indices = peak_data.peak_indices[
                seed_voxels[:, 0], seed_voxels[:, 1], seed_voxels[:, 2]
            ].astype(int)

            seed_peak_values = peak_data.peak_values[
                seed_voxels[:, 0], seed_voxels[:, 1], seed_voxels[:, 2]
            ]

            seed_peak_indices = np.clip(seed_peak_indices, 0, len(odf_vertices) - 1)
            peak_dirs_at_seeds = odf_vertices[seed_peak_indices]

            seed_positions, seed_directions = seeds_directions_pairs(
                seed_positions,
                peak_dirs_at_seeds,
                max_cross=max_cross,
                peak_values=seed_peak_values,
            )
        else:
            peaks_obj = peaks_from_positions(
                seed_positions, None, None, npeaks=1, affine=affine, pmf_gen=pmf_gen
            )
            seed_positions, seed_directions = seeds_directions_pairs(
                seed_positions, peaks_obj, max_cross=max_cross
            )

    return generate_tractogram(
        seed_positions,
        seed_directions,
        sc,
        params,
        pmf_gen,
        affine=affine,
        nbr_threads=nbr_threads,
        buffer_frac=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def probabilistic_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    pam=None,
    sf=None,
    inr_model=None,
    inr_spatial_shape=None,
    inr_sh_order=8,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Probabilistic tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
       Spherical Harmonics (SH).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "prob",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=return_all,
        inr_spatial_shape=inr_spatial_shape,
        inr_sh_order=inr_sh_order,
    )

    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        pam=pam,
        sf=sf,
        inr_model=inr_model,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def deterministic_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    pam=None,
    sf=None,
    inr_model=None,
    inr_spatial_shape=None,
    inr_sh_order=8,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Deterministic tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "det",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=return_all,
        inr_spatial_shape=inr_spatial_shape,
        inr_sh_order=inr_sh_order,
    )
    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        pam=pam,
        sf=sf,
        inr_model=inr_model,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def ptt_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    pam=None,
    sf=None,
    inr_model=None,
    inr_spatial_shape=None,
    inr_sh_order=8,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=10,
    pmf_threshold=0.1,
    probe_length=1.5,
    probe_radius=0,
    probe_quality=7,
    probe_count=1,
    data_support_exponent=1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Parallel Transport Tractography (PTT) tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH) data.
    pam : PeakAndMetrics, optional
        Peaks and Metrics object.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    probe_length : float, optional
        Probe length.
    probe_radius : float, optional
        Probe radius.
    probe_quality : int, optional
        Probe quality.
    probe_count : int, optional
        Probe count.
    data_support_exponent : int, optional
        Data support exponent.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.
    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "ptt",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        probe_length=probe_length,
        probe_radius=probe_radius,
        probe_quality=probe_quality,
        probe_count=probe_count,
        data_support_exponent=data_support_exponent,
        return_all=return_all,
        inr_spatial_shape=inr_spatial_shape,
        inr_sh_order=inr_sh_order,
    )
    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        pam=pam,
        sf=sf,
        inr_model=inr_model,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def closestpeak_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Closest peak tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    dg = None
    sphere = sphere if sphere is not None else default_sphere
    if sh is not None:
        dg = ClosestPeakDirectionGetter.from_shcoeff(
            sh,
            sphere=sphere,
            max_angle=max_angle,
            pmf_threshold=pmf_threshold,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif sf is not None:
        dg = ClosestPeakDirectionGetter.from_pmf(
            sf, sphere=sphere, max_angle=max_angle, pmf_threshold=pmf_threshold
        )
    else:
        raise ValueError("SH or SF should be defined.")

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        dg,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def bootstrap_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    data=None,
    model=None,
    sh=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Bootstrap tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    data : ndarray, optional
        Diffusion data.
    model : Model, optional
        Reconstruction model.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere
    if data is None or model is None:
        raise ValueError("Data and model should be defined.")

    dg = BootDirectionGetter.from_data(
        data,
        model,
        max_angle=max_angle,
    )

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        dg,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def eudx_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    sf=None,
    pam=None,
    max_cross=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.0239,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """EuDX tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    sf : ndarray, optional
        Spherical Function (SF).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object
    max_cross : int or None, optional
        The maximum number of directions to track from each seed in crossing
        voxels. By default (None), all peak directions are tracked.
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        Peak-values threshold used to filter weak peaks.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for parallel processing. Default is 0, which
        uses all available cores.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere
    if pam is None:
        raise ValueError("PAM should be defined.")

    # Get voxel size
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "eudx",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        peak_values_threshold=pmf_threshold,
        angle_threshold=max_angle,
        min_total_weight=0.5,
        random_seed=random_seed,
        return_all=return_all,
    )

    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        pam=pam,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        max_cross=max_cross,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def pft_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    sf=None,
    pam=None,
    max_cross=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    pft_back_tracking_dist=2,
    pft_front_tracking_dist=1,
    pft_max_trial=20,
    particle_count=15,
    save_seeds=False,
    min_wm_pve_before_stopping=0,
    unidirectional=False,
    randomize_forward_direction=False,
):
    """Particle Filtering Tracking (PFT) tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    sf : ndarray, optional
        Spherical Function (SF).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object.
    max_cross : int, optional
        Maximum number of crossing fibers.
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    pft_back_tracking_dist : float, optional
        Back tracking distance.
    pft_front_tracking_dist : float, optional
        Front tracking distance.
    pft_max_trial : int, optional
        Maximum number of trials.
    particle_count : int, optional
        Number of particles.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.
    min_wm_pve_before_stopping : float, optional
        Minimum white matter partial volume estimation before stopping.
    unidirectional : bool, optional
        True to use unidirectional tracking.
    randomize_forward_direction : bool, optional
        True to randomize forward direction

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere

    dg = None
    if sh is not None:
        dg = ProbabilisticDirectionGetter.from_shcoeff(
            sh,
            max_angle=max_angle,
            sphere=sphere,
            sh_to_pmf=True,
            pmf_threshold=pmf_threshold,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif sf is not None:
        dg = ProbabilisticDirectionGetter.from_pmf(
            sf, max_angle=max_angle, sphere=sphere, pmf_threshold=pmf_threshold
        )
    elif pam is not None and sh is None:
        sh = pam.shm_coeff
    else:
        msg = "SH, SF or PAM should be defined."
        raise ValueError(msg)

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return ParticleFilteringTracking(
        dg,
        sc,
        seed_positions,
        affine,
        max_cross=max_cross,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        pft_back_tracking_dist=pft_back_tracking_dist,
        pft_front_tracking_dist=pft_front_tracking_dist,
        particle_count=particle_count,
        pft_max_trial=pft_max_trial,
        return_all=return_all,
        random_seed=random_seed,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
        min_wm_pve_before_stopping=min_wm_pve_before_stopping,
        unidirectional=unidirectional,
        randomize_forward_direction=randomize_forward_direction,
    )


def _build_pmf_gen(sh, pam, sf, inr_model, sphere, basis_type, legacy, params=None):
    """Build a PmfGen from whichever data source is provided."""
    pmf_type = [
        {"name": "sh", "value": sh, "cls": SHCoeffPmfGen},
        {"name": "pam", "value": pam, "cls": SimplePeakGen},
        {"name": "sf", "value": sf, "cls": SimplePmfGen},
        {"name": "inr_model", "value": inr_model, "cls": INRPmfGen},
    ]

    initialized_pmf = [d for d in pmf_type if d["value"] is not None]
    if len(initialized_pmf) != 1:
        names = ", ".join(d["name"] for d in pmf_type)
        if len(initialized_pmf) == 0:
            raise ValueError(f"No PMF found. One of ({names}) should be initialized.")
        raise ValueError(
            "Only one pmf type should be initialized. "
            f"Variables initialized: {', '.join(p['name'] for p in initialized_pmf)}"
        )

    selected_pmf = initialized_pmf[0]

    if selected_pmf["name"] == "sf" and sphere is None:
        raise ValueError("A sphere should be defined when using SF (an ODF).")

    sphere = sphere or default_sphere

    if selected_pmf["name"] == "pam":
        peak_data = selected_pmf["value"]
        if not hasattr(peak_data, "peak_indices") or not hasattr(
            peak_data, "peak_values"
        ):
            raise ValueError(
                "pam must be a PeaksAndMetrics object with "
                "peak_indices and peak_values attributes"
            )
        odf_vertices = (
            peak_data.odf_vertices
            if hasattr(peak_data, "odf_vertices") and peak_data.odf_vertices is not None
            else sphere.vertices
        )
        pmf_gen = selected_pmf["cls"](
            np.asarray(peak_data.peak_indices, dtype=np.int32, order="C"),
            np.asarray(peak_data.peak_values, dtype=float, order="C"),
            np.asarray(odf_vertices, dtype=float, order="C"),
            sphere,
        )
    elif selected_pmf["name"] == "sh":
        pmf_gen = selected_pmf["cls"](
            np.asarray(selected_pmf["value"], dtype=float),
            sphere,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif selected_pmf["name"] == "inr_model":
        if INRPmfGen is None:
            raise RuntimeError(
                "INRPmfGen is not available. "
                "Rebuild dipy with libtorch support to use inr_model."
            )
        if params is None or params.inr is None:
            raise ValueError(
                "inr_model requires INR parameters in the tracker params. "
                "Pass inr_spatial_shape and inr_sh_order to mlft_tracking."
            )
        pmf_gen = selected_pmf["cls"](
            selected_pmf["value"],
            params.inr.spatial_shape,
            sphere,
            params.inr.sh_order,
            basis_type=basis_type,
            legacy=legacy,
        )
    else:
        pmf_gen = selected_pmf["cls"](
            np.asarray(selected_pmf["value"], dtype=float), sphere
        )

    return pmf_gen, sphere


def _resolve_seed_directions(seed_positions, seed_directions, pmf_gen, affine):
    """Resolve seed directions from pmf_gen when not explicitly provided."""
    if seed_directions is not None:
        if isinstance(seed_directions, list):
            seed_directions = np.array(seed_directions)
        if not np.array_equal(seed_directions.shape, seed_positions.shape):
            raise ValueError(
                "seed_directions and seed_positions should have the same shape."
            )
        return seed_positions, seed_directions

    peaks_obj = peaks_from_positions(
        seed_positions, None, None, npeaks=1, affine=affine, pmf_gen=pmf_gen
    )
    return seeds_directions_pairs(seed_positions, peaks_obj, max_cross=None)


def _extract_branch_seeds(
    streamlines,
    vertex_indices_list,
    pmf_gen,
    sphere,
    affine,
    relative_peak_threshold,
    min_separation_angle,
):
    """Extract unused FOD peak directions at streamline points as branch seeds.

    For each point along each streamline, the FOD is evaluated, all peaks
    above threshold are extracted, and any peak that is not the one chosen
    by the tracker (identified via ``vertex_indices_list``) becomes a
    candidate branch seed.

    Parameters
    ----------
    streamlines : list of ndarray
        Streamlines in world space, each shape (N, 3).
    vertex_indices_list : list of ndarray
        Per-streamline sphere vertex indices, each shape (N,) int32.
    pmf_gen : PmfGen
        Probability mass function generator.
    sphere : Sphere
        Sphere used for tracking.
    affine : ndarray
        Voxel-to-world affine.
    relative_peak_threshold : float
        Minimum peak height relative to the largest peak.
    min_separation_angle : float
        Minimum angular separation between distinct peaks (degrees).

    Returns
    -------
    branch_positions : ndarray or None
        World-space positions of branch seeds, shape (M, 3).
    branch_directions : ndarray or None
        Directions for each branch seed, shape (M, 3).

    """
    from dipy.reconst.dirspeed import peak_directions

    inv_affine = np.linalg.inv(affine)
    cos_sep = np.cos(np.deg2rad(min_separation_angle))

    branch_positions = []
    branch_directions = []

    for streamline, vertex_indices in zip(streamlines, vertex_indices_list):
        # Convert to voxel space for pmf_gen
        vox_points = np.dot(streamline, inv_affine[:3, :3].T) + inv_affine[:3, 3]

        for k in range(len(streamline)):
            used_idx = vertex_indices[k]
            if used_idx < 0:
                continue  # seed point, skip

            used_dir = sphere.vertices[used_idx]
            odf = pmf_gen.get_pmf(vox_points[k])
            if odf is None or np.max(odf) <= 0:
                continue

            peaks, values, _ = peak_directions(
                odf,
                sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
            )

            for peak_dir in peaks:
                cos_angle = abs(np.dot(peak_dir, used_dir))
                if cos_angle < cos_sep:
                    branch_positions.append(streamline[k])
                    branch_directions.append(peak_dir)

    if branch_positions:
        return np.array(branch_positions), np.array(branch_directions)
    return None, None


def mlft_tracking(
    seed_positions,
    sc,
    affine,
    target_mask,
    *,
    seed_directions=None,
    sh=None,
    pam=None,
    sf=None,
    inr_model=None,
    inr_spatial_shape=None,
    inr_sh_order=8,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=45,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    max_levels=2,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
):
    """Multi-Level Fiber Tractography (MLFT) tracking algorithm.

    Implements the MLFT method from Hamed et al. (MAGMA 2022). Performs
    deterministic CSD-based tracking and iteratively branches from unused
    FOD peaks at streamline points that did not reach the target region.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space, shape (N, 3).
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Voxel-to-world affine matrix, shape (4, 4).
    target_mask : ndarray
        Binary mask of the target region. Only streamlines passing through
        this region are considered successful.
    seed_directions : ndarray, optional
        Seed directions, shape (N, 3). If None, directions are estimated
        from the FOD data.
    sh : ndarray, optional
        Spherical Harmonics (SH) coefficients.
    pam : PeaksAndMetrics, optional
        Peaks and Metrics object.
    sf : ndarray, optional
        Spherical Function (SF).
    inr_model : object, optional
        Implicit Neural Representation (TorchScript) model for FOD evaluation.
    inr_spatial_shape : tuple, optional
        Spatial shape (X, Y, Z) of the volume the INR was trained on.
        Required when using ``inr_model``.
    inr_sh_order : int, optional
        SH order produced by the INR. Default 8.
    min_len : int, optional
        Minimum streamline length in mm.
    max_len : int, optional
        Maximum streamline length in mm.
    step_size : float, optional
        Step size of the tracking in mm.
    voxel_size : ndarray, optional
        Voxel size. Inferred from affine if not provided.
    max_angle : float, optional
        Maximum angle between successive steps in degrees.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere for SH evaluation.
    basis_type : str, optional
        SH basis type.
    legacy : bool, optional
        Use legacy SH basis definition.
    nbr_threads : int, optional
        Number of threads (0 = all available).
    random_seed : int, optional
        Random seed for reproducibility.
    seed_buffer_fraction : float, optional
        Fraction of seed buffer to process per batch.
    return_all : bool, optional
        If True, return all streamlines (target-reaching and non-target).
        If False, return only target-reaching streamlines.
    max_levels : int, optional
        Maximum number of branching levels. The paper recommends 2.
    relative_peak_threshold : float, optional
        Minimum peak height relative to the largest peak, used during
        branch seed extraction.
    min_separation_angle : float, optional
        Minimum angular separation between peaks in degrees, used during
        branch seed extraction.

    Returns
    -------
    list of ndarray
        Streamlines in world space.

    """
    from dipy.tracking.utils import target

    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "det",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=True,  # always collect all at tracking level; filter later
        inr_spatial_shape=inr_spatial_shape,
        inr_sh_order=inr_sh_order,
    )

    pmf_gen, sphere = _build_pmf_gen(
        sh, pam, sf, inr_model, sphere, basis_type, legacy, params=params
    )
    seed_positions, seed_directions = _resolve_seed_directions(
        seed_positions, seed_directions, pmf_gen, affine
    )

    # --- Level 1 tracking ---
    all_streamlines = []
    all_indices = []
    for track, idx_arr in generate_tractogram_with_dirs(
        seed_positions,
        seed_directions,
        sc,
        params,
        pmf_gen,
        affine=affine,
        nbr_threads=nbr_threads,
        buffer_frac=seed_buffer_fraction,
    ):
        all_streamlines.append(track)
        all_indices.append(idx_arr)

    if not all_streamlines:
        return []

    # --- Split into target-reaching and non-target ---
    target_streamlines = []
    non_target_streamlines = []
    non_target_indices = []

    target_set = set()
    for sl in target(all_streamlines, affine, target_mask, include=True):
        target_set.add(id(sl))

    for sl, idx in zip(all_streamlines, all_indices):
        if id(sl) in target_set:
            target_streamlines.append(sl)
        else:
            non_target_streamlines.append(sl)
            non_target_indices.append(idx)

    # --- Levels 2..max_levels ---
    for _level in range(2, max_levels + 1):
        if not non_target_streamlines:
            break

        branch_pos, branch_dirs = _extract_branch_seeds(
            non_target_streamlines,
            non_target_indices,
            pmf_gen,
            sphere,
            affine,
            relative_peak_threshold,
            min_separation_angle,
        )

        if branch_pos is None or len(branch_pos) == 0:
            break

        # Track from branch seeds
        level_streamlines = []
        level_indices = []
        for track, idx_arr in generate_tractogram_with_dirs(
            branch_pos,
            branch_dirs,
            sc,
            params,
            pmf_gen,
            affine=affine,
            nbr_threads=nbr_threads,
            buffer_frac=seed_buffer_fraction,
        ):
            level_streamlines.append(track)
            level_indices.append(idx_arr)

        if not level_streamlines:
            break

        # Filter new streamlines by target
        new_target_set = set()
        for sl in target(level_streamlines, affine, target_mask, include=True):
            new_target_set.add(id(sl))

        non_target_streamlines = []
        non_target_indices = []
        for sl, idx in zip(level_streamlines, level_indices):
            if id(sl) in new_target_set:
                target_streamlines.append(sl)
            else:
                non_target_streamlines.append(sl)
                non_target_indices.append(idx)

    if return_all:
        return target_streamlines + non_target_streamlines
    return target_streamlines
