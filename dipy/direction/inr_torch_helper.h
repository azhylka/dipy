/**
 * inr_torch_helper.h — thin C API wrapping libtorch TorchScript inference.
 *
 * All functions are safe to call from a Cython ``noexcept nogil`` context:
 * they never touch the Python C API.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to a loaded TorchScript module. */
typedef struct INRTorchModule_t* INRTorchModule;

/**
 * Load a TorchScript module from ``path`` (as saved by ``torch.jit.save``).
 * Returns NULL on failure (file not found, wrong format, …).
 * Thread-safe; each call returns an independent module handle.
 */
INRTorchModule inr_torch_load(const char* path);

/** Release a module handle created by inr_torch_load. */
void inr_torch_free(INRTorchModule m);

/**
 * Run one forward pass of the TorchScript module.
 *
 * Parameters
 * ----------
 * m          : module handle from inr_torch_load
 * coord      : (3,) float32 array, values in [-1, 1]^3
 * coeff_out  : (n_coeffs,) double array — output SH coefficients
 * n_coeffs   : expected number of output coefficients
 *
 * Returns 0 on success, -1 on any error (wrong output shape, CUDA OOM, …).
 * The function is nogil-safe: no Python objects are touched.
 */
int inr_torch_infer(INRTorchModule  m,
                    const float*    coord,
                    double*         coeff_out,
                    int             n_coeffs);

#ifdef __cplusplus
}
#endif
