/**
 * inr_torch_helper_stub.c — no-op fallback when libtorch is not available.
 *
 * Provides the same C API as inr_torch_helper.cpp so that pmf.so links
 * cleanly on systems without PyTorch.  INRPmfGen.__init__ will raise a
 * RuntimeError at construction time (inr_torch_load returns NULL).
 */
#include "inr_torch_helper.h"

INRTorchModule inr_torch_load(const char* path)
{
    (void)path;
    return NULL;
}

void inr_torch_free(INRTorchModule m)
{
    (void)m;
}

int inr_torch_infer(INRTorchModule m,
                    const float*   coord,
                    double*        coeff_out,
                    int            n_coeffs)
{
    (void)m; (void)coord; (void)coeff_out; (void)n_coeffs;
    return -1;
}
