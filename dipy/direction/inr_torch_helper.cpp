/**
 * inr_torch_helper.cpp — libtorch implementation of the C API.
 *
 * Compiled as C++17 and linked against libtorch.  The Python GIL is never
 * acquired; the code uses only the PyTorch C++ API.
 */
#include "inr_torch_helper.h"

#include <torch/script.h>   // torch::jit::load, IValue, Module
#include <torch/torch.h>    // torch::from_blob, TensorOptions, kCPU, kCUDA
#include <cstdio>
#include <stdexcept>
#include <vector>

/* ── internal struct ────────────────────────────────────────────────────── */

struct INRTorchModule_t {
    torch::jit::Module module;
    torch::Device      device;

    explicit INRTorchModule_t(const char* path)
        : module(torch::jit::load(path, torch::kCPU))
        , device(torch::kCPU)
    {
        module.eval();
    }
};

/* ── C API ──────────────────────────────────────────────────────────────── */

extern "C" {

INRTorchModule inr_torch_load(const char* path)
{
    try {
        return new INRTorchModule_t(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[inr_torch_load] %s\n", e.what());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[inr_torch_load] unknown exception\n");
        return nullptr;
    }
}

void inr_torch_free(INRTorchModule m)
{
    delete m;
}

int inr_torch_infer(INRTorchModule  m,
                    const float*    coord,
                    double*         coeff_out,
                    int             n_coeffs)
{
    if (!m || !coord || !coeff_out || n_coeffs <= 0)
        return -1;

    try {
        /* Build (1, 3) float32 CPU tensor from the raw coord pointer.
         * torch::from_blob does NOT take ownership — the pointer must stay
         * valid for the duration of this call, which it does (caller-owned). */
        auto cpu_opts = torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .requires_grad(false);
        auto input = torch::from_blob(
            const_cast<float*>(coord), {1, 3}, cpu_opts);

        /* Forward pass. */
        std::vector<torch::jit::IValue> inputs = {input};
        auto output = m->module.forward(inputs).toTensor();

        /* Bring result to CPU float64, contiguous. */
        output = output.squeeze(0).contiguous().cpu().to(torch::kDouble);

        const auto actual_n = static_cast<int>(output.numel());
        if (actual_n != n_coeffs) {
            std::fprintf(stderr,
                "[inr_torch_infer] output size mismatch: got %d, expected %d\n",
                actual_n, n_coeffs);
            return -1;
        }

        const double* raw = output.data_ptr<double>();
        for (int i = 0; i < n_coeffs; ++i)
            coeff_out[i] = raw[i];

        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[inr_torch_infer] %s\n", e.what());
        return -1;
    } catch (...) {
        std::fprintf(stderr, "[inr_torch_infer] unknown exception\n");
        return -1;
    }
}

} /* extern "C" */
