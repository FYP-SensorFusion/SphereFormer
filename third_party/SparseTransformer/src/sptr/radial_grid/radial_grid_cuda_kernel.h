#ifndef _RADIAL_GRID_CUDA_KERNEL
#define _RADIAL_GRID_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>


void compute_radial_grid_cuda(torch::Tensor pos, const float radius, torch::Tensor& out);

#ifdef __cplusplus
extern "C" {
#endif

void radial_grid_cuda_launcher(torch::Tensor pos, const float radius, torch::Tensor out);

#ifdef __cplusplus
}
#endif
#endif