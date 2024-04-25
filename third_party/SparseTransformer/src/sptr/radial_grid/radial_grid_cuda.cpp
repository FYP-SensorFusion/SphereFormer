#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "radial_grid_cuda_kernel.h"  


void compute_radial_grid_cuda(torch::Tensor pos, const float radius, torch::Tensor& out) {
    radial_grid_cuda_launcher(pos, radius, out);
}

