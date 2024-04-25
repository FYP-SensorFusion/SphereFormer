#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "grid_cluster_cuda_kernel.h"

torch::Tensor grid_cluster(torch::Tensor pos, torch::Tensor size,
                        torch::optional<torch::Tensor> optional_start,
                        torch::optional<torch::Tensor> optional_end) {
                      return grid_cluster_launcher( pos, size, optional_start, optional_end);  
                        }
  