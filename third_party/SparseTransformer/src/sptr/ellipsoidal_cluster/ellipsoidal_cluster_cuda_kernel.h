#ifndef _ELLIPSOIDAL_CLUSTER_CUDA_KERNEL
#define _ELLIPSOIDAL_CLUSTER_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>


torch::Tensor ellipsoidal_cluster_launcher(torch::Tensor pos, torch::Tensor size,
                        torch::optional<torch::Tensor> optional_start,
                        torch::optional<torch::Tensor> optional_end) ;

#ifdef __cplusplus
extern "C" {
#endif


torch::Tensor ellipsoidal_cluster(torch::Tensor pos, torch::Tensor size,
                        torch::optional<torch::Tensor> optional_start,
                        torch::optional<torch::Tensor> optional_end);
#ifdef __cplusplus
}
#endif
#endif