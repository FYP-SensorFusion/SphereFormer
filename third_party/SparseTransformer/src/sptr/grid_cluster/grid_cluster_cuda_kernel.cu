#include "../cuda_utils.h"
#include "grid_cluster_cuda_kernel.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void grid_cluster_kernel(const scalar_t *pos, const scalar_t *size,
                            const scalar_t *start, const scalar_t *end,
                            int64_t *out, int64_t D, int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t c = 0, k = 1;
    for (int64_t d = 0; d < D; d++) {
      scalar_t p = pos[thread_idx * D + d] - start[d];
      c += (int64_t)(p / size[d]) * k;
      k *= (int64_t)((end[d] - start[d]) / size[d]);
    }
    out[thread_idx] = c + 1;
  }
}

torch::Tensor grid_cluster_launcher(torch::Tensor pos, torch::Tensor size,
                        torch::optional<torch::Tensor> optional_start,
                        torch::optional<torch::Tensor> optional_end) {
  assert(pos.is_cuda());  // Replaces CHECK_CUDA(pos)
  assert(size.is_cuda());  // Replaces CHECK_CUDA(size)
  cudaSetDevice(pos.get_device());

  if (optional_start.has_value())
    assert(optional_start.value().is_cuda());  // Replaces CHECK_CUDA(optional_start.value())
  if (optional_start.has_value())
    assert(optional_start.value().is_cuda());  // Replaces CHECK_CUDA(optional_start.value())

  pos = pos.view({pos.size(0), -1}).contiguous();
  size = size.contiguous();

  assert(size.numel() == pos.size(1));  // Replaces CHECK_INPUT(size.numel() == pos.size(1))

  if (!optional_start.has_value())
    optional_start = std::get<0>(pos.min(0));
  else {
    optional_start = optional_start.value().contiguous();
    assert(optional_start.value().numel() == pos.size(1));  // Replaces CHECK_INPUT(optional_start.value().numel() == pos.size(1))
  }

  if (!optional_end.has_value())
    optional_end = std::get<0>(pos.max(0));
  else {
    optional_start = optional_start.value().contiguous();
    assert(optional_start.value().numel() == pos.size(1));  // Replaces CHECK_INPUT(optional_start.value().numel() == pos.size(1))
  }

  auto start = optional_start.value();
  auto end = optional_end.value();

  auto out = torch::empty({pos.size(0)}, pos.options().dtype(torch::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, pos.scalar_type(), "_", [&] {
    grid_cluster_kernel<scalar_t><<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
        pos.data_ptr<scalar_t>(), size.data_ptr<scalar_t>(),
        start.data_ptr<scalar_t>(), end.data_ptr<scalar_t>(),
        out.data_ptr<int64_t>(), pos.size(1), out.numel());
  });

  return out;
}
