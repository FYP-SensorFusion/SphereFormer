#include "../cuda_utils.h"
#include "ellipsoidal_cluster_cuda_kernel.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void ellipsoidal_cluster_kernel(const scalar_t *pos, const scalar_t *size,
                            const scalar_t *start, const scalar_t *end,
                            int64_t *out,  int64_t D,  int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  float_t x = pos[thread_idx * D + 0], y = pos[thread_idx * D + 1], z = pos[thread_idx * D + 2];
  float_t x_0 = 0, y_0 = 0, z_0 = 0, center = 0;
  float_t a = size[0], b = size[1], c = size[2];
  if (thread_idx < numel) {
    int64_t c = 0, k = 1;
    for (int64_t d = 0; d < D; d++) {
      scalar_t p = pos[thread_idx * D + d] - start[d];
      int64_t steps = p / size[d];
      center = steps * size[d] + size[d]/2;
      if (d==0)
      {
        x_0 = center;
      }
      else if (d==1)
      {
        y_0 = center;
      }else if (d==2){
        z_0 = center;
      }
      c += steps * k;
      k *= (end[d] - start[d]) / size[d];
    }
    float_t equation = ((x-x_0)*(x-x_0) / (a*a)) + ((y-y_0)*(y-y_0) / (b*b)) + ((z-z_0)*(z-z_0) / (c*c));
    if (equation <= 1) {
        out[thread_idx] =c * 9;
    } else {
      if (x - a >= 0 && y - b >= 0 && z - c >= 0) {    // 5
        out[thread_idx] = c * 9 + 4;
      } else if (x - a >= 0 && y - b >= 0 && z - c < 0) {   // 9
        out[thread_idx] = c * 9 + 8;
      } else if (x - a >= 0 && y - b < 0 && z - c >= 0) {   // 3
        out[thread_idx] = c * 9 + 2;
      } else if (x - a >= 0 && y - b < 0 && z - c < 0) {    // 7
        out[thread_idx] = c * 9 + 6;
      } else if (x - a < 0 && y - b >= 0 && z - c >= 0) {   // 4
        out[thread_idx] = c * 9 + 3;
      } else if (x - a < 0 && y - b >= 0 && z - c < 0) {    // 8
        out[thread_idx] = c * 9 + 7;
      } else if (x - a < 0 && y - b < 0 && z - c >= 0) {    // 2
        out[thread_idx] = c * 9 + 1;
      } else if (x - a < 0 && y - b < 0 && z - c < 0) {     // 6
        out[thread_idx] = c * 9 + 5;
      }else{
        out[thread_idx] =c * 9;
      }
    }
  }
}

torch::Tensor ellipsoidal_cluster_launcher(torch::Tensor pos, torch::Tensor size,
                        torch::optional<torch::Tensor> optional_start,
                        torch::optional<torch::Tensor> optional_end) {
  assert(pos.is_cuda());
  assert(size.is_cuda());
  cudaSetDevice(pos.get_device());
  if (optional_start.has_value())
    assert(optional_start.value().is_cuda());
  if (optional_end.has_value())
    assert(optional_end.value().is_cuda());
  pos = pos.view({pos.size(0), -1}).contiguous();
  size = size.contiguous();
  assert(size.numel() == pos.size(1));
  if (!optional_start.has_value())
    optional_start = std::get<0>(pos.min(0));
  else {
    optional_start = optional_start.value().contiguous();
    assert(optional_start.value().numel() == pos.size(1));
  }
  if (!optional_end.has_value())
    optional_end = std::get<0>(pos.max(0));
  else {
    optional_end = optional_end.value().contiguous();
    assert(optional_end.value().numel() == pos.size(1));
  }

  auto start = optional_start.value();
  auto end = optional_end.value();

  auto out = torch::empty({pos.size(0)}, pos.options().dtype(torch::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, pos.scalar_type(), "_", [&] {
    ellipsoidal_cluster_kernel<<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
        pos.data_ptr<scalar_t>(), size.data_ptr<scalar_t>(),
        start.data_ptr<scalar_t>(), end.data_ptr<scalar_t>(),
        out.data_ptr<int64_t>(),  pos.size(1), out.numel());
  });
  return out;
}
