#include "../cuda_utils.h"
#include "radial_grid_cuda_kernel.h"  // Include the correct header file
#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS
#define GRID_RESOLUTION = 1024

template <typename scalar_t>
__global__ void radial_grid_kernel(const scalar_t *pos, const scalar_t radius,
                                   const scalar_t *start, const scalar_t *end,
                                   int64_t *out, int64_t numel)
{
    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < numel)
    {
        // Compute distance from center (assuming pos contains (x, y, z) coordinates)
        scalar_t distance = sqrt(pos[thread_idx * 3] * pos[thread_idx * 3] +
                                 pos[thread_idx * 3 + 1] * pos[thread_idx * 3 + 1] +
                                 pos[thread_idx * 3 + 2] * pos[thread_idx * 3 + 2]);

        // Check if point is within radial shape (inside the sphere)
        if (distance <= radius)
        {
            // Compute voxel indices based on distance from center and radial grid resolution
            out[thread_idx] = static_cast<int64_t>(distance / radius * 1024);
        }
        else
        {
            // Set voxel index to -1 for points outside the radial shape
            out[thread_idx] = -1;
        }
    }
}

void radial_grid_cuda_launcher(torch::Tensor pos, const float radius, torch::Tensor out)
{
    assert(pos.is_cuda());  // Replaces CHECK_CUDA(pos)

    cudaSetDevice(pos.get_device());

    pos = pos.view({pos.size(0), -1}).contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "_", [&]
                               { radial_grid_kernel<scalar_t><<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
                                     pos.data_ptr<scalar_t>(), radius, nullptr, nullptr, out.data_ptr<int64_t>(), pos.size(0)); });

    cudaDeviceSynchronize();
}
