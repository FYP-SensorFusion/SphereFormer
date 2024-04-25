#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    name='sptr',
    ext_modules=[
        CUDAExtension('sptr_cuda', [
            'src/sptr/pointops_api.cpp',
            'src/sptr/attention/attention_cuda.cpp',
            'src/sptr/attention/attention_cuda_kernel.cu',
            'src/sptr/precompute/precompute.cpp',
            'src/sptr/precompute/precompute_cuda_kernel.cu',
            'src/sptr/rpe/relative_pos_encoding_cuda.cpp',
            'src/sptr/rpe/relative_pos_encoding_cuda_kernel.cu',
            'src/sptr/radial_grid/radial_grid_cuda.cpp',
            'src/sptr/radial_grid/radial_grid_cuda_kernel.cu',
            'src/sptr/grid_cluster/grid_cluster_cuda.cpp',
            'src/sptr/grid_cluster/grid_cluster_cuda_kernel.cu',
            'src/sptr/ellipsoidal_cluster/ellipsoidal_cluster_cuda.cpp',
            'src/sptr/ellipsoidal_cluster/ellipsoidal_cluster_cuda_kernel.cu',
            ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
