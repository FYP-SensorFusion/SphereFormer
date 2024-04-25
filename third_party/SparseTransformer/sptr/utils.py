import numbers
import torch
import numpy as np
from torch_scatter import segment_csr, gather_csr
from torch_geometric.nn import voxel_grid
from . import precompute_all


# import torch
import sptr_cuda
from torch_geometric.utils.repeat import repeat
    

def to_3d_numpy(size):
    if isinstance(size, numbers.Number):
        size = np.array([size, size, size]).astype(np.float32)
    elif isinstance(size, list):
        size = np.array(size)
    elif isinstance(size, np.ndarray):
        size = size
    else:
        raise ValueError("size is either a number, or a list, or a np.ndarray")
    return size

def grid_sample(pos, batch, size, start, return_p2v=True, return_counts=True, return_unique=False):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None
    # print("=============pos==================",pos)
    # cluster = voxel_grid(pos, batch, size, start=start) #[N, ]
    # print("previous_cluster",cluster)
    # print("previous_cluster_shape",cluster.shape)
    # cluster = voxel_grid_new(pos, batch, size, start=start) #[N, ]
    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]
    # print("new_cluster", cluster2)
    # print("new_cluster_shape", cluster2.shape)
    # cluster_radial_grid = voxel_radial_grid(pos, batch, size, start=start) #[N, ]
    # print("cluster_grid ================== ", cluster)
    # print("cluster_radial_grid ================== ", cluster_radial_grid)
    # print("cluster_grid_shape ================== ",cluster.shape)
    # print("cluster_radial_grid_shape ================== ",cluster_radial_grid.shape)


    if return_p2v == False and return_counts == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        # unique2, cluster2 = torch.unique(cluster2, sorted=True, return_inverse=True)
        # print("unique _ cluster ",unique.shape, cluster.shape)
        # print("unique _ cluster 2 ",unique2.shape, cluster2.shape)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
    # unique2, cluster2, counts2 = torch.unique(cluster2, sorted=True, return_inverse=True, return_counts=True)
    # print("unique _ cluster _ count",unique.shape, cluster.shape, counts.shape)
    # print("unique _ cluster _ count 2",unique2.shape, cluster2.shape, counts2.shape)

    if return_p2v == False and return_counts == True:
        return cluster, counts.max().item(), counts

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)

    if return_unique:
        return cluster, p2v_map, counts, unique

    return cluster, p2v_map, counts

def grid_sample_ellipsoidal(pos, batch, size, start, return_p2v=True, return_counts=True, return_unique=False):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None
    # print("=============pos==================",pos)

    cluster = voxel_grid_ellipsoidal(pos, batch, size, start=start) #[N, ]

    if return_p2v == False and return_counts == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        # unique2, cluster2 = torch.unique(cluster2, sorted=True, return_inverse=True)
        # print("unique _ cluster ",unique.shape, cluster.shape)
        # print("unique _ cluster 2 ",unique2.shape, cluster2.shape)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
    # unique2, cluster2, counts2 = torch.unique(cluster2, sorted=True, return_inverse=True, return_counts=True)
    # print("unique _ cluster _ count",unique.shape, cluster.shape, counts.shape)
    # print("unique _ cluster _ count 2",unique2.shape, cluster2.shape, counts2.shape)

    if return_p2v == False and return_counts == True:
        return cluster, counts.max().item(), counts

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)

    if return_unique:
        return cluster, p2v_map, counts, unique

    return cluster, p2v_map, counts

def get_indices_params(xyz, batch, window_size, shift_win: bool):
    
    if isinstance(window_size, list) or isinstance(window_size, np.ndarray):
        window_size = torch.from_numpy(window_size).type_as(xyz).to(xyz.device)
    else:
        window_size = torch.tensor([window_size]*3).type_as(xyz).to(xyz.device)
    # print("========xyz=============",xyz)
    if shift_win:
        v2p_map, k, counts = grid_sample(xyz+1/2*window_size, batch, window_size, start=xyz.min(0)[0], return_p2v=False, return_counts=True)
    else:
        v2p_map, k, counts = grid_sample(xyz, batch, window_size, start=None, return_p2v=False, return_counts=True)

    v2p_map, sort_idx = v2p_map.sort()

    n = counts.shape[0]
    N = v2p_map.shape[0]

    n_max = k
    # print(v2p_map)
    # print(N,n, n_max, counts)
    index_0_offsets, index_1_offsets, index_0, index_1 = precompute_all(N, n, n_max, counts)
    # print(index_0_offsets, index_1_offsets, index_0, index_1)
    index_0 = index_0.long()
    index_1 = index_1.long()

    return index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx

def get_indices_params_ellipsoidal(xyz, batch, window_size, shift_win: bool):
    
    if isinstance(window_size, list) or isinstance(window_size, np.ndarray):
        window_size = torch.from_numpy(window_size).type_as(xyz).to(xyz.device)
    else:
        window_size = torch.tensor([window_size]*3).type_as(xyz).to(xyz.device)
    # print("========xyz=============",xyz)
    if shift_win:
        v2p_map, k, counts = grid_sample_ellipsoidal(xyz+1/2*window_size, batch, window_size, start=xyz.min(0)[0], return_p2v=False, return_counts=True)
    else:
        v2p_map, k, counts = grid_sample_ellipsoidal(xyz, batch, window_size, start=None, return_p2v=False, return_counts=True)

    v2p_map, sort_idx = v2p_map.sort()

    n = counts.shape[0]
    N = v2p_map.shape[0]

    n_max = k
    # print(v2p_map)
    # print(N,n, n_max, counts)
    index_0_offsets, index_1_offsets, index_0, index_1 = precompute_all(N, n, n_max, counts)
    # print(index_0_offsets, index_1_offsets, index_0, index_1)
    index_0 = index_0.long()
    index_1 = index_1.long()

    return index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx

def scatter_softmax_csr(src: torch.Tensor, indptr: torch.Tensor, dim: int = -1):
    ''' src: (N, C),
        index: (Ni+1, ), [0, n0^2, n0^2+n1^2, ...]
    '''
    max_value_per_index = segment_csr(src, indptr, reduce='max')
    max_per_src_element = gather_csr(max_value_per_index, indptr)
    
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = segment_csr(
        recentered_scores_exp, indptr, reduce='sum')
    
    normalizing_constants = gather_csr(sum_per_index, indptr)

    return recentered_scores_exp.div(normalizing_constants)


def voxel_radial_grid(pos, batch, size, start=None, end=None):
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    num_nodes, dim = pos.size()

    start = start.tolist() if torch.is_tensor(start) else start
    end = end.tolist() if torch.is_tensor(end) else end

    start, end = repeat(start, dim), repeat(end, dim)

    pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
    start = None if start is None else start + [0]
    end = None if end is None else end + [batch.max().item()]

    if start is not None:
        start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
    if end is not None:
        end = torch.tensor(end, dtype=pos.dtype, device=pos.device)

    # Calculate the radius as half of the first element of the size tensor
    radius = size[0] / 2.0

    return sptr_cuda.compute_radial_grid_cuda(pos, radius, start, end)  # Call your function

def voxel_grid_new(pos, batch, size, start=None, end=None) -> torch.Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    num_nodes, dim = pos.size()

    size = size.tolist() if torch.is_tensor(size) else size
    start = start.tolist() if torch.is_tensor(start) else start
    end = end.tolist() if torch.is_tensor(end) else end

    size, start, end = repeat(size, dim), repeat(start, dim), repeat(end, dim)

    pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
    size = size + [1]
    start = None if start is None else start + [0]
    end = None if end is None else end + [batch.max().item()]

    size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    if start is not None:
        start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
    if end is not None:
        end = torch.tensor(end, dtype=pos.dtype, device=pos.device)

    return sptr_cuda.grid_cluster(pos, size, start, end)


def voxel_grid_ellipsoidal(pos, batch, size, start=None, end=None) -> torch.Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    num_nodes, dim = pos.size()

    size = size.tolist() if torch.is_tensor(size) else size
    start = start.tolist() if torch.is_tensor(start) else start
    end = end.tolist() if torch.is_tensor(end) else end

    size, start, end = repeat(size, dim), repeat(start, dim), repeat(end, dim)

    pos = torch.cat([pos, batch.unsqueeze(-1).type_as(pos)], dim=-1)
    size = size + [1]
    start = None if start is None else start + [0]
    end = None if end is None else end + [batch.max().item()]

    size = torch.tensor(size, dtype=pos.dtype, device=pos.device)
    if start is not None:
        start = torch.tensor(start, dtype=pos.dtype, device=pos.device)
    if end is not None:
        end = torch.tensor(end, dtype=pos.dtype, device=pos.device)

    return sptr_cuda.ellipsoidal_cluster(pos, size, start, end)