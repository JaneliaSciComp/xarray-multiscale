import numba as nb
import numpy as np
from typing import Tuple


@nb.njit
def index_to_coords(idx: int, shape: Tuple[int]):
    ndim = len(shape)
    result = np.zeros(ndim, dtype="int")
    init = idx
    strides = make_strides(shape)
    for region_idx in range(0, ndim - 1):
        result[region_idx] = init // strides[region_idx]
        init -= result[region_idx] * strides[region_idx]
    result[-1] = init
    return result


@nb.njit
def make_strides(shape: Tuple[int]):
    ndim = len(shape)
    result = np.ones(ndim, dtype="int")
    for d in range(ndim - 2, -1, -1):
        result[d] = shape[d + 1] * result[d + 1]
    return result


@nb.njit
def create_stencil(array_shape: Tuple[int], region_shape: Tuple[int]):
    ndim = len(array_shape)

    result_shape = 1
    for x in region_shape:
        result_shape *= x

    array_strides = make_strides(array_shape)

    result = np.zeros(result_shape, dtype="int64")

    for lidx in range(0, result_shape):
        shift = 0
        region_idx = index_to_coords(lidx, region_shape)
        for c in range(ndim):
            shift += region_idx[c] * array_strides[c]
        result[lidx] = shift
    return result


@nb.njit
def mean_reduction(v):
    return v.mean()


@nb.jit
def reduce(arr, region_shape: Tuple[int], reduction):
    array_shape = np.array(arr.shape)
    region_shape = np.array(region_shape)
    region_size = np.prod(region_shape)
    flat = arr.ravel()
    stencil = create_stencil(array_shape, region_shape)

    num_reductions = int(np.floor(np.log(array_shape) / np.log(region_shape)).min())

    partitions = np.zeros(num_reductions, dtype="int")
    partitions[0] = len(flat) // region_size

    for n in range(1, num_reductions):
        partitions[n] = partitions[n - 1] // region_size

    output = np.zeros(partitions.sum())

    region_grid_shape = array_shape // region_shape
    array_strides = make_strides(array_shape)
    region_grid_strides = array_strides * region_shape

    for idx in range(partitions[0]):
        region_coord = index_to_coords(idx, region_grid_shape)
        stencil_shift = (region_coord * region_grid_strides).sum()
        output[idx] = reduction(flat[stencil + stencil_shift])

    return output
