"""Pure-Python KD-Tree for KNN, ported from scikit-learn.

Reference files:
  sklearn/neighbors/_binary_tree.pxi.tp   (tree build / query / heap query)
  sklearn/neighbors/_kd_tree.pyx.tp        (node bounds / min_rdist)
  sklearn/neighbors/_partition_nodes.pyx   (nth_element partition)
  sklearn/utils/_heap.pyx                  (fixed-size max-heap push)
  sklearn/utils/_sorting.pyx               (simultaneous introsort)

scikit-learn implements the hot loops in Cython; here they are translated to
plain Python while keeping the exact same *algorithm* (the data layout, the
branch-and-bound traversal, the reduced distances and the fixed-size heap).
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Distance metrics (reduced distance = quantity used to rank neighbours)
# ---------------------------------------------------------------------------

def rdist(x1, x2, p):
    """Reduced distance between two 1D points for the Minkowski p-norm."""
    diff = np.abs(x1 - x2)
    if p == np.inf:
        return float(diff.max()) if diff.size else 0.0
    return float(np.sum(diff ** p))


def dist(x1, x2, p):
    """True distance between two 1D points for the Minkowski p-norm."""
    if p == np.inf:
        return rdist(x1, x2, p)
    return rdist(x1, x2, p) ** (1.0 / p)


def rdist_to_dist(r, p):
    """Map reduced distance(s) to true distance(s)."""
    if p == np.inf:
        return r
    return r ** (1.0 / p)


def min_rdist(node_bounds, i_node, pt, p):
    """Minimum reduced distance between a point and a KD-Tree node.

    Mirrors ``min_rdist`` from ``_kd_tree.pyx.tp``:

        d_lo = lower[j] - pt[j]
        d_hi = pt[j] - upper[j]
        d    = (d_lo + |d_lo|) + (d_hi + |d_hi|)   # == 2 * gap
        rdist += (0.5 * d) ** p
    """
    lo = node_bounds[0, i_node]
    hi = node_bounds[1, i_node]
    d_lo = lo - pt
    d_hi = pt - hi
    if p == np.inf:
        gap = np.maximum(np.maximum(d_lo, d_hi), 0.0)
        return float(gap.max()) if gap.size else 0.0
    gap = np.maximum(np.maximum(d_lo, d_hi), 0.0)
    return float(np.sum(gap ** p))


# ---------------------------------------------------------------------------
# Fixed-size max-heap storing (distance, index), one row per query point
# ---------------------------------------------------------------------------

def heap_push(values, indices, val, val_idx):
    """Port of ``heap_push`` (sklearn/utils/_heap.pyx).

    ``values`` / ``indices`` are the 1D rows of the neighbour heap.  Values are
    kept in a max-heap so the root always holds the current worst neighbour,
    which makes branch pruning a single comparison.
    """
    size = values.shape[0]
    if val >= values[0]:
        return

    values[0] = val
    indices[0] = val_idx

    current_idx = 0
    while True:
        left_child_idx = 2 * current_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= size:
            break
        elif right_child_idx >= size:
            if values[left_child_idx] > val:
                swap_idx = left_child_idx
            else:
                break
        elif values[left_child_idx] >= values[right_child_idx]:
            if val < values[left_child_idx]:
                swap_idx = left_child_idx
            else:
                break
        else:
            if val < values[right_child_idx]:
                swap_idx = right_child_idx
            else:
                break

        values[current_idx] = values[swap_idx]
        indices[current_idx] = indices[swap_idx]
        current_idx = swap_idx

    values[current_idx] = val
    indices[current_idx] = val_idx


class NeighborsHeap:
    """Port of ``NeighborsHeap`` - n_pts independent size-``n_nbrs`` max-heaps."""

    def __init__(self, n_pts, n_nbrs):
        self.distances = np.full((n_pts, n_nbrs), np.inf, dtype=np.float64)
        self.indices = np.zeros((n_pts, n_nbrs), dtype=np.intp)

    def largest(self, row):
        return self.distances[row, 0]

    def push(self, row, val, i_val):
        heap_push(self.distances[row], self.indices[row], val, i_val)

    def get_arrays(self, sort=True):
        if sort:
            for row in range(self.distances.shape[0]):
                _simultaneous_sort(self.distances[row], self.indices[row])
        return self.distances, self.indices


# ---------------------------------------------------------------------------
# Simultaneous introsort (sklearn/utils/_sorting.pyx)
# ---------------------------------------------------------------------------

def _swap(values, indices, i, j):
    values[i], values[j] = values[j], values[i]
    indices[i], indices[j] = indices[j], indices[i]


def _simultaneous_sort(values, indices):
    """Sort ``values`` ascending, permuting ``indices`` identically."""
    n = values.shape[0]
    if n == 0:
        return
    maxd = 2 * int(math.log2(n)) if n > 1 else 0
    _introsort_2way(values, indices, 0, n, maxd)


def _insertion_sort(values, indices, lo, hi):
    for i in range(lo + 1, hi):
        temp_val = values[i]
        temp_idx = indices[i]
        j = i
        while j > lo and values[j - 1] > temp_val:
            values[j] = values[j - 1]
            indices[j] = indices[j - 1]
            j -= 1
        values[j] = temp_val
        indices[j] = temp_idx


def _inplace_median3(values, indices, lo, hi):
    pivot_idx = lo + (hi - lo) // 2
    if values[lo] > values[hi - 1]:
        _swap(values, indices, lo, hi - 1)
    if values[hi - 1] > values[pivot_idx]:
        _swap(values, indices, hi - 1, pivot_idx)
        if values[lo] > values[hi - 1]:
            _swap(values, indices, lo, hi - 1)
    return values[hi - 1]


def _sift_down(values, indices, lo, root, end):
    """Restore a max-heap inside ``[lo, lo + end)``.

    ``root`` and ``end`` are relative to ``lo``.  The Cython implementation
    passes an offset pointer to heapsort; keeping the offset explicit avoids
    accidentally heapifying the beginning of the whole array when introsort
    falls back on a non-zero sub-array.
    """
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and values[lo + maxind] < values[lo + child]:
            maxind = child
        if child + 1 < end and values[lo + maxind] < values[lo + child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        _swap(values, indices, lo + root, lo + maxind)
        root = maxind


def _heapsort(values, indices, lo, hi):
    n = hi - lo
    for start in range((n - 2) // 2, -1, -1):
        _sift_down(values, indices, lo, start, n)

    for end in range(n - 1, 0, -1):
        _swap(values, indices, lo, lo + end)
        _sift_down(values, indices, lo, 0, end)


def _introsort_2way(values, indices, lo, hi, maxd):
    n = hi - lo
    while n > 15:
        if maxd <= 0:
            _heapsort(values, indices, lo, hi)
            return
        maxd -= 1

        pivot = _inplace_median3(values, indices, lo, hi)

        i = lo + 1
        j = hi - 2
        while True:
            while i <= j and values[i] < pivot:
                i += 1
            while i <= j and values[j] > pivot:
                j -= 1
            if i >= j:
                break
            _swap(values, indices, i, j)
            i += 1
            j -= 1

        pivot_idx = i
        _swap(values, indices, pivot_idx, hi - 1)

        _introsort_2way(values, indices, lo, pivot_idx, maxd)

        lo = pivot_idx + 1
        n = hi - lo

    _insertion_sort(values, indices, lo, hi)


# ---------------------------------------------------------------------------
# Linear-time partition used while building the tree (nth_element)
# ---------------------------------------------------------------------------

def _partition_node_indices(data, idx_array, start, end, split_dim, split_index):
    """Rearrange ``idx_array[start:end]`` around ``split_index``.

    Equivalent to sklearn's ``partition_node_indices`` (std::nth_element) using
    the comparator ``(data[i, dim], i)`` so the partition is stable with
    respect to the original indices.  Implemented as quickselect -> O(n).
    """
    sub = list(idx_array[start:end])
    k = split_index

    def less(a, b):
        va, vb = data[a, split_dim], data[b, split_dim]
        if va == vb:
            return a < b
        return va < vb

    lo, hi = 0, len(sub) - 1
    while lo < hi:
        pivot = sub[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while less(sub[i], pivot):
                i += 1
            while less(pivot, sub[j]):
                j -= 1
            if i <= j:
                sub[i], sub[j] = sub[j], sub[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break

    idx_array[start:end] = sub


# ---------------------------------------------------------------------------
# KD-Tree
# ---------------------------------------------------------------------------

VALID_METRICS = ("euclidean", "manhattan", "chebyshev", "minkowski")


class KDTree:
    """KD-Tree with a depth-first branch-and-bound k-NN query.

    Mirrors ``BinaryTree`` (``_binary_tree.pxi.tp``) specialised by
    ``_kd_tree.pyx.tp``.  Nodes are stored in flat arrays: the children of node
    ``i`` are ``2 * i + 1`` and ``2 * i + 2``.
    """

    def __init__(self, data, leaf_size=40, metric="minkowski", p=2):
        self.data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
        if self.data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if self.data.shape[0] == 0:
            raise ValueError("X is an empty array")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("data must contain only finite values")
        if not isinstance(leaf_size, (int, np.integer)) or isinstance(
            leaf_size, (bool, np.bool_)
        ) or leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")

        if metric == "euclidean":
            p = 2.0
        elif metric == "manhattan":
            p = 1.0
        elif metric == "chebyshev":
            p = np.inf
        elif metric == "minkowski":
            p = float(p)
            if p < 1 or np.isnan(p):
                raise ValueError("KDTree requires Minkowski p >= 1")
        else:
            raise ValueError("unsupported KDTree metric: %r" % metric)

        self.leaf_size = int(leaf_size)
        self.metric = metric
        self.p = float(p)

        n_samples, n_features = self.data.shape
        self.n_features = n_features

        # Number of levels and nodes of a balanced tree (see module docs).
        self.n_levels = int(
            math.log2(max(1.0, (n_samples - 1) / self.leaf_size)) + 1
        )
        self.n_nodes = 2 ** self.n_levels - 1

        self.idx_array = np.arange(n_samples, dtype=np.intp)
        self.node_bounds = np.zeros((2, self.n_nodes, n_features), dtype=np.float64)
        self.radius = np.zeros(self.n_nodes, dtype=np.float64)
        self.is_leaf = np.zeros(self.n_nodes, dtype=bool)
        self.idx_start = np.zeros(self.n_nodes, dtype=np.intp)
        self.idx_end = np.zeros(self.n_nodes, dtype=np.intp)

        self._recursive_build(0, 0, n_samples)

    # -- build ------------------------------------------------------------

    def _init_node(self, i_node, idx_start, idx_end):
        lower = self.node_bounds[0, i_node]
        upper = self.node_bounds[1, i_node]
        lower[:] = np.inf
        upper[:] = -np.inf

        points = self.data[self.idx_array[idx_start:idx_end]]
        lower[:] = points.min(axis=0)
        upper[:] = points.max(axis=0)

        half = 0.5 * (upper - lower)
        if self.p == np.inf:
            rad = float(half.max()) if half.size else 0.0
        else:
            rad = float(np.sum(half ** self.p)) ** (1.0 / self.p)
        self.radius[i_node] = rad
        self.idx_start[i_node] = idx_start
        self.idx_end[i_node] = idx_end

    def _find_node_split_dim(self, idx_start, idx_end):
        points = self.data[self.idx_array[idx_start:idx_end]]
        spread = points.max(axis=0) - points.min(axis=0)
        return int(np.argmax(spread))

    def _recursive_build(self, i_node, idx_start, idx_end):
        self._init_node(i_node, idx_start, idx_end)
        n_points = idx_end - idx_start

        if 2 * i_node + 1 >= self.n_nodes:
            self.is_leaf[i_node] = True
        elif n_points < 2:
            self.is_leaf[i_node] = True
        else:
            self.is_leaf[i_node] = False
            split_dim = self._find_node_split_dim(idx_start, idx_end)
            n_mid = n_points // 2
            _partition_node_indices(
                self.data, self.idx_array, idx_start, idx_end, split_dim, n_mid
            )
            self._recursive_build(2 * i_node + 1, idx_start, idx_start + n_mid)
            self._recursive_build(2 * i_node + 2, idx_start + n_mid, idx_end)

    # -- query ------------------------------------------------------------

    def _rdist(self, i_point, pt):
        return rdist(self.data[i_point], pt, self.p)

    def _query_single_depthfirst(self, i_node, pt, i_pt, heap, reduced_dist_lb):
        """Recursive single-tree k-NN query, depth-first (branch and bound)."""
        if reduced_dist_lb > heap.largest(i_pt):
            return  # Case 1: point is outside node radius -> trim

        if self.is_leaf[i_node]:
            # Case 2: leaf -> update the neighbour heap
            for i in range(self.idx_start[i_node], self.idx_end[i_node]):
                idx = self.idx_array[i]
                heap.push(i_pt, self._rdist(idx, pt), idx)
            return

        # Case 3: interior node -> visit the closer child first
        i1 = 2 * i_node + 1
        i2 = i1 + 1
        lb1 = min_rdist(self.node_bounds, i1, pt, self.p)
        lb2 = min_rdist(self.node_bounds, i2, pt, self.p)
        if lb1 <= lb2:
            self._query_single_depthfirst(i1, pt, i_pt, heap, lb1)
            self._query_single_depthfirst(i2, pt, i_pt, heap, lb2)
        else:
            self._query_single_depthfirst(i2, pt, i_pt, heap, lb2)
            self._query_single_depthfirst(i1, pt, i_pt, heap, lb1)

    def query(self, X, k=1, return_distance=True):
        X = np.asarray(X, dtype=np.float64)
        n_features = self.data.shape[1]
        if X.ndim == 0 or X.shape[-1] != n_features:
            raise ValueError("query data dimension must match training data dimension")
        if not np.all(np.isfinite(X)):
            raise ValueError("query data must contain only finite values")
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
            raise ValueError("k must be a positive integer")
        Xarr = X.reshape((-1, n_features))

        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal to the number of training points")

        heap = NeighborsHeap(Xarr.shape[0], k)
        for i in range(Xarr.shape[0]):
            pt = Xarr[i]
            lb = min_rdist(self.node_bounds, 0, pt, self.p)
            self._query_single_depthfirst(0, pt, i, heap, lb)

        distances, indices = heap.get_arrays(sort=True)

        shape = X.shape[:-1] + (k,)
        if return_distance:
            distances = rdist_to_dist(distances, self.p)
            return distances.reshape(shape), indices.reshape(shape)
        return indices.reshape(shape)
