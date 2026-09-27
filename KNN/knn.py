"""Manual K-Nearest-Neighbors classifier mirroring scikit-learn.

Reference files:
  sklearn/neighbors/_base.py           (NeighborsBase, KNeighborsMixin, _get_weights)
  sklearn/neighbors/_classification.py (KNeighborsClassifier.predict/predict_proba)
  sklearn/utils/extmath.py             (weighted_mode)

Algorithmic choices follow scikit-learn:
  * ``algorithm='auto'`` picks a KD-Tree for low-dimensional data and a
    vectorised brute-force search otherwise;
  * brute force ranks neighbours with ``np.argpartition`` (O(n) selection per
    query instead of a full sort) and uses the ``|x|^2 + |y|^2 - 2xy`` trick for
    the squared Euclidean distance;
  * predictions use the class-index encoding and the exact mode/weighted-mode
    tie-breaking of scikit-learn.
"""

import numpy as np

from .kdtree import KDTree, VALID_METRICS

KD_VALID_METRICS = set(VALID_METRICS)


# ---------------------------------------------------------------------------
# Small helpers ported from scikit-learn
# ---------------------------------------------------------------------------

def _get_weights(dist, weights):
    """Port of ``sklearn.neighbors._base._get_weights``."""
    if weights in (None, "uniform"):
        return None

    if weights == "distance":
        # A zero distance means the query point is a training point: it gets
        # weight 1.0 and every other neighbour 0.0 (see sklearn source).
        with np.errstate(divide="ignore"):
            dist = 1.0 / dist
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return dist

    if callable(weights):
        return weights(dist)

    raise ValueError("weights must be 'uniform', 'distance', callable or None")


def weighted_mode(a, w, axis=0):
    """Port of ``sklearn.utils.extmath.weighted_mode``.

    Returns the weighted modal value.  Ties are resolved in favour of the
    smallest label because the strict ``counts > oldcounts`` comparison keeps
    the first (sorted) candidate encountered.
    """
    a = np.asarray(a)
    w = np.asarray(w)
    if a.shape != w.shape:
        w = np.full(a.shape, w, dtype=w.dtype)

    scores = np.unique(np.ravel(a))
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape, dtype=a.dtype)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = np.zeros(a.shape)
        ind = a == score
        template[ind] = w[ind]
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts


def _mode(a, axis=0):
    """Tie-safe mode (equivalent to ``scipy.stats.mode`` with keepdims)."""
    return weighted_mode(a, np.ones(np.asarray(a).shape), axis=axis)


# ---------------------------------------------------------------------------
# Brute-force nearest-neighbour search
# ---------------------------------------------------------------------------

def _pairwise_reduced_distances(X, Y, metric, p):
    """Reduced distances from every row of X to every row of Y.

    Roots are monotonic and therefore unnecessary while selecting neighbours.
    Euclidean returns squared distance, and general Minkowski returns the sum
    of p-th powers.  Conversion to true distances happens only for the final k
    neighbours and only when the caller asks for distances.
    """
    if metric == "euclidean":
        XX = np.einsum("ij,ij->i", X, X)[:, None]
        YY = np.einsum("ij,ij->i", Y, Y)[None, :]
        D = X @ Y.T
        D *= -2.0
        D += XX
        D += YY
        np.maximum(D, 0, out=D)
        return D

    diff = np.subtract(X[:, None, :], Y[None, :, :])
    np.abs(diff, out=diff)
    if metric == "manhattan":
        D = diff.sum(axis=2)
    elif metric == "chebyshev":
        D = diff.max(axis=2)
    elif metric == "minkowski":
        np.power(diff, p, out=diff)
        D = diff.sum(axis=2)
    else:
        raise ValueError("unsupported metric for brute force: %r" % metric)
    return D


def _rdist_to_dist(distances, metric, p):
    if metric == "euclidean":
        return np.sqrt(distances)
    if metric == "minkowski":
        return distances ** (1.0 / p)
    return distances


def _chunk_sizes(n_queries, n_fit, n_features, metric, working_memory):
    """Choose query and training chunk sizes from an approximate byte budget."""
    if working_memory <= 0:
        raise ValueError("working_memory must be greater than zero")

    # Non-Euclidean metrics materialise a q x y x n_features difference
    # tensor.  Euclidean needs only a q x y reduced-distance matrix.  Reserve
    # an extra matrix for top-k merging and temporary ufunc output.
    values_per_pair = 3 if metric == "euclidean" else max(n_features + 2, 3)
    pair_budget = max(1, int(working_memory) // (8 * values_per_pair))
    query_chunk = min(n_queries, max(1, int(np.sqrt(pair_budget))))
    fit_chunk = min(n_fit, max(1, pair_budget // query_chunk))
    return query_chunk, fit_chunk


def _argkmin_chunks(X, Y, k, metric, p, working_memory):
    """Yield reduced-distance top-k results using two-dimensional blocking."""
    n_queries = X.shape[0]
    n_fit = Y.shape[0]
    if n_queries == 0 or n_fit == 0:
        raise ValueError("X and Y must each contain at least one sample")
    n_features = Y.shape[1]
    query_chunk, fit_chunk = _chunk_sizes(
        n_queries, n_fit, n_features, metric, working_memory
    )

    for start in range(0, n_queries, query_chunk):
        stop = min(start + query_chunk, n_queries)
        Xc = X[start:stop]
        n_chunk_queries = Xc.shape[0]
        best_dist = np.full((n_chunk_queries, k), np.inf, dtype=np.float64)
        best_ind = np.full((n_chunk_queries, k), -1, dtype=np.intp)

        for fit_start in range(0, n_fit, fit_chunk):
            fit_stop = min(fit_start + fit_chunk, n_fit)
            distances = _pairwise_reduced_distances(
                Xc, Y[fit_start:fit_stop], metric, p
            )
            block_ind = np.broadcast_to(
                np.arange(fit_start, fit_stop, dtype=np.intp), distances.shape
            )
            candidate_dist = np.concatenate((best_dist, distances), axis=1)
            candidate_ind = np.concatenate((best_ind, block_ind), axis=1)

            rows = np.arange(n_chunk_queries)[:, None]
            selected = np.argpartition(candidate_dist, k - 1, axis=1)[:, :k]
            best_dist = candidate_dist[rows, selected]
            best_ind = candidate_ind[rows, selected]

        rows = np.arange(n_chunk_queries)[:, None]
        order = np.argsort(best_dist, axis=1)
        yield start, stop, best_dist[rows, order], best_ind[rows, order]


def argkmin(X, Y, k, metric, p, return_distance=True, working_memory=1e6):
    """Vectorised brute-force k-NN search.

    Mirrors ``_kneighbors_reduce_func``: rank with ``np.argpartition``
    (O(n) selection) then sort only the k selected neighbours.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2D arrays with the same feature count")

    n_queries = X.shape[0]
    n_fit = Y.shape[0]
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError("k must be a positive integer")
    if k > n_fit:
        raise ValueError("k must be less than or equal to the number of training points")

    neigh_ind = np.empty((n_queries, k), dtype=np.intp)
    neigh_dist = np.empty((n_queries, k), dtype=np.float64) if return_distance else None

    for start, stop, reduced_dist, indices in _argkmin_chunks(
        X, Y, k, metric, p, working_memory
    ):
        neigh_ind[start:stop] = indices
        if return_distance:
            neigh_dist[start:stop] = _rdist_to_dist(reduced_dist, metric, p)

    if return_distance:
        return neigh_dist, neigh_ind
    return neigh_ind


def argkmin_class_mode(
    X, Y, y, k, n_classes, metric, p, weights, working_memory=1e6
):
    """Fused blocked neighbour reduction and class-probability accumulation."""
    probabilities = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for start, stop, reduced_dist, indices in _argkmin_chunks(
        X, Y, k, metric, p, working_memory
    ):
        labels = y[indices]
        if weights in (None, "uniform"):
            chunk_weights = np.ones_like(reduced_dist)
        else:
            distances = _rdist_to_dist(reduced_dist, metric, p)
            chunk_weights = _get_weights(distances, weights)
            if np.any(np.all(chunk_weights == 0, axis=1)):
                raise ValueError("all neighbors of a sample have zero weight")

        chunk_proba = probabilities[start:stop]
        rows = np.arange(stop - start)[:, None]
        np.add.at(chunk_proba, (rows, labels), chunk_weights)
        chunk_proba /= chunk_proba.sum(axis=1, keepdims=True)
    return probabilities


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class KNeighborsClassifier:
    """K-nearest neighbours vote classifier (scikit-learn compatible API)."""

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    # -- effective metric -------------------------------------------------

    def _resolve_metric(self):
        metric_params = {} if self.metric_params is None else dict(self.metric_params)
        metric = self.metric
        if metric not in KD_VALID_METRICS:
            raise ValueError(
                "unsupported metric %r; expected one of %s"
                % (metric, sorted(KD_VALID_METRICS))
            )

        if metric == "minkowski":
            eff_p = metric_params.pop("p", self.p)
            if metric_params.pop("w", None) is not None:
                raise ValueError("weighted Minkowski distance is not implemented")
            if metric_params:
                raise ValueError(
                    "unsupported metric_params for Minkowski: %s"
                    % sorted(metric_params)
                )
            try:
                eff_p = float(eff_p)
            except (TypeError, ValueError) as exc:
                raise ValueError("p must be a positive number") from exc
            if eff_p <= 0 or np.isnan(eff_p):
                raise ValueError("p must be a positive number")
            if eff_p == 1:
                metric = "manhattan"
            elif eff_p == 2:
                metric = "euclidean"
            elif eff_p == np.inf:
                metric = "chebyshev"
        else:
            if metric_params:
                raise ValueError(
                    "metric_params are only supported for metric='minkowski'"
                )
            # Named metrics have fixed norms; sklearn ignores the estimator's
            # standalone p parameter for these aliases.
            eff_p = {
                "manhattan": 1.0,
                "euclidean": 2.0,
                "chebyshev": np.inf,
            }[metric]

        self.effective_metric_ = metric
        self.effective_p_ = eff_p
        self.effective_metric_params_ = (
            {"p": eff_p} if metric == "minkowski" else {}
        )

    def _check_algorithm(self, X):
        n_samples, n_features = X.shape
        fit_method = self.algorithm

        if fit_method == "auto":
            # Same heuristic as sklearn: trees win for few features/neighbours.
            if (
                n_features > 15
                or (self.n_neighbors is not None and self.n_neighbors >= n_samples // 2)
                or (
                    self.effective_metric_ == "minkowski"
                    and self.effective_p_ < 1
                )
            ):
                fit_method = "brute"
            elif self.effective_metric_ in KD_VALID_METRICS:
                fit_method = "kd_tree"
            else:
                fit_method = "brute"
        elif fit_method == "kd_tree":
            if self.effective_metric_ not in KD_VALID_METRICS:
                raise ValueError(
                    "metric %r is not valid for algorithm=%r"
                    % (self.effective_metric_, fit_method)
                )
            if self.effective_metric_ == "minkowski" and self.effective_p_ < 1:
                raise ValueError("KD-Tree requires Minkowski p >= 1")
        elif fit_method == "ball_tree":
            raise ValueError(
                "algorithm='ball_tree' is not implemented; use 'auto', "
                "'kd_tree', or 'brute'"
            )
        elif fit_method != "brute":
            raise ValueError("unknown algorithm: %r" % self.algorithm)

        self._fit_method = fit_method

    # -- fit --------------------------------------------------------------

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if y.ndim != 1:
            raise ValueError("only single-output 1D classification targets are supported")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y contain different numbers of samples")
        if not isinstance(self.n_neighbors, (int, np.integer)) or isinstance(
            self.n_neighbors, (bool, np.bool_)
        ) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors must not exceed the number of samples")
        if not isinstance(self.leaf_size, (int, np.integer)) or isinstance(
            self.leaf_size, (bool, np.bool_)
        ) or self.leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        if not (
            self.weights in (None, "uniform", "distance")
            or callable(self.weights)
        ):
            raise ValueError("weights must be 'uniform', 'distance', callable or None")

        self._resolve_metric()
        self._check_algorithm(X)

        # Classification targets: encode to contiguous integer class indices.
        self.classes_, self._y = np.unique(y, return_inverse=True)
        self._y = self._y.astype(np.intp)
        self.outputs_2d_ = False

        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]

        if self._fit_method == "kd_tree":
            self._tree = KDTree(
                X, leaf_size=self.leaf_size,
                metric=self.effective_metric_, p=self.effective_p_,
            )
            self._fit_X = self._tree.data
        else:
            self._tree = None
            self._fit_X = X
        return self

    # -- neighbour search -------------------------------------------------

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if not hasattr(self, "_fit_method"):
            raise ValueError("estimator is not fitted")
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if not isinstance(n_neighbors, (int, np.integer)) or isinstance(
            n_neighbors, (bool, np.bool_)
        ) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")

        query_is_train = X is None
        if query_is_train:
            X = self._fit_X
            # An extra neighbour accounts for the sample itself (removed below).
            n_neighbors += 1
        else:
            X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.ndim != 2 or X.shape[1] != self.n_features_in_:
                raise ValueError(
                    "X must be a 2D array with %d features" % self.n_features_in_
                )
            if X.shape[0] == 0:
                raise ValueError("X must contain at least one query sample")
            if not np.all(np.isfinite(X)):
                raise ValueError("X must contain only finite values")

        if n_neighbors > self.n_samples_fit_:
            raise ValueError(
                "Expected n_neighbors <= n_samples_fit, but "
                "n_neighbors = %d, n_samples_fit = %d"
                % (n_neighbors, self.n_samples_fit_)
            )

        if self._fit_method == "kd_tree":
            results = self._tree.query(X, k=n_neighbors, return_distance=return_distance)
        else:
            results = argkmin(
                X, self._fit_X, n_neighbors,
                self.effective_metric_, self.effective_p_,
                return_distance=return_distance,
            )

        if not query_is_train:
            return results

        # Drop the query point itself from its own neighbour list.
        if return_distance:
            neigh_dist, neigh_ind = results
        else:
            neigh_ind = results

        n_queries = X.shape[0]
        sample_range = np.arange(n_queries)[:, None]
        sample_mask = neigh_ind != sample_range

        # Duplicates: if a point is its own neighbour more than once, mask the
        # first duplicate instead (mirrors sklearn).
        dup_gr_nbrs = np.all(sample_mask, axis=1)
        sample_mask[:, 0][dup_gr_nbrs] = False
        neigh_ind = neigh_ind[sample_mask].reshape(n_queries, n_neighbors - 1)

        if return_distance:
            neigh_dist = neigh_dist[sample_mask].reshape(n_queries, n_neighbors - 1)
            return neigh_dist, neigh_ind
        return neigh_ind

    # -- prediction -------------------------------------------------------

    def _check_is_fitted(self):
        if not hasattr(self, "_fit_method"):
            raise ValueError("estimator is not fitted")

    def _brute_predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must be a 2D array with %d features" % self.n_features_in_
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one query sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return argkmin_class_mode(
            X,
            self._fit_X,
            self._y,
            self.n_neighbors,
            self.classes_.size,
            self.effective_metric_,
            self.effective_p_,
            self.weights,
        )

    def predict(self, X):
        self._check_is_fitted()
        if self._fit_method == "brute" and X is not None:
            probabilities = self._brute_predict_proba(X)
            return self.classes_.take(np.argmax(probabilities, axis=1))

        if self.weights == "uniform":
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y.reshape((-1, 1))
        classes_ = [self.classes_]

        n_queries = neigh_ind.shape[0]
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and np.any(np.all(weights == 0, axis=1)):
            raise ValueError("all neighbors of a sample have zero weight")

        y_pred = np.empty((n_queries, 1), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = _mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)
            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        return y_pred.ravel()

    def predict_proba(self, X):
        self._check_is_fitted()
        if self._fit_method == "brute" and X is not None:
            return self._brute_predict_proba(X)

        if self.weights == "uniform":
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        _y = self._y.reshape((-1, 1))
        classes_ = [self.classes_]

        n_queries = neigh_ind.shape[0]
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and np.any(np.all(weights == 0, axis=1)):
            raise ValueError("all neighbors of a sample have zero weight")
        if weights is None:
            weights = np.ones_like(neigh_ind, dtype=np.float64)

        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # Accumulate the weight of each neighbour into its class bin.
            for i, idx in enumerate(pred_labels.T):
                proba_k[all_rows, idx] += weights[:, i]

            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            proba_k /= normalizer
            probabilities.append(proba_k)

        return probabilities[0]

    # -- misc -------------------------------------------------------------

    def score(self, X, y):
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))
