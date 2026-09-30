"""NB 使用的公共流程，依据 sklearn 1.9.1，以 Python/NumPy/SciPy 实现。

仅实现目标模型会调用的路径，不以 sklearn 的校验或 estimator 代替实现。
对应源码：base.py、utils/{validation,multiclass,_param_validation,_tags,
_array_api}.py、preprocessing/{_label,_data}.py。兼容边界见 readme.md。
"""
# Authors: The scikit-learn developers; DLdemo contributors
# SPDX-License-Identifier: BSD-3-Clause

import copy
import functools
import inspect
import warnings
from contextvars import ContextVar
from dataclasses import dataclass, field
from numbers import Integral, Number, Real
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ._config import get_config
from ._metadata import MetadataRequest, UNCHANGED


class NotFittedError(ValueError, AttributeError):
    """预测时模型尚未拟合。"""


class DataConversionWarning(UserWarning):
    """列向量标签被转换为一维。"""


class _NumPyNamespace:
    """对应 sklearn._NumPyAPIWrapper，unique_values 必须保证排序。"""

    __name__ = "numpy"

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def unique_values(array):
        return np.unique(array)


_NUMPY_NAMESPACE = _NumPyNamespace()


class InvalidParameterError(ValueError, TypeError):
    """构造参数不满足目标源码约束。"""


@dataclass
class InputTags:
    one_d_array: bool = False
    two_d_array: bool = True
    three_d_array: bool = False
    sparse: bool = False
    categorical: bool = False
    string: bool = False
    dict: bool = False
    positive_only: bool = False
    allow_nan: bool = False
    pairwise: bool = False


@dataclass
class TargetTags:
    required: bool = True
    one_d_labels: bool = False
    two_d_labels: bool = False
    positive_only: bool = False
    multi_output: bool = False
    single_output: bool = True


@dataclass
class ClassifierTags:
    poor_score: bool = False
    multi_class: bool = True
    multi_label: bool = False


@dataclass
class Tags:
    estimator_type: str = "classifier"
    target_tags: TargetTags = field(default_factory=TargetTags)
    transformer_tags: object = None
    classifier_tags: ClassifierTags = field(default_factory=ClassifierTags)
    regressor_tags: object = None
    array_api_support: bool = False
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: InputTags = field(default_factory=InputTags)


@dataclass
class Interval:
    type: type
    left: object
    right: object
    closed: str

    def accepts(self, value):
        if not isinstance(value, self.type) or np.isnan(value):
            return False
        left = -np.inf if self.left is None else self.left
        right = np.inf if self.right is None else self.right
        return ((value >= left if self.closed in ("left", "both") else value > left)
                and (value <= right if self.closed in ("right", "both") else value < right))

    def __str__(self):
        name = "int" if self.type is Integral else "float"
        left = "-inf" if self.left is None else str(self.left)
        right = "inf" if self.right is None else str(self.right)
        if self.type is Real:
            left = "-inf" if self.left is None else str(float(self.left))
            right = "inf" if self.right is None else str(float(self.right))
        bounds = ("[" if self.closed in ("left", "both") else "(") + left + ", " + right
        bounds += "]" if self.closed in ("right", "both") else ")"
        article = "an" if self.type is Integral else "a"
        return f"{article} {name} in the range {bounds}"


def _constraint_accepts(constraint, value):
    if constraint is None:
        return value is None
    if isinstance(constraint, Interval):
        return constraint.accepts(value)
    if constraint == "array-like":
        return (not sparse.issparse(value)
                and (hasattr(value, "__len__") or hasattr(value, "shape")
                     or hasattr(value, "__array__")) and not isinstance(value, str))
    if constraint == "boolean":
        return isinstance(value, (bool, np.bool_))
    return False


class BaseEstimator:
    """目标模型所需的参数管理、克隆、文本表示与标签。"""

    @classmethod
    def _get_param_names(cls):
        return sorted(p.name for p in inspect.signature(cls.__init__).parameters.values()
                      if p.name != "self" and p.kind != p.VAR_KEYWORD)

    def get_params(self, deep=True):
        params = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                params.update((f"{name}__{k}", v) for k, v in value.get_params().items())
            params[name] = value
        return params

    def set_params(self, **params):
        valid = self.get_params(deep=True)
        nested = {}
        for key, value in params.items():
            name, sep, child = key.partition("__")
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for estimator {self}. "
                                 f"Valid parameters are: {self._get_param_names()!r}.")
            if sep:
                nested.setdefault(name, {})[child] = value
            else:
                setattr(self, name, value)
                valid[name] = value
        for name, values in nested.items():
            valid[name].set_params(**values)
        return self

    def __sklearn_clone__(self):
        cloned = type(self)(**copy.deepcopy(self.get_params(deep=False)))
        if hasattr(self, "_metadata_request"):
            cloned._metadata_request = self._metadata_request.__sklearn_clone__()
        return cloned

    def get_metadata_routing(self):
        if hasattr(self, "_metadata_request"):
            return self._metadata_request.__sklearn_clone__()
        request = MetadataRequest(owner=self)
        request.fit.add_request(param="sample_weight", alias=None)
        request.partial_fit.add_request(param="classes", alias=None)
        request.partial_fit.add_request(param="sample_weight", alias=None)
        request.score.add_request(param="sample_weight", alias=None)
        return request

    def _set_request(self, method, values):
        if not get_config()["enable_metadata_routing"]:
            raise RuntimeError("This method is only available when metadata routing is enabled. "
                               "You can enable it using NB.sklearn_impl.set_config(enable_metadata_routing=True).")
        request = self.get_metadata_routing()
        for param, alias in values.items():
            if alias is not UNCHANGED:
                getattr(request, method).add_request(param=param, alias=alias)
        self._metadata_request = request
        return self

    def set_fit_request(self, *, sample_weight=UNCHANGED):
        return self._set_request("fit", {"sample_weight": sample_weight})

    def set_partial_fit_request(self, *, classes=UNCHANGED, sample_weight=UNCHANGED):
        return self._set_request("partial_fit", {"classes": classes, "sample_weight": sample_weight})

    def set_score_request(self, *, sample_weight=UNCHANGED):
        return self._set_request("score", {"sample_weight": sample_weight})

    def __repr__(self):
        defaults = inspect.signature(type(self).__init__).parameters
        changed = []
        for name, value in self.get_params(deep=False).items():
            if not get_config()["print_changed_only"] or repr(value) != repr(defaults[name].default):
                changed.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(changed)})"

    def __sklearn_tags__(self):
        return Tags()

    def _validate_params(self):
        for name, constraints in self._parameter_constraints.items():
            value = getattr(self, name)
            if any(_constraint_accepts(c, value) for c in constraints):
                continue
            descriptions = ["None" if c is None else "an array-like" if c == "array-like"
                            else "an instance of 'bool' or an instance of 'numpy.bool'" if c == "boolean" else str(c)
                            for c in constraints]
            expected = descriptions[0] if len(descriptions) == 1 else ", ".join(descriptions[:-1]) + " or " + descriptions[-1]
            raise InvalidParameterError(f"The {name!r} parameter of {type(self).__name__} "
                                        f"must be {expected}. Got {value!r} instead.")


class ClassifierMixin:
    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        y = _check_y(y)
        if y.shape[0] != prediction.shape[0]:
            raise ValueError("Found input variables with inconsistent numbers of samples: "
                             f"[{y.shape[0]}, {prediction.shape[0]}]")
        xp, _ = get_namespace(prediction, y)
        correct = prediction == y
        if sample_weight is None:
            return float(xp.mean(xp.astype(correct, xp.float64)))
        xp, _, device = get_namespace_and_device(correct)
        weights = xp.asarray(sample_weight, device=device)
        return float(_average(xp.astype(correct, xp.float64), axis=0, weights=weights, xp=xp))


_skip_parameter_validation = ContextVar("nb_skip_parameter_validation", default=False)


def _fit_context(*, prefer_skip_nested_validation):
    def decorate(method):
        @functools.wraps(method)
        def wrapped(estimator, *args, **kwargs):
            already_fitted = method.__name__ == "partial_fit" and _is_fitted(estimator)
            if not get_config()["skip_parameter_validation"] and not _skip_parameter_validation.get() and not already_fitted:
                estimator._validate_params()
            token = _skip_parameter_validation.set(prefer_skip_nested_validation)
            try:
                return method(estimator, *args, **kwargs)
            finally:
                _skip_parameter_validation.reset(token)
        return wrapped
    return decorate


def _is_fitted(estimator):
    return any(name.endswith("_") and not name.startswith("__") for name in vars(estimator))


def check_is_fitted(estimator):
    if not _is_fitted(estimator):
        raise NotFittedError(f"This {type(estimator).__name__} instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this estimator.")


def get_namespace(*arrays):
    # NumPy 的默认路径；外部 Array API namespace 直接采用其协议。
    if not get_config()["array_api_dispatch"]:
        return _NUMPY_NAMESPACE, False
    for array in arrays:
        if not isinstance(array, (np.ndarray, np.generic)) and hasattr(array, "__array_namespace__"):
            return array.__array_namespace__(), True
    return _NUMPY_NAMESPACE, False


def get_namespace_and_device(*arrays):
    xp, is_array_api = get_namespace(*arrays)
    device = next((getattr(a, "device", None) for a in arrays if hasattr(a, "shape")), None)
    return xp, is_array_api, device


def move_to(array, *, xp, device):
    if xp in (np, _NUMPY_NAMESPACE):
        try:
            return np.from_dlpack(array, device="cpu")
        except (AttributeError, TypeError, NotImplementedError, BufferError, ValueError):
            pass
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        else:
            source_xp, _ = get_namespace(array)
            if source_xp.__name__ == "array_api_strict":
                array = source_xp.asarray(array, device=source_xp.Device("CPU_DEVICE"))
        return np.asarray(array)
    return xp.asarray(array, device=device)


def _find_matching_floating_dtype(*arrays, xp):
    dtypes = [a.dtype for a in arrays if hasattr(a, "dtype") and xp.isdtype(a.dtype, "real floating")]
    if dtypes:
        return xp.result_type(*dtypes)
    return xp.asarray(0.0).dtype


def _average(X, axis, weights, xp):
    if xp in (np, _NUMPY_NAMESPACE):
        return np.average(X, axis=axis, weights=weights)
    shape = [1] * X.ndim
    shape[axis] = weights.shape[0]
    dtype = _find_matching_floating_dtype(X, weights, xp=xp)
    X = xp.astype(X, dtype, copy=False)
    weights = xp.astype(weights, dtype, copy=False)
    scale = xp.sum(weights)
    if bool(scale == 0):
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    return xp.sum(X * xp.reshape(weights, tuple(shape)), axis=axis) / scale


def _isin(element, test_elements, xp):
    if xp in (np, _NUMPY_NAMESPACE):
        return np.isin(element, test_elements)
    return xp.any(element[:, None] == test_elements[None, :], axis=1)


def _logsumexp(array, axis, xp):
    # utils._array_api._logsumexp：单独统计最大项，再用 log1p。
    # 保留其全 -inf/+inf 以及 NaN 边界，不能换成另一版本的归一化规则。
    supported = (xp.float64, xp.float32) + ((xp.float16,) if hasattr(xp, "float16") else ())
    if array.dtype not in supported:
        array = xp.asarray(array, dtype=xp.float64)
    maximum = xp.max(array, axis=axis, keepdims=True)
    index_max = array == maximum
    array = xp.asarray(array, copy=True)
    array[index_max] = -xp.inf
    m = xp.sum(xp.astype(index_max, array.dtype), axis=axis, keepdims=True,
               dtype=array.dtype)
    shift = xp.where(xp.isfinite(maximum), maximum, 0)
    exp = xp.exp(array - shift)
    s = xp.sum(exp, axis=axis, keepdims=True, dtype=exp.dtype)
    s = xp.where(s == 0, s, s / m)
    return xp.squeeze(xp.log1p(s) + xp.log(m) + maximum, axis=axis)


def _atleast_nd(array, *, ndim):
    xp, _ = get_namespace(array)
    return xp.reshape(array, (1,) * max(0, ndim - array.ndim) + array.shape)


def _isclose(a, b):
    xp, _ = get_namespace(a)
    return xp.abs(a - b) <= 1e-8 + 1e-5 * abs(b)


xpx = SimpleNamespace(atleast_nd=_atleast_nd, isclose=_isclose)


def size(array):
    return int(np.prod(array.shape))


def _ensure_finite(array, name):
    if get_config()["assume_finite"]:
        return
    xp, _ = get_namespace(array)
    if xp in (np, _NUMPY_NAMESPACE) and array.dtype.kind not in "fcO":
        return
    if xp in (np, _NUMPY_NAMESPACE) and array.dtype.kind == "O":
        if any(isinstance(x, Real) and np.isnan(x) for x in array.ravel()):
            raise ValueError("Input contains NaN")
        return
    # validation._assert_all_finite 的常见路径：O(n) 扫描、O(1) 临时空间。
    with np.errstate(over="ignore"):
        if bool(xp.isfinite(xp.sum(array))):
            return
    if bool(xp.any(xp.isnan(array))):
        raise ValueError(f"Input {name} contains NaN.")
    if not bool(xp.all(xp.isfinite(array))):
        raise ValueError(f"Input {name} contains infinity or a value too large for "
                         f"{array.dtype!r}.")


def _check_y(y):
    if y is None:
        raise ValueError("This estimator requires y to be passed, but the target y is None")
    if sparse.issparse(y):
        raise TypeError("Sparse data was passed for y, but dense data is required.")
    xp, _, device = get_namespace_and_device(y)
    y = xp.asarray(y, device=device)
    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn("A column-vector y was passed when a 1d array was expected. "
                      "Please change the shape of y to (n_samples, ), for example using ravel().",
                      DataConversionWarning)
        y = xp.reshape(y, (-1,))
    elif y.ndim != 1:
        raise ValueError(f"y should be a 1d array, got an array of shape {y.shape} instead.")
    if xp.isdtype(y.dtype, "complex floating"):
        raise ValueError(f"Complex data not supported\n{y}\n")
    _ensure_finite(y, "y")
    return y


def _unique_labels(y):
    y = _check_y(y)
    xp, _ = get_namespace(y)
    if xp.isdtype(y.dtype, "real floating") and bool(xp.any(y != xp.floor(y))):
        raise ValueError(f"Unknown label type: {y!r}")
    if xp in (np, _NUMPY_NAMESPACE) and y.dtype.kind == "O":
        if len(y) and not isinstance(y[0], str):
            raise ValueError(f"Unknown label type: {y!r}")
        if not all(isinstance(v, str) for v in y):
            raise ValueError("Mix of label input types (string and number)")
    return xp.unique_values(y)


def _check_partial_fit_first_call(estimator, classes=None):
    if getattr(estimator, "classes_", None) is None and classes is None:
        raise ValueError("classes must be passed on the first call to partial_fit.")
    if classes is not None:
        labels = _unique_labels(classes)
        if getattr(estimator, "classes_", None) is None:
            estimator.classes_ = labels
            return True
        if not np.array_equal(move_to(estimator.classes_, xp=np, device="cpu"),
                              move_to(labels, xp=np, device="cpu")):
            raise ValueError("`classes=%r` is not the same as on last call "
                             "to partial_fit, was: %r" % (classes, estimator.classes_))
    return False


def label_binarize(y, *, classes):
    _unique_labels(y)  # LabelBinarizer 的分类目标检查。
    y = np.asarray(y)
    classes = np.asarray(classes)
    dtype = y.dtype if np.issubdtype(y.dtype, np.signedinteger) else np.int64
    if len(classes) == 1:
        return np.zeros((len(y), 1), dtype=dtype)
    if np.issubdtype(y.dtype, np.integer):
        classes = classes.astype(y.dtype, copy=False)
    sorted_classes = np.sort(classes)
    known = np.isin(y, classes)
    indices = np.searchsorted(sorted_classes, y[known])
    indptr = np.concatenate(([0], np.cumsum(known, dtype=dtype)))
    # preprocessing._label.label_binarize：稀疏指示矩阵，再转为稠密 Y。
    Y = sparse.csr_array((np.ones_like(indices), indices, indptr),
                         shape=(y.shape[0], classes.shape[0])).toarray()
    Y = Y.astype(dtype, copy=False)
    return Y[:, -1:] if len(classes) == 2 else Y


class LabelBinarizer:
    def fit_transform(self, y):
        self.classes_ = _unique_labels(y)
        if len(y) == 0:
            raise ValueError("y has 0 samples")
        return label_binarize(y, classes=self.classes_)


def _feature_names(X):
    if hasattr(X, "columns"):
        names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        names = np.asarray(list(X.__dataframe__().column_names()), dtype=object)
    else:
        return None
    if not len(names):
        return None
    types = sorted({type(v).__qualname__ for v in names})
    if len(types) > 1 and "str" in types:
        raise TypeError("Feature names are only supported if all input features have string "
                        f"names, but your input has {types} as feature name / column name types.")
    return names if types == ["str"] else None


def _check_feature_names(estimator, X, *, reset):
    names = _feature_names(X)
    fitted = getattr(estimator, "feature_names_in_", None)
    if reset:
        if names is not None:
            estimator.feature_names_in_ = names
        elif fitted is not None:
            del estimator.feature_names_in_
        return
    if names is None and fitted is None:
        return
    if fitted is None:
        warnings.warn(f"X has feature names, but {type(estimator).__name__} was fitted "
                      "without feature names")
        return
    if names is None:
        warnings.warn(f"X does not have valid feature names, but {type(estimator).__name__} "
                      "was fitted with feature names")
        return
    if not np.array_equal(names, fitted):
        message = "The feature names should match those that were passed during fit.\n"
        unexpected = sorted(set(names) - set(fitted))
        missing = sorted(set(fitted) - set(names))
        for label, values in [("Feature names unseen at fit time:", unexpected),
                              ("Feature names seen at fit time, yet now missing:", missing)]:
            if values:
                message += label + "\n" + "".join(f"- {v}\n" for v in values[:5])
                if len(values) > 5:
                    message += "- ...\n"
        if not unexpected and not missing:
            message += "Feature names must be in the same order as they were in fit.\n"
        raise ValueError(message)


def _check_n_features(estimator, X, *, reset):
    if reset:
        estimator.n_features_in_ = X.shape[1]
    elif hasattr(estimator, "n_features_in_") and X.shape[1] != estimator.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features, but {type(estimator).__name__} "
                         f"is expecting {estimator.n_features_in_} features as input.")


_NO_VALIDATION = "no_validation"


def validate_data(estimator, X=_NO_VALIDATION, y=_NO_VALIDATION, *, reset=True,
                  dtype="numeric", accept_sparse=False, ensure_all_finite=True):
    no_X = isinstance(X, str) and X == _NO_VALIDATION
    no_y = isinstance(y, str) and y == _NO_VALIDATION
    if no_X:
        return _check_y(y)
    _check_feature_names(estimator, X, reset=reset)
    if not no_y and y is None:
        raise ValueError(f"This {type(estimator).__name__} estimator requires y to be passed, "
                         "but the target y is None.")
    if hasattr(X, "dtypes"):
        dtypes = list(X.dtypes)
        if dtypes and all(type(t).__name__ == "SparseDtype" for t in dtypes):
            X = X.sparse.to_coo()
        elif any(type(t).__name__ in {"Float64Dtype", "Float32Dtype", "Int64Dtype", "Int32Dtype",
                                     "Int16Dtype", "Int8Dtype", "UInt64Dtype", "UInt32Dtype",
                                     "UInt16Dtype", "UInt8Dtype", "BooleanDtype"} for t in dtypes):
            X = X.astype(np.float64)
    if sparse.issparse(X):
        if not accept_sparse:
            raise TypeError("Sparse data was passed for X, but dense data is required. "
                            "Use '.toarray()' to convert to a dense numpy array.")
        if np.issubdtype(X.dtype, np.complexfloating):
            raise ValueError(f"Complex data not supported\n{X}\n")
        X = X.tocsr(copy=False)
        if X.ndim < 2:
            raise ValueError(f"Expected 2D input, got input with shape {X.shape}.\n"
                             "Reshape your data either using array.reshape(-1, 1) if "
                             "your data has a single feature or array.reshape(1, -1) "
                             "if it contains a single sample.")
        if X.dtype.kind == "O":
            X = X.astype(np.float64)
        _ensure_finite(X.data, "X")
    else:
        xp, _, device = get_namespace_and_device(X)
        if xp in (np, _NUMPY_NAMESPACE) and hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = xp.asarray(X, device=device)
        if xp.isdtype(X.dtype, "complex floating"):
            raise ValueError(f"Complex data not supported\n{X}\n")
        if xp in (np, _NUMPY_NAMESPACE):
            if dtype == "numeric" and X.dtype.kind == "O":
                X = X.astype(np.float64)
            elif dtype == "numeric" and X.dtype.kind in "USV":
                raise ValueError("dtype='numeric' is not compatible with arrays of "
                                 "bytes/strings. Convert your data to numeric values explicitly instead.")
            elif dtype == "int":
                _ensure_finite(X, "X")
                X = X.astype(int, copy=False)
        if X.ndim != 2:
            if X.ndim > 2:
                raise ValueError(f"Found array with dim {X.ndim}, while dim <= 2 is required "
                                 f"by {type(estimator).__name__}.")
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead:\narray={X}.\n"
                             "Reshape your data either using array.reshape(-1, 1) if your "
                             "data has a single feature or array.reshape(1, -1) if it "
                             "contains a single sample.")
        _ensure_finite(X, "X")
    if X.shape[0] < 1:
        raise ValueError(f"Found array with 0 sample(s) (shape={X.shape}) while a minimum "
                         f"of 1 is required by {type(estimator).__name__}.")
    if X.shape[1] < 1:
        raise ValueError(f"Found array with 0 feature(s) (shape={X.shape}) while a minimum "
                         f"of 1 is required by {type(estimator).__name__}.")
    if not no_y:
        y = _check_y(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Found input variables with inconsistent numbers of samples: "
                             f"[{X.shape[0]}, {y.shape[0]}]")
    _check_n_features(estimator, X, reset=reset)
    return X if no_y else (X, y)


def _check_sample_weight(sample_weight, X, dtype=None):
    xp, _, device = get_namespace_and_device(X)
    if dtype is not None and dtype not in (xp.float64, xp.float32):
        dtype = xp.float64
    if dtype is None and hasattr(sample_weight, "dtype") and sample_weight.dtype == xp.float32:
        dtype = xp.float32
    if dtype is None:
        dtype = xp.float64
    if isinstance(sample_weight, Number):
        weights = xp.full(X.shape[0], sample_weight, dtype=dtype, device=device)
    else:
        if sparse.issparse(sample_weight):
            raise TypeError("Sparse data was passed for sample_weight, but dense data is required.")
        weights = xp.asarray(sample_weight, dtype=dtype, device=device)
        _ensure_finite(weights, "sample_weight")
        if weights.ndim != 1:
            raise ValueError(f"Sample weights must be 1D array or scalar, got {weights.ndim}D "
                             "array. Expected either a scalar value or a 1D array of length "
                             f"{X.shape[0]}.")
        if weights.shape != (X.shape[0],):
            raise ValueError(f"sample_weight.shape == {weights.shape}, expected {(X.shape[0],)}!")
        if xp in (np, _NUMPY_NAMESPACE):
            weights = np.asarray(weights, order="C")
    if bool(xp.all(weights == 0)):
        raise ValueError("Sample weights must contain at least one non-zero number.")
    return weights


def check_non_negative(X, whom):
    # validation.check_non_negative：只检查存储值，不调用会排序索引的稀疏 min。
    data = X.data if sparse.issparse(X) else X
    if data.size and np.min(data) < 0:
        raise ValueError(f"Negative values in data passed to {whom}.")


def binarize(X, *, threshold):
    # _data.binarize 默认 copy=True；重复 CSR 索引逐存储值二值化，不先 sum_duplicates。
    if sparse.issparse(X):
        if threshold < 0:
            raise ValueError("Cannot binarize a sparse matrix with threshold < 0")
        X = X.copy()
        cond = X.data > threshold
        not_cond = np.logical_not(cond)
        X.data[cond] = 1
        X.data[not_cond] = 0
        X.eliminate_zeros()
        return X
    X = X.copy()
    float_dtype = _find_matching_floating_dtype(X, threshold, xp=_NUMPY_NAMESPACE)
    cond = X.astype(float_dtype, copy=False) > threshold
    not_cond = np.logical_not(cond)
    X[cond] = 1
    X[not_cond] = 0
    return X
