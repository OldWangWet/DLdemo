"""NB 简单消费者的元数据请求，改编自 sklearn 1.9.1 _metadata_requests.py。

不实现 sklearn 整个路由器框架；此对象可被其路由器按消费者协议读取。
"""
# Authors: The scikit-learn developers; DLdemo contributors
# SPDX-License-Identifier: BSD-3-Clause
from copy import deepcopy
from warnings import warn

class UnsetMetadataPassedError(ValueError):
    def __init__(self, *, message, unrequested_params, routed_params):
        super().__init__(message)
        self.unrequested_params = unrequested_params
        self.routed_params = routed_params

class Bunch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

def _routing_repr(owner):
    return owner if isinstance(owner, str) else type(owner).__name__

SIMPLE_METHODS = ['fit', 'partial_fit', 'predict', 'predict_proba', 'predict_log_proba', 'decision_function', 'score', 'split', 'transform', 'inverse_transform']

COMPOSITE_METHODS = {'fit_transform': ['fit', 'transform'], 'fit_predict': ['fit', 'predict']}

METHODS = SIMPLE_METHODS + list(COMPOSITE_METHODS.keys())

UNUSED = '$UNUSED$'

WARN = '$WARN$'

UNCHANGED = '$UNCHANGED$'

VALID_REQUEST_VALUES = [False, True, None, UNUSED, WARN]

def request_is_alias(item):
    if item in VALID_REQUEST_VALUES:
        return False
    return isinstance(item, str) and item.isidentifier()

def request_is_valid(item):
    return item in VALID_REQUEST_VALUES

class MethodMetadataRequest:

    def __init__(self, owner, method, requests=None):
        self._requests = requests or dict()
        self.owner = owner
        self.method = method

    def __sklearn_clone__(self):
        return MethodMetadataRequest(owner=self.owner, method=self.method, requests=deepcopy(self._requests))

    @property
    def requests(self):
        return self._requests

    def add_request(self, *, param, alias):
        if not request_is_alias(alias) and (not request_is_valid(alias)):
            raise ValueError(f"The alias you're setting for `{param}` should be either a valid identifier or one of {{None, True, False}}, but given value is: `{alias}`")
        if alias == param:
            alias = True
        if alias == UNUSED:
            if param in self._requests:
                del self._requests[param]
            else:
                raise ValueError(f"Trying to remove parameter {param} with UNUSED which doesn't exist.")
        else:
            self._requests[param] = alias
        return self

    def _get_param_names(self, return_alias):
        return set((alias if return_alias and (not request_is_valid(alias)) else prop for prop, alias in self._requests.items() if not request_is_valid(alias) or alias is not False))

    def _check_warnings(self, *, params):
        params = {} if params is None else params
        warn_params = {prop for prop, alias in self._requests.items() if alias == WARN and prop in params}
        for param in warn_params:
            warn(f'Support for {param} has recently been added to {self.owner} class. To maintain backward compatibility, it is ignored now. Using `set_{self.method}_request({param}={{True, False}})` on this method of the class, you can set the request value to False to silence this warning, or to True to consume and use the metadata.')

    def _route_params(self, params, parent, caller):
        self._check_warnings(params=params)
        unrequested = dict()
        args = {arg: value for arg, value in params.items() if value is not None}
        res = Bunch()
        for prop, alias in self._requests.items():
            if alias is False or alias == WARN:
                continue
            elif alias is True and prop in args:
                res[prop] = args[prop]
            elif alias is None and prop in args:
                unrequested[prop] = args[prop]
            elif alias in args:
                res[prop] = args[alias]
        if unrequested:
            if self.method in COMPOSITE_METHODS:
                callee_methods = COMPOSITE_METHODS[self.method]
            else:
                callee_methods = [self.method]
            set_requests_on = ''.join([f'.set_{method}_request({{metadata}}=True/False)' for method in callee_methods])
            message = f"[{', '.join([key for key in unrequested])}] are passed but are not explicitly set as requested or not requested for {_routing_repr(self.owner)}.{self.method}, which is used within {_routing_repr(parent)}.{caller}. Call `{_routing_repr(self.owner)}" + set_requests_on + '` for each metadata you want to request/ignore. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.'
            raise UnsetMetadataPassedError(message=message, unrequested_params=unrequested, routed_params=res)
        return res

    def _consumes(self, params):
        params = set(params)
        consumed_params = set()
        for metadata_name, alias in self._requests.items():
            if alias is True and metadata_name in params:
                consumed_params.add(metadata_name)
            elif isinstance(alias, str) and alias in params:
                consumed_params.add(alias)
        return consumed_params

    def _serialize(self):
        return self._requests

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))

class MetadataRequest:
    _type = 'metadata_request'

    def __init__(self, owner):
        self.owner = owner
        for method in SIMPLE_METHODS:
            setattr(self, method, MethodMetadataRequest(owner=owner, method=method))

    def __sklearn_clone__(self):
        new = MetadataRequest(owner=self.owner)
        for method in SIMPLE_METHODS:
            setattr(new, method, getattr(self, method).__sklearn_clone__())
        return new

    def consumes(self, method, params):
        return getattr(self, method)._consumes(params=params)

    def __getattr__(self, name):
        if name not in COMPOSITE_METHODS:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        requests = {}
        for method in COMPOSITE_METHODS[name]:
            mmr = getattr(self, method)
            existing = set(requests.keys())
            upcoming = set(mmr.requests.keys())
            common = existing & upcoming
            conflicts = [key for key in common if requests[key] != mmr._requests[key]]
            if conflicts:
                raise ValueError(f"Conflicting metadata requests for {', '.join(conflicts)} while composing the requests for {name}. Metadata with the same name for methods {', '.join(COMPOSITE_METHODS[name])} should have the same request value.")
            requests.update(mmr._requests)
        return MethodMetadataRequest(owner=self.owner, method=name, requests=requests)

    def _get_param_names(self, method, return_alias, ignore_self_request=None):
        return getattr(self, method)._get_param_names(return_alias=return_alias)

    def _route_params(self, *, params, method, parent, caller):
        return getattr(self, method)._route_params(params=params, parent=parent, caller=caller)

    def _check_warnings(self, *, method, params):
        getattr(self, method)._check_warnings(params=params)

    def _serialize(self):
        output = dict()
        for method in SIMPLE_METHODS:
            mmr = getattr(self, method)
            if len(mmr.requests):
                output[method] = mmr._serialize()
        return output

    def __repr__(self):
        return str(self._serialize())

    def __str__(self):
        return str(repr(self))
