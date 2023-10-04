from __future__ import annotations


class ExprDispatch:
    """Expression-level dispatching on Expr._meta"""

    def __init__(self, default):
        self._default = default
        self._lookup = {}

    def __reduce__(self):
        return type(self), (self._default,)

    def register(self, typ, func=None):
        def wrapper(func):
            if isinstance(typ, tuple):
                for t in typ:
                    self.register(t, func)
            else:
                self._lookup[typ] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def dispatch(self, cls):
        lk = self._lookup
        for cls2 in cls.__mro__:
            try:
                impl = lk[cls2]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    # Cache lookup
                    lk[cls] = impl
                return impl
        return self._default

    def __call__(self, *args, **kwargs):
        from dask_expr._expr import Expr

        use_cls = self._default
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, Expr):
                use_cls = self.dispatch(type(arg._meta))
                break
        return use_cls(*args, **kwargs)


### Define "dispatch" classes

from dask_expr._groupby import GroupbyAggregation

GroupbyAggregationDispatch = ExprDispatch(GroupbyAggregation)
