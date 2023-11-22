import functools

import numpy as np
from dask import is_dask_collection
from dask.dataframe.core import _concat, is_dataframe_like, is_series_like
from dask.dataframe.dispatch import make_meta, meta_nonempty
from dask.dataframe.groupby import (
    _agg_finalize,
    _apply_chunk,
    _build_agg_args,
    _determine_levels,
    _groupby_aggregate,
    _groupby_apply_funcs,
    _groupby_slice_apply,
    _groupby_slice_shift,
    _groupby_slice_transform,
    _normalize_spec,
    _value_counts,
    _value_counts_aggregate,
    _var_agg,
    _var_chunk,
    _var_combine,
)
from dask.utils import M, is_index_like

from dask_expr._collection import DataFrame, Index, Series, new_collection
from dask_expr._expr import (
    Assign,
    Blockwise,
    Expr,
    MapPartitions,
    Projection,
    no_default,
)
from dask_expr._reductions import ApplyConcatApply, Chunk, Reduction
from dask_expr._shuffle import Shuffle
from dask_expr._util import is_scalar


def _as_dict(key, value):
    # Utility to convert a single kwarg to a dict.
    # The dict will be empty if the value is None
    return {} if value is None else {key: value}


###
### Groupby-aggregation expressions
###


class GroupByChunk(Chunk):
    _parameters = Chunk._parameters + ["by"]
    _defaults = Chunk._defaults | {"by": None}

    @functools.cached_property
    def _args(self) -> list:
        return [self.frame, self.by]


class GroupByApplyConcatApply(ApplyConcatApply):
    _chunk_cls = GroupByChunk

    @functools.cached_property
    def _meta_chunk(self):
        meta = meta_nonempty(self.frame._meta)
        by = self.by if not isinstance(self.by, Expr) else meta_nonempty(self.by._meta)
        return self.chunk(meta, by, **self.chunk_kwargs)

    @property
    def _chunk_cls_args(self):
        return [self.by]


class SingleAggregation(GroupByApplyConcatApply):
    """Single groupby aggregation

    This is an abstract class. Sub-classes must implement
    the following methods:

    -   `groupby_chunk`: Applied to each group within
        the `chunk` method of `GroupByApplyConcatApply`
    -   `groupby_aggregate`: Applied to each group within
        the `aggregate` method of `GroupByApplyConcatApply`

    Parameters
    ----------
    frame: Expr
        Dataframe- or series-like expression to group.
    by: str, list or Series
        The key for grouping
    observed:
        Passed through to dataframe backend.
    dropna:
        Whether rows with NA values should be dropped.
    chunk_kwargs:
        Key-word arguments to pass to `groupby_chunk`.
    aggregate_kwargs:
        Key-word arguments to pass to `aggregate_chunk`.
    """

    _parameters = [
        "frame",
        "by",
        "observed",
        "dropna",
        "chunk_kwargs",
        "aggregate_kwargs",
        "_slice",
        "split_every",
        "split_out",
        "sort",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
        "chunk_kwargs": None,
        "aggregate_kwargs": None,
        "_slice": None,
        "split_every": 8,
        "split_out": 1,
        "sort": None,
    }

    groupby_chunk = None
    groupby_aggregate = None

    @property
    def split_by(self):
        return self.by

    @classmethod
    def chunk(cls, df, by=None, **kwargs):
        if hasattr(by, "dtype"):
            by = [by]
        return _apply_chunk(df, *by, **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        return _groupby_aggregate(_concat(inputs), **kwargs)

    @property
    def chunk_kwargs(self) -> dict:
        chunk_kwargs = self.operand("chunk_kwargs") or {}
        columns = self._slice
        return {
            "chunk": self.groupby_chunk,
            "columns": columns,
            **_as_dict("observed", self.observed),
            **_as_dict("dropna", self.dropna),
            **chunk_kwargs,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        aggregate_kwargs = self.operand("aggregate_kwargs") or {}
        groupby_aggregate = self.groupby_aggregate or self.groupby_chunk
        return {
            "aggfunc": groupby_aggregate,
            "levels": _determine_levels(self.by),
            "sort": self.sort,
            **_as_dict("observed", self.observed),
            **_as_dict("dropna", self.dropna),
            **aggregate_kwargs,
        }

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            by_columns = self.by if not isinstance(self.by, Expr) else []
            columns = sorted(set(parent.columns + by_columns))
            if columns == self.frame.columns:
                return
            return type(parent)(
                type(self)(self.frame[columns], *self.operands[1:]),
                *parent.operands[1:],
            )


class GroupbyAggregation(GroupByApplyConcatApply):
    """General groupby aggregation

    This class can be used directly to perform a general
    groupby aggregation by passing in a `str`, `list` or
    `dict`-based specification using the `arg` operand.

    Parameters
    ----------
    frame: Expr
        Dataframe- or series-like expression to group.
    by: str, list or Series
        The key for grouping
    arg: str, list or dict
        Aggregation spec defining the specific aggregations
        to perform.
    observed:
        Passed through to dataframe backend.
    dropna:
        Whether rows with NA values should be dropped.
    chunk_kwargs:
        Key-word arguments to pass to `groupby_chunk`.
    aggregate_kwargs:
        Key-word arguments to pass to `aggregate_chunk`.
    """

    _parameters = [
        "frame",
        "by",
        "arg",
        "observed",
        "dropna",
        "split_every",
        "split_out",
        "sort",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
        "split_every": 8,
        "split_out": 1,
        "sort": None,
    }

    def __exec__(self):
        frame = self.frame.__exec__()
        kwargs = {
            "sort": self.sort,
            **_as_dict("observed", self.observed),
            **_as_dict("dropna", self.dropna),
        }
        return frame.groupby(self.by, **kwargs).aggregate(self.arg)

    @property
    def split_by(self):
        return self.by

    @functools.cached_property
    def spec(self):
        # Converts the `arg` operand into specific
        # chunk, aggregate, and finalizer functions
        if isinstance(self.by, Expr):
            group_columns = []
        else:
            group_columns = set(self.by)
        non_group_columns = [
            col for col in self.frame.columns if col not in group_columns
        ]
        spec = _normalize_spec(self.arg, non_group_columns)

        # Median not supported yet
        has_median = any(s[1] in ("median", np.median) for s in spec)
        if has_median:
            raise NotImplementedError("median not yet supported")

        keys = ["chunk_funcs", "aggregate_funcs", "finalizers"]
        return dict(zip(keys, _build_agg_args(spec)))

    @classmethod
    def chunk(cls, df, by=None, **kwargs):
        if hasattr(by, "dtype"):
            by = [by]
        return _groupby_apply_funcs(df, *by, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        return _groupby_apply_funcs(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        return _agg_finalize(_concat(inputs), **kwargs)

    @property
    def chunk_kwargs(self) -> dict:
        return {
            "funcs": self.spec["chunk_funcs"],
            "sort": self.sort,
            **_as_dict("observed", self.observed),
            **_as_dict("dropna", self.dropna),
        }

    @property
    def combine_kwargs(self) -> dict:
        return {
            "funcs": self.spec["aggregate_funcs"],
            "level": _determine_levels(self.by),
            "sort": self.sort,
            **_as_dict("observed", self.observed),
            **_as_dict("dropna", self.dropna),
        }

    @property
    def aggregate_kwargs(self) -> dict:
        return {
            "aggregate_funcs": self.spec["aggregate_funcs"],
            "finalize_funcs": self.spec["finalizers"],
            "level": _determine_levels(self.by),
            "sort": self.sort,
            **_as_dict("observed", self.observed),
            **_as_dict("dropna", self.dropna),
        }

    def _simplify_down(self):
        # Use agg-spec information to add column projection
        column_projection = None
        by_columns = self.by if not isinstance(self.by, Expr) else []
        if isinstance(self.arg, dict):
            column_projection = (
                set(by_columns).union(self.arg.keys()).intersection(self.frame.columns)
            )
        if column_projection and column_projection < set(self.frame.columns):
            return type(self)(self.frame[list(column_projection)], *self.operands[1:])


class Sum(SingleAggregation):
    groupby_chunk = M.sum


class Prod(SingleAggregation):
    groupby_chunk = M.prod


class Min(SingleAggregation):
    groupby_chunk = M.min


class Max(SingleAggregation):
    groupby_chunk = M.max


class First(SingleAggregation):
    groupby_chunk = M.first


class Last(SingleAggregation):
    groupby_chunk = M.last


class Count(SingleAggregation):
    groupby_chunk = M.count
    groupby_aggregate = M.sum


class Size(SingleAggregation):
    groupby_chunk = M.size
    groupby_aggregate = M.sum


class ValueCounts(SingleAggregation):
    groupby_chunk = staticmethod(_value_counts)
    groupby_aggregate = staticmethod(_value_counts_aggregate)


class GroupByReduction(Reduction):
    _chunk_cls = GroupByChunk

    @property
    def _chunk_cls_args(self):
        return [self.by]

    @functools.cached_property
    def _meta_chunk(self):
        meta = meta_nonempty(self.frame._meta)
        by = self.by if not isinstance(self.by, Expr) else meta_nonempty(self.by._meta)
        return self.chunk(meta, by, **self.chunk_kwargs)


class Var(GroupByReduction):
    _parameters = ["frame", "by", "ddof", "numeric_only", "split_out", "sort"]
    _defaults = {"split_out": 1, "sort": None}
    reduction_aggregate = _var_agg
    reduction_combine = _var_combine

    @property
    def split_by(self):
        return self.by

    @staticmethod
    def chunk(frame, by, **kwargs):
        if hasattr(by, "dtype"):
            by = [by]
        return _var_chunk(frame, *by, **kwargs)

    @functools.cached_property
    def levels(self):
        return _determine_levels(self.by)

    @functools.cached_property
    def aggregate_kwargs(self):
        return {
            "ddof": self.ddof,
            "levels": self.levels,
            "numeric_only": self.numeric_only,
            "sort": self.sort,
        }

    @functools.cached_property
    def chunk_kwargs(self):
        return {"numeric_only": self.numeric_only}

    @functools.cached_property
    def combine_kwargs(self):
        return {"levels": self.levels}

    def _divisions(self):
        return (None,) * (self.split_out + 1)

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            by_columns = self.by if not isinstance(self.by, Expr) else []
            columns = sorted(set(parent.columns + by_columns))
            if columns == self.frame.columns:
                return
            return type(parent)(
                type(self)(self.frame[columns], *self.operands[1:]),
                *parent.operands[1:],
            )


class Std(SingleAggregation):
    _parameters = ["frame", "by", "ddof", "numeric_only", "split_out", "sort"]
    _defaults = {"split_out": 1, "sort": None}

    @functools.cached_property
    def _meta(self):
        return self._lower()._meta

    def _lower(self):
        v = Var(*self.operands)
        return MapPartitions(
            v,
            func=np.sqrt,
            meta=v._meta,
            enforce_metadata=True,
            transform_divisions=True,
            clear_divisions=True,
        )


class Mean(SingleAggregation):
    @functools.cached_property
    def _meta(self):
        return self._lower()._meta

    def _lower(self):
        s = Sum(*self.operands)
        # Drop chunk/aggregate_kwargs for count
        c = Count(
            *[
                self.operand(param)
                if param not in ("chunk_kwargs", "aggregate_kwargs")
                else {}
                for param in self._parameters
            ]
        )
        if is_dataframe_like(s._meta):
            c = c[s.columns]
        return s / c


class GroupByApply(Expr):
    _parameters = [
        "frame",
        "by",
        "observed",
        "dropna",
        "_slice",
        "func",
        "args",
        "kwargs",
    ]
    _defaults = {"observed": None, "dropna": None, "_slice": None}

    @functools.cached_property
    def grp_func(self):
        return functools.partial(_groupby_slice_apply, func=self.func)

    @functools.cached_property
    def _meta(self):
        if "meta" in self.operand("kwargs"):
            return self.operand("kwargs")["meta"]
        return _meta_apply_transform(self, self.grp_func)

    def _divisions(self):
        # TODO: Can we do better? Divisions might change if we have to shuffle, so using
        # self.frame.divisions is not an option.
        return (None,) * (self.frame.npartitions + 1)

    def _shuffle_grp_func(self, shuffled=False):
        return self.grp_func

    def _lower(self):
        df = self.frame
        by = self.by

        if any(div is None for div in self.frame.divisions) or not _contains_index_name(
            self.frame._meta.index.name, self.by
        ):
            if isinstance(self.by, Expr):
                df = Assign(df, "_by", self.by)

            df = Shuffle(df, self.by, df.npartitions)
            if isinstance(self.by, Expr):
                by = Projection(df, self.by.name)
                by.name = self.by.name
                df = Projection(df, [col for col in df.columns if col != "_by"])

            grp_func = self._shuffle_grp_func(True)
        else:
            grp_func = self._shuffle_grp_func(False)

        return GroupByUDFBlockwise(
            df,
            by,
            self._slice,
            self.observed,
            self.dropna,
            self.operand("args"),
            self.operand("kwargs"),
            dask_func=grp_func,
        )


class GroupByTransform(GroupByApply):
    @functools.cached_property
    def grp_func(self):
        return functools.partial(_groupby_slice_transform, func=self.func)


class GroupByShift(GroupByApply):
    _defaults = {"observed": None, "dropna": None, "_slice": None, "func": None}

    @functools.cached_property
    def grp_func(self):
        return functools.partial(_groupby_slice_shift, shuffled=False)

    def _shuffle_grp_func(self, shuffled=False):
        return functools.partial(_groupby_slice_shift, shuffled=shuffled)


class GroupByUDFBlockwise(Blockwise):
    _parameters = [
        "frame",
        "by",
        "_slice",
        "observed",
        "dropna",
        "args",
        "kwargs",
        "dask_func",
    ]
    _defaults = {"observed": None, "dropna": None}

    @functools.cached_property
    def _meta(self):
        return _meta_apply_transform(self, self.dask_func)

    @staticmethod
    def operation(
        frame,
        by,
        _slice,
        observed=None,
        dropna=None,
        args=None,
        kwargs=None,
        dask_func=None,
    ):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        return dask_func(
            frame,
            by,
            key=_slice,
            observed=observed,
            dropna=dropna,
            *args,
            **kwargs,
        )


def _contains_index_name(index_name, by):
    if index_name is None:
        return False

    if isinstance(by, Expr):
        return False

    if not is_scalar(by):
        return False

    return index_name == by


def _meta_apply_transform(obj, grp_func):
    kwargs = obj.operand("kwargs")
    if "meta" in kwargs and kwargs["meta"] is not no_default:
        return kwargs["meta"]
    by_meta = obj.by if not isinstance(obj.by, Expr) else meta_nonempty(obj.by._meta)
    meta_args, meta_kwargs = _extract_meta((obj.operand("args"), kwargs), nonempty=True)
    return make_meta(
        grp_func(
            meta_nonempty(obj.frame._meta),
            by_meta,
            key=obj._slice,
            observed=obj.observed,
            dropna=obj.dropna,
            *meta_args,
            **meta_kwargs,
        )
    )


def _extract_meta(x, nonempty=False):
    """
    Extract internal cache data (``_meta``) from dd.DataFrame / dd.Series
    """
    if isinstance(x, Expr):
        return meta_nonempty(x._meta) if nonempty else x._meta
    elif isinstance(x, list):
        return [_extract_meta(_x, nonempty) for _x in x]
    elif isinstance(x, tuple):
        return tuple(_extract_meta(_x, nonempty) for _x in x)
    elif isinstance(x, dict):
        res = {}
        for k in x:
            res[k] = _extract_meta(x[k], nonempty)
        return res
    else:
        return x


###
### Groupby Collection API
###


class GroupBy:
    """Collection container for groupby aggregations

    The purpose of this class is to expose an API similar
    to Pandas' `Groupby` for dask-expr collections.

    See Also
    --------
    SingleAggregation
    """

    def __init__(
        self,
        obj,
        by,
        sort=None,
        observed=None,
        dropna=None,
        slice=None,
    ):
        if (
            isinstance(by, Series)
            and by.name in obj.columns
            and by._name == obj[by.name]._name
        ):
            by = by.name
        elif isinstance(by, Index) and by._name == obj.index._name:
            pass
        elif isinstance(by, Series):
            # TODO: Implement this
            raise ValueError("by must be in the DataFrames columns.")

        by_ = by if isinstance(by, (tuple, list)) else [by]
        self._slice = slice
        # Check if we can project columns
        projection = None
        if (
            np.isscalar(slice)
            or isinstance(slice, (str, list, tuple))
            or (
                (is_index_like(slice) or is_series_like(slice))
                and not is_dask_collection(slice)
            )
        ):
            projection = set(by_).union(
                {slice} if (np.isscalar(slice) or isinstance(slice, str)) else slice
            )
            projection = [c for c in obj.columns if c in projection]

        self.obj = obj[projection] if projection is not None else obj
        self.sort = sort
        self.observed = observed
        self.dropna = dropna

        if not isinstance(self.obj, DataFrame):
            raise NotImplementedError(
                "groupby only supports DataFrame collections for now."
            )

        if isinstance(by, Index):
            self.by = by.expr
        else:
            self.by = [by] if np.isscalar(by) else list(by)
            for key in self.by:
                if not (np.isscalar(key) and key in self.obj.columns):
                    raise NotImplementedError(
                        "Can only group on column names (for now)."
                    )

    def _numeric_only_kwargs(self, numeric_only):
        kwargs = {"numeric_only": numeric_only}
        return {"chunk_kwargs": kwargs, "aggregate_kwargs": kwargs}

    def _single_agg(
        self,
        expr_cls,
        split_every=8,
        split_out=1,
        chunk_kwargs=None,
        aggregate_kwargs=None,
    ):
        return new_collection(
            expr_cls(
                self.obj.expr,
                self.by,
                self.observed,
                self.dropna,
                chunk_kwargs=chunk_kwargs,
                aggregate_kwargs=aggregate_kwargs,
                _slice=self._slice,
                split_every=split_every,
                split_out=split_out,
                sort=self.sort,
            )
        )

    def _aca_agg(self, expr_cls, split_out=1, **kwargs):
        x = new_collection(
            expr_cls(
                self.obj.expr,
                by=self.by,
                split_out=split_out,
                **kwargs,
                # TODO: Add observed and dropna when supported in dask/dask
            )
        )
        return x

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e) from e

    def __getitem__(self, key):
        g = GroupBy(
            self.obj,
            by=self.by,
            slice=key,
            sort=self.sort,
            dropna=self.dropna,
            observed=self.observed,
        )
        return g

    def count(self, **kwargs):
        return self._single_agg(Count, **kwargs)

    def sum(self, numeric_only=False, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Sum, **kwargs, **numeric_kwargs)

    def prod(self, numeric_only=False, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Prod, **kwargs, **numeric_kwargs)

    def mean(self, numeric_only=False, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Mean, **kwargs, **numeric_kwargs)

    def min(self, numeric_only=False, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Min, **kwargs, **numeric_kwargs)

    def max(self, numeric_only=False, **kwargs):
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Max, **kwargs, **numeric_kwargs)

    def first(self, numeric_only=False, sort=None, **kwargs):
        if sort:
            raise NotImplementedError()
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(First, **kwargs, **numeric_kwargs)

    def last(self, numeric_only=False, sort=None, **kwargs):
        if sort:
            raise NotImplementedError()
        numeric_kwargs = self._numeric_only_kwargs(numeric_only)
        return self._single_agg(Last, **kwargs, **numeric_kwargs)

    def size(self, **kwargs):
        return self._single_agg(Size, **kwargs)

    def value_counts(self, **kwargs):
        return self._single_agg(ValueCounts, **kwargs)

    def var(self, ddof=1, numeric_only=True, split_out=1):
        if not numeric_only:
            raise NotImplementedError("numeric_only=False is not implemented")
        return self._aca_agg(
            Var,
            ddof=ddof,
            numeric_only=numeric_only,
            split_out=split_out,
            sort=self.sort,
        )

    def std(self, ddof=1, numeric_only=True, split_out=1):
        if not numeric_only:
            raise NotImplementedError("numeric_only=False is not implemented")
        return self._aca_agg(
            Std,
            ddof=ddof,
            numeric_only=numeric_only,
            split_out=split_out,
            sort=self.sort,
        )

    def aggregate(self, arg=None, split_every=8, split_out=1):
        if arg is None:
            raise NotImplementedError("arg=None not supported")

        return new_collection(
            GroupbyAggregation(
                self.obj.expr,
                self.by,
                arg,
                self.observed,
                self.dropna,
                split_every,
                split_out,
                self.sort,
            )
        )

    def agg(self, *args, **kwargs):
        return self.aggregate(*args, **kwargs)

    def apply(self, func, *args, **kwargs):
        return new_collection(
            GroupByApply(
                self.obj.expr,
                self.by,
                self.observed,
                self.dropna,
                _slice=self._slice,
                func=func,
                args=args,
                kwargs=kwargs,
            )
        )

    def transform(self, func, *args, **kwargs):
        return new_collection(
            GroupByTransform(
                self.obj.expr,
                self.by,
                self.observed,
                self.dropna,
                _slice=self._slice,
                func=func,
                args=args,
                kwargs=kwargs,
            )
        )

    def shift(self, periods=1, *args, **kwargs):
        kwargs = {"periods": periods, **kwargs}
        return new_collection(
            GroupByShift(
                self.obj.expr,
                self.by,
                self.observed,
                self.dropna,
                _slice=self._slice,
                args=args,
                kwargs=kwargs,
            )
        )
