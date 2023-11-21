from __future__ import annotations

import functools
import math
import operator

from dask.dataframe import methods
from dask.dataframe.core import apply_and_enforce, is_dataframe_like, make_meta
from dask.dataframe.io.io import sorted_division_locations
from dask.utils import apply, funcname

from dask_expr._expr import (
    Blockwise,
    Expr,
    Lengths,
    Literal,
    PartitionsFiltered,
    Projection,
    no_default,
)
from dask_expr._reductions import Len
from dask_expr._util import _BackendData, _convert_to_list, _tokenize_deterministic


class IO(Expr):
    def __str__(self):
        return f"{type(self).__name__}({self._name[-7:]})"


class FromGraph(IO):
    """A DataFrame created from an opaque Dask task graph

    This is used in persist, for example, and would also be used in any
    conversion from legacy dataframes.
    """

    _parameters = ["layer", "_meta", "divisions", "_name"]

    @property
    def _meta(self):
        return self.operand("_meta")

    def _divisions(self):
        return self.operand("divisions")

    @property
    def _name(self):
        return self.operand("_name")

    def _layer(self):
        return dict(self.operand("layer"))


class BlockwiseIO(Blockwise, IO):
    _absorb_projections = False

    @functools.cached_property
    def _fusion_compression_factor(self):
        return 1

    def _simplify_up(self, parent):
        if (
            self._absorb_projections
            and isinstance(parent, Projection)
            and is_dataframe_like(self._meta)
        ):
            # Column projection
            parent_columns = parent.operand("columns")
            proposed_columns = _convert_to_list(parent_columns)
            make_series = isinstance(parent_columns, (str, int)) and not self._series
            if set(proposed_columns) == set(self.columns) and not make_series:
                # Already projected
                return
            substitutions = {"columns": _convert_to_list(parent_columns)}
            if make_series:
                substitutions["_series"] = True
            return self.substitute_parameters(substitutions)

    def _combine_similar(self, root: Expr):
        if self._absorb_projections:
            # For BlockwiseIO expressions with "columns"/"_series"
            # attributes (`_absorb_projections == True`), we can avoid
            # redundant file-system access by aggregating multiple
            # operations with different column projections into the
            # same operation.
            alike = self._find_similar_operations(root, ignore=["columns", "_series"])
            if alike:
                # We have other BlockwiseIO operations (of the same
                # sub-type) in the expression graph that can be combined
                # with this one.

                # Find the column-projection union needed to combine
                # the qualified BlockwiseIO operations
                columns_operand = self.operand("columns")
                if columns_operand is None:
                    columns_operand = self.columns
                columns = set(columns_operand)
                for op in alike:
                    op_columns = op.operand("columns")
                    if op_columns is None:
                        op_columns = op.columns
                    columns |= set(op_columns)
                columns = sorted(columns)
                if columns_operand is None:
                    columns_operand = self.columns
                # Can bail if we are not changing columns or the "_series" operand
                if columns_operand == columns and (
                    len(columns) > 1 or not self._series
                ):
                    return

                # Check if we have the operation we want elsewhere in the graph
                for op in alike:
                    if set(op.columns) == set(columns) and not op.operand("_series"):
                        return (
                            op[columns_operand[0]]
                            if self._series
                            else op[columns_operand]
                        )

                if set(self.columns) == set(columns):
                    return  # Skip unnecessary projection change

                # Create the "combined" ReadParquet operation
                subs = {"columns": columns}
                if self._series:
                    subs["_series"] = False
                new = self.substitute_parameters(subs)
                return new[columns_operand[0]] if self._series else new[columns_operand]

        return

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        if isinstance(parent, FusedIO):
            return
        return parent.substitute(self, FusedIO(self))


class FusedIO(BlockwiseIO):
    _parameters = ["expr"]

    @functools.cached_property
    def _name(self):
        return (
            funcname(type(self.operand("expr"))).lower()
            + "-fused-"
            + _tokenize_deterministic(*self.operands)
        )

    @functools.cached_property
    def _meta(self):
        return self.operand("expr")._meta

    def dependencies(self):
        return []

    @functools.cached_property
    def npartitions(self):
        return len(self._fusion_buckets)

    def _divisions(self):
        divisions = self.operand("expr")._divisions()
        new_divisions = [divisions[b[0]] for b in self._fusion_buckets]
        new_divisions.append(self._fusion_buckets[-1][-1])
        return tuple(new_divisions)

    def _task(self, index: int):
        expr = self.operand("expr")
        bucket = self._fusion_buckets[index]
        return (methods.concat, [expr._filtered_task(i) for i in bucket])

    @functools.cached_property
    def _fusion_buckets(self):
        partitions = self.operand("expr")._partitions
        npartitions = len(partitions)

        step = math.ceil(1 / self.operand("expr")._fusion_compression_factor)
        step = min(step, math.ceil(math.sqrt(npartitions)), 100)

        buckets = [partitions[i : i + step] for i in range(0, npartitions, step)]
        return buckets

    def _tune_up(self, parent):
        return


class FromMap(PartitionsFiltered, BlockwiseIO):
    _parameters = [
        "func",
        "iterables",
        "args",
        "kwargs",
        "user_meta",
        "enforce_metadata",
        "user_divisions",
        "label",
        "_partitions",
    ]
    _defaults = {
        "user_meta": no_default,
        "enforce_metadata": False,
        "user_divisions": None,
        "label": None,
        "_partitions": None,
    }
    _absorb_projections = False

    @functools.cached_property
    def _name(self):
        if self.label is None:
            return (
                funcname(self.func).lower()
                + "-"
                + _tokenize_deterministic(*self.operands)
            )
        else:
            return self.label + "-" + _tokenize_deterministic(*self.operands)

    @functools.cached_property
    def _meta(self):
        if self.operand("user_meta") is not no_default:
            meta = self.operand("user_meta")
        else:
            vals = [v[0] for v in self.iterables]
            meta = self.func(*vals, *self.args, **self.kwargs)
        return make_meta(meta)

    def _divisions(self):
        if self.operand("user_divisions"):
            return self.operand("user_divisions")
        else:
            npartitions = len(self.iterables[0])
            return (None,) * (npartitions + 1)

    @property
    def apply_func(self):
        if self.enforce_metadata:
            return apply_and_enforce
        return self.func

    @functools.cached_property
    def apply_kwargs(self):
        kwargs = self.kwargs
        if self.enforce_metadata:
            kwargs = kwargs.copy()
            kwargs.update(
                {
                    "_func": self.func,
                    "_meta": self._meta,
                }
            )
        return kwargs

    def _filtered_task(self, index: int):
        vals = [v[index] for v in self.iterables]
        if self.apply_kwargs:
            return (apply, self.apply_func, vals + self.args, self.apply_kwargs)
        else:
            return (self.func, *vals, *self.args)


class FromMapProjectable(FromMap):
    _parameters = [
        "func",
        "iterables",
        "columns",
        "args",
        "kwargs",
        "user_meta",
        "enforce_metadata",
        "user_divisions",
        "label",
        "_partitions",
        "_series",
    ]
    _defaults = {
        "user_meta": no_default,
        "enforce_metadata": False,
        "user_divisions": None,
        "label": None,
        "_partitions": None,
        "_series": False,
    }
    _absorb_projections = True

    @functools.cached_property
    def columns_operand(self):
        return _convert_to_list(self.operand("columns"))

    @property
    def columns(self):
        if self.columns_operand is None:
            return list(self.frame_meta.columns)
        else:
            return self.columns_operand

    @functools.cached_property
    def _series(self):
        # Only need to convert to _series if func
        # doesn't produce a Series already
        return self.operand("_series") and self.frame_meta.ndim > 1

    @functools.cached_property
    def kwargs(self):
        options = self.operand("kwargs")
        if self.columns_operand:
            options = options.copy()
            options["columns"] = self.columns_operand
        return options

    @functools.cached_property
    def apply_kwargs(self):
        kwargs = self.kwargs
        if self.enforce_metadata:
            kwargs = kwargs.copy()
            kwargs.update(
                {
                    "_func": self.func,
                    "_meta": self.frame_meta,
                }
            )
        return kwargs

    @functools.cached_property
    def frame_meta(self):
        # This is our `_meta` result before possibly
        # converting to a Series
        meta = super()._meta
        if meta.ndim > 1 and self.columns_operand is not None:
            return meta[self.columns_operand]
        return meta

    @property
    def _meta(self):
        # This is our final `_meta` result
        # (may need to be a Series)
        meta = self.frame_meta
        if self._series:
            assert len(self.columns_operand) > 0
            return meta[self.columns_operand[0]]
        return meta

    def _filtered_task(self, index: int):
        tsk = super()._filtered_task(index)
        if self._series:
            return (operator.getitem, tsk, self.columns[0])
        return tsk


class FromPandas(PartitionsFiltered, BlockwiseIO):
    """The only way today to get a real dataframe"""

    _parameters = ["frame", "npartitions", "sort", "columns", "_partitions", "_series"]
    _defaults = {
        "npartitions": 1,
        "sort": True,
        "columns": None,
        "_partitions": None,
        "_series": False,
    }
    _pd_length_stats = None
    _absorb_projections = True

    @functools.cached_property
    def frame(self):
        frame = self.operand("frame")._data
        if self.sort and not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()
            return _BackendData(frame)
        return self.operand("frame")

    def __exec__(self):
        pdf = self.operand("frame")._data
        if self.columns:
            return pdf[self.columns[0]] if self._series else pdf[self.columns]
        return pdf

    @functools.cached_property
    def _meta(self):
        meta = self.frame.head(0)
        if self.operand("columns") is not None:
            return meta[self.columns[0]] if self._series else meta[self.columns]
        return meta

    @functools.cached_property
    def columns(self):
        columns_operand = self.operand("columns")
        if columns_operand is None:
            try:
                return list(self.frame.columns)
            except AttributeError:
                return []
        else:
            return _convert_to_list(columns_operand)

    @property
    def _divisions_and_locations(self):
        assert isinstance(self.frame, _BackendData)
        npartitions = self.operand("npartitions")
        sort = self.sort
        key = (npartitions, sort)
        _division_info_cache = self.frame._division_info
        if key not in _division_info_cache:
            data = self.frame._data
            nrows = len(data)
            if sort:
                divisions, locations = sorted_division_locations(
                    data.index,
                    npartitions=npartitions,
                    chunksize=None,
                )
            else:
                chunksize = int(math.ceil(nrows / npartitions))
                locations = list(range(0, nrows, chunksize)) + [len(data)]
                divisions = (None,) * len(locations)
            _division_info_cache[key] = divisions, locations
        return _division_info_cache[key]

    def _get_lengths(self) -> tuple | None:
        if self._pd_length_stats is None:
            locations = self._locations()
            self._pd_length_stats = tuple(
                offset - locations[i]
                for i, offset in enumerate(locations[1:])
                if not self._filtered or i in self._partitions
            )
        return self._pd_length_stats

    def _simplify_up(self, parent):
        if isinstance(parent, Lengths):
            _lengths = self._get_lengths()
            if _lengths:
                return Literal(_lengths)

        if isinstance(parent, Len):
            _lengths = self._get_lengths()
            if _lengths:
                return Literal(sum(_lengths))

        if isinstance(parent, Projection):
            return super()._simplify_up(parent)

    def _divisions(self):
        return self._divisions_and_locations[0]

    def _locations(self):
        return self._divisions_and_locations[1]

    def _filtered_task(self, index: int):
        start, stop = self._locations()[index : index + 2]
        part = self.frame.iloc[start:stop]
        if self.columns:
            return part[self.columns[0]] if self._series else part[self.columns]
        return part

    def __str__(self):
        if self._absorb_projections and self.operand("columns"):
            if self._series:
                return f"df[{self.columns[0]}]"
            return f"df[{self.columns}]"
        return "df"

    __repr__ = __str__


class FromPandasDivisions(FromPandas):
    _parameters = ["frame", "divisions", "columns", "_partitions", "_series"]
    _defaults = {"columns": None, "_partitions": None, "_series": False}
    sort = True

    @property
    def _divisions_and_locations(self):
        assert isinstance(self.frame, _BackendData)
        key = tuple(self.operand("divisions"))
        _division_info_cache = self.frame._division_info
        if key not in _division_info_cache:
            data = self.frame._data
            indexer = data.index.get_indexer(key, method="bfill")
            indexer[-1] = len(data)
            _division_info_cache[key] = key, indexer
        return _division_info_cache[key]
