import functools

import numpy as np
from dask.dataframe.core import _concat, make_meta
from dask.dataframe.dispatch import meta_nonempty

from dask_expr._reductions import ApplyConcatApply
from dask_expr._util import _convert_to_list

OPTIMIZED_AGGS = (
    "count",
    "mean",
    "std",
    "var",
    "sum",
    "min",
    "max",
    "collect",
    "first",
    "last",
)


def _normalize_aggs(arg):
    """Redirect aggregations to their corresponding name in cuDF"""
    redirects = {
        sum: "sum",
        max: "max",
        min: "min",
        list: "collect",
        "list": "collect",
    }
    if isinstance(arg, dict):
        new_arg = dict()
        for col in arg:
            if isinstance(arg[col], list):
                new_arg[col] = [redirects.get(agg, agg) for agg in arg[col]]
            elif isinstance(arg[col], dict):
                new_arg[col] = {k: redirects.get(v, v) for k, v in arg[col].items()}
            else:
                new_arg[col] = redirects.get(arg[col], arg[col])
        return new_arg
    if isinstance(arg, list):
        return [redirects.get(agg, agg) for agg in arg]
    return redirects.get(arg, arg)


def _aggs_optimized(arg, supported: set):
    """Check that aggregations in `arg` are a subset of `supported`"""
    if isinstance(arg, (list, dict)):
        if isinstance(arg, dict):
            _global_set = set()
            for col in arg:
                if isinstance(arg[col], list):
                    _global_set = _global_set.union(set(arg[col]))
                elif isinstance(arg[col], dict):
                    _global_set = _global_set.union(set(arg[col].values()))
                else:
                    _global_set.add(arg[col])
        else:
            _global_set = set(arg)

        return bool(_global_set.issubset(supported))
    elif isinstance(arg, str):
        return arg in supported
    return False


def _coarse_supported(by, arg):
    _arg = _normalize_aggs(arg)
    return isinstance(by, (list, str)) and _aggs_optimized(_arg, OPTIMIZED_AGGS)


def _make_name(col_name, sep="_"):
    """Combine elements of `col_name` into a single string, or no-op if
    `col_name` is already a string
    """
    if isinstance(col_name, str):
        return col_name
    return sep.join(name for name in col_name if name != "")


def _groupby_partition_agg(
    df, gb_cols=None, aggs=None, dropna=True, sort=False, sep="__"
):
    """Initial partition-level aggregation task.

    This is the first operation to be executed on each input
    partition in `groupby_agg`.  Depending on `aggs`, four possible
    groupby aggregations ("count", "sum", "min", and "max") are
    performed.  The result is then partitioned (by hashing `gb_cols`)
    into a number of distinct dictionary elements.  The number of
    elements in the output dictionary (`split_out`) corresponds to
    the number of partitions in the final output of `groupby_agg`.
    """

    # Modify dict for initial (partition-wise) aggregations
    _agg_dict = {}
    for col, agg_list in aggs.items():
        _agg_dict[col] = set()
        for agg in agg_list:
            if agg in ("mean", "std", "var"):
                _agg_dict[col].add("count")
                _agg_dict[col].add("sum")
            else:
                _agg_dict[col].add(agg)
        _agg_dict[col] = list(_agg_dict[col])
        if set(agg_list).intersection({"std", "var"}):
            pow2_name = _make_name((col, "pow2"), sep=sep)
            df[pow2_name] = df[col].astype("float64").pow(2)
            _agg_dict[pow2_name] = ["sum"]

    gb = (
        df.groupby(gb_cols, dropna=dropna, sort=sort)
        .agg(_agg_dict)
        .reset_index(drop=False)
    )
    output_columns = [_make_name(name, sep=sep) for name in gb.columns]
    gb.columns = output_columns
    # Return with deterministic column ordering
    return gb[sorted(output_columns)]


def _tree_node_agg(df, gb_cols=None, dropna=True, sort=False, sep="__"):
    """Node in groupby-aggregation reduction tree.

    The input DataFrame (`df`) corresponds to the
    concatenated output of one or more `_groupby_partition_agg`
    tasks. In this function, "sum", "min" and/or "max" groupby
    aggregations will be used to combine the statistics for
    duplicate keys.
    """

    agg_dict = {}
    for col in df.columns:
        if col in gb_cols:
            continue
        agg = col.split(sep)[-1]
        if agg in ("count", "sum"):
            agg_dict[col] = ["sum"]
        elif agg in OPTIMIZED_AGGS:
            agg_dict[col] = [agg]
        else:
            raise ValueError(f"Unexpected aggregation: {agg}")

    gb = (
        df.groupby(gb_cols, dropna=dropna, sort=sort)
        .agg(agg_dict)
        .reset_index(drop=False)
    )

    # Don't include the last aggregation in the column names
    output_columns = [
        _make_name(name[:-1] if isinstance(name, tuple) else name, sep=sep)
        for name in gb.columns
    ]
    gb.columns = output_columns
    # Return with deterministic column ordering
    return gb[sorted(output_columns)]


def _var_agg_2(df, count_name, sum_name, pow2_sum_name, ddof=1):
    """Calculate variance (given count, sum, and sum-squared columns)."""

    # Select count, sum, and sum-squared
    n = df[count_name]
    x = df[sum_name]
    x2 = df[pow2_sum_name]

    # Use sum-squared approach to get variance
    var = x2 - x**2 / n
    div = n - ddof
    div[div < 1] = 1  # Avoid division by 0
    var /= div

    # Set appropriate NaN elements
    # (since we avoided 0-division)
    var[(n - ddof) == 0] = np.nan

    return var


def _finalize_gb_agg(
    gb_in,
    gb_cols=None,
    aggs=None,
    columns=None,
    final_columns=None,
    as_index=True,
    dropna=True,
    sort=False,
    sep="__",
    str_cols_out=None,
    aggs_renames=None,
):
    """Final aggregation task.

    This is the final operation on each output partitions
    of the `groupby_agg` algorithm.  This function must
    take care of higher-order aggregations, like "mean",
    "std" and "var".  We also need to deal with the column
    index, the row index, and final sorting behavior.
    """
    import pandas as pd

    gb = _tree_node_agg(gb_in, gb_cols, dropna, sort, sep)

    # Deal with higher-order aggregations
    for col in columns:
        agg_list = aggs.get(col, [])
        agg_set = set(agg_list)
        if agg_set.intersection({"mean", "std", "var"}):
            count_name = _make_name((col, "count"), sep=sep)
            sum_name = _make_name((col, "sum"), sep=sep)
            if agg_set.intersection({"std", "var"}):
                pow2_sum_name = _make_name((col, "pow2", "sum"), sep=sep)
                var = _var_agg_2(gb, count_name, sum_name, pow2_sum_name)
                if "var" in agg_list:
                    name_var = _make_name((col, "var"), sep=sep)
                    gb[name_var] = var
                if "std" in agg_list:
                    name_std = _make_name((col, "std"), sep=sep)
                    gb[name_std] = np.sqrt(var)
                gb.drop(columns=[pow2_sum_name], inplace=True)
            if "mean" in agg_list:
                mean_name = _make_name((col, "mean"), sep=sep)
                gb[mean_name] = gb[sum_name] / gb[count_name]
            if "sum" not in agg_list:
                gb.drop(columns=[sum_name], inplace=True)
            if "count" not in agg_list:
                gb.drop(columns=[count_name], inplace=True)
        if "collect" in agg_list:
            collect_name = _make_name((col, "collect"), sep=sep)
            gb[collect_name] = gb[collect_name].list.concat()

    # Ensure sorted keys if `sort=True`
    if sort:
        gb = gb.sort_values(gb_cols)

    # Set index if necessary
    if as_index:
        gb.set_index(gb_cols, inplace=True)

    # Unflatten column names
    col_array = []
    agg_array = []
    for col in gb.columns:
        if col in gb_cols:
            col_array.append(col)
            agg_array.append("")
        else:
            name, agg = col.split(sep)
            col_array.append(name)
            agg_array.append(aggs_renames.get((name, agg), agg))
    if str_cols_out:
        gb.columns = col_array
    else:
        gb.columns = pd.MultiIndex.from_arrays([col_array, agg_array])

    return gb[final_columns]


class CoarseGroupbyAggregation(ApplyConcatApply):
    _parameters = [
        "frame",
        "by",
        "arg",
        "observed",
        "dropna",
        "split_every",
        "sep",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
        "split_every": 8,
        "sep": "__",
    }

    @functools.cached_property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        return make_meta(meta.groupby(self.by).agg(self.arg))

    @functools.cached_property
    def spec(self):
        # Normalize `gb_cols`, `columns`, and `aggs`
        gb_cols = _convert_to_list(self.by)
        aggs = self.arg
        columns = [c for c in self.frame.columns if c not in gb_cols]
        if not isinstance(aggs, dict):
            aggs = {col: aggs for col in columns}
        else:
            aggs = aggs.copy()  # Make sure we dont modify self.arg

        # Assert if our output will have a MultiIndex; this will be the case if
        # any value in the `aggs` dict is not a string (i.e. multiple/named
        # aggregations per column)
        str_cols_out = True
        aggs_renames = {}
        for col in aggs:
            if isinstance(aggs[col], str) or callable(aggs[col]):
                aggs[col] = [aggs[col]]
            elif isinstance(aggs[col], dict):
                str_cols_out = False
                col_aggs = []
                for k, v in aggs[col].items():
                    aggs_renames[col, v] = k
                    col_aggs.append(v)
                aggs[col] = col_aggs
            else:
                str_cols_out = False
            if col in gb_cols:
                columns.append(col)

        return {
            "gb_cols": gb_cols,
            "columns": columns,
            "aggs": aggs,
            "str_cols_out": str_cols_out,
            "aggs_renames": aggs_renames,
        }

    @classmethod
    def chunk(cls, inputs, **kwargs):
        return _groupby_partition_agg(inputs, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        return _tree_node_agg(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        return _finalize_gb_agg(_concat(inputs), **kwargs)

    @property
    def chunk_kwargs(self) -> dict:
        spec = self.spec
        # df, gb_cols, aggs, dropna, sort, sep
        return {
            "gb_cols": spec["gb_cols"],
            "aggs": spec["aggs"],
            "dropna": self.dropna,
            "sort": False,
            "sep": self.sep,
        }

    @property
    def combine_kwargs(self) -> dict:
        spec = self.spec
        # df, gb_cols, dropna, sort, sep
        return {
            "gb_cols": spec["gb_cols"],
            "dropna": self.dropna,
            "sort": False,
            "sep": self.sep,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        spec = self.spec
        # gb_in, gb_cols, aggs, columns, final_columns, as_index,
        # dropna, sort, sep, str_cols_out, aggs_renames
        return {
            "gb_cols": spec["gb_cols"],
            "aggs": spec["aggs"],
            "columns": spec["columns"],
            "final_columns": self.columns,
            "as_index": True,
            "dropna": self.dropna,
            "sort": False,
            "sep": self.sep,
            "str_cols_out": spec["str_cols_out"],
            "aggs_renames": spec["aggs_renames"],
        }

    def _simplify_down(self):
        # Use agg-spec information to add column projection
        gb_cols = self.spec["gb_cols"]
        aggs = self.spec["aggs"]
        column_projection = (
            set(gb_cols).union(aggs.keys()).intersection(self.frame.columns)
        )
        if column_projection and column_projection < set(self.frame.columns):
            return type(self)(self.frame[list(column_projection)], *self.operands[1:])
