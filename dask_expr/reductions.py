import functools
import math
import operator

import pandas as pd
import toolz
from dask.base import tokenize
from dask.dataframe.core import (
    _concat,
    hash_shard,
    is_dataframe_like,
    is_series_like,
    make_meta,
    meta_nonempty,
)
from dask.utils import M, apply, funcname

from dask_expr.expr import Blockwise, Elemwise, Expr, Projection


class ExprReference:
    """Utility to wrap an `Expr` object

    This class can be used to wrap an `Expr` object, and
    include it within the operands of another `Expr` class
    without the referenced object being included as a proper
    node in the expression graph. `ApplyConcatApply` uses
    this mechanism to split itself into multiple expressions.

    See Also
    --------
    ApplyConcatApply
    Chunk
    TreeReduction
    """

    def __init__(self, expr: Expr):
        self.expr = expr

    def __str__(self):
        return funcname(type(self.expr))

    def __repr__(self):
        return str(self)


class ApplyConcatApply(Expr):
    """Perform reduction-like operation on dataframes

    This pattern is commonly used for reductions, groupby-aggregations, and
    more.  It requires three methods to be implemented:

    -   `chunk`: applied to each input partition
    -   `combine`: applied to lists of intermediate partitions as they are
        combined in batches
    -   `aggregate`: applied at the end to finalize the computation

    These methods should be easy to serialize, and can take in keyword
    arguments defined in `chunks/combine/aggregate_kwargs`.

    In many cases people don't define all three functions.  In these cases
    combine takes from aggregate and aggregate takes from chunk.
    """

    _parameters = ["frame"]
    chunk = None
    combine = None
    aggregate = None
    split_every = 0
    chunk_kwargs = {}
    combine_kwargs = {}
    aggregate_kwargs = {}

    @functools.cached_property
    def split_out(self):
        if "split_out" in self._parameters:
            return self.operand("split_out")
        return 1

    @property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        meta = self.chunk(meta, **self.chunk_kwargs)
        meta = self.combine([meta], **self.combine_kwargs)
        meta = self.aggregate([meta], **self.aggregate_kwargs)
        return make_meta(meta)

    def _divisions(self):
        return (None,) * (self.split_out + 1)

    def _simplify_down(self):
        aca_ref = ExprReference(self)
        if True:  # self.split_out > 1:
            # raise NotImplementedError("split_out > 1 is not supported yet")
            chunked = ChunkSplit(self.frame, aca_ref, self.split_out)
            return SplitTreeReduction(
                chunked, aca_ref, self.split_out, self.split_every
            )
        else:
            chunked = Chunk(self.frame, aca_ref)
            return TreeReduction(chunked, aca_ref, self.split_every)


class Chunk(Blockwise):
    """Partition-wise component of `ApplyConcatApply`

    This class is used within `ApplyConcatApply._simplify_down`.

    See Also
    --------
    ApplyConcatApply
    """

    _parameters = ["frame", "aca"]

    @functools.cached_property
    def _name(self):
        return "chunk-" + tokenize(self.frame, self.aca._name, *self.operands[2:])

    @property
    def aca(self):
        return self.operand("aca").expr

    @property
    def chunk(self):
        return self.aca.chunk

    @property
    def chunk_kwargs(self):
        return self.aca.chunk_kwargs or {}

    @functools.cached_property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        chunk_kwargs = self.chunk_kwargs or {}
        return make_meta(self.chunk(meta, **chunk_kwargs))

    def _task(self, index: int):
        args = [self._blockwise_arg(self.frame, index)]
        if self.chunk_kwargs:
            return (apply, self.chunk, args, self.chunk_kwargs)
        else:
            return (self.chunk,) + tuple(args)

    def _divisions(self):
        return (None,) * (self.frame.npartitions + 1)


class TreeReduction(Expr):
    """Reduction component of `ApplyConcatApply`

    This class is used within `ApplyConcatApply._simplify_down`.

    See Also
    --------
    ApplyConcatApply
    """

    _parameters = ["frame", "aca", "split_every"]
    _defaults = {"split_every": 0}

    @functools.cached_property
    def _name(self):
        return "reduction-" + tokenize(self.frame, self.aca._name, *self.operands[2:])

    @property
    def aca(self):
        return self.operand("aca").expr

    @property
    def aggregate(self):
        return self.aca.aggregate or self.aca.chunk

    @property
    def combine(self):
        return self.aca.combine or self.aggregate

    @property
    def aggregate_kwargs(self):
        return self.aca.aggregate_kwargs or {}

    @property
    def combine_kwargs(self):
        if self.aca.combine:
            return self.aca.combine_kwargs or {}
        return self.aggregate_kwargs

    def _layer(self):
        aggregate = self.aggregate
        aggregate_kwargs = self.aggregate_kwargs
        combine = self.combine
        combine_kwargs = self.combine_kwargs

        d = {}
        j = 1
        keys = self.frame.__dask_keys__()
        # apply combine to batches of intermediate results
        while len(keys) > 1:
            new_keys = []
            for i, batch in enumerate(
                toolz.partition_all(self.split_every or len(keys), keys)
            ):
                batch = list(batch)
                if combine_kwargs:
                    d[(self._name, j, i)] = (apply, combine, [batch], combine_kwargs)
                else:
                    d[(self._name, j, i)] = (combine, batch)
                new_keys.append((self._name, j, i))
            j += 1
            keys = new_keys

        # apply aggregate to the final result
        d[(self._name, 0)] = (apply, aggregate, [keys], aggregate_kwargs)

        return d

    @property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        meta = self.combine([meta], **self.combine_kwargs)
        meta = self.aggregate([meta], **self.aggregate_kwargs)
        return make_meta(meta)

    def _divisions(self):
        return (None, None)


class ChunkSplit(Chunk):
    _parameters = ["frame", "aca", "split_out"]

    @staticmethod
    def chunk_split(*args, chunk_func=None, split_out=None, **kwargs):
        chunked = chunk_func(*args, **kwargs)
        return hash_shard(
            chunked,
            split_out,
            # self.split_out_setup,
            # self.split_out_setup_kwargs,
            # self.ignore_index,
        )

    @functools.cached_property
    def chunk(self):
        return functools.partial(
            self.chunk_split,
            split_out=self.split_out,
            chunk_func=self.aca.chunk,
        )


class SplitTreeReduction(TreeReduction):
    _parameters = ["frame", "aca", "split_out", "split_every"]
    _defaults = {"split_every": 16}

    def _layer(self):
        aggregate = self.aggregate
        aggregate_kwargs = self.aggregate_kwargs
        combine = self.combine
        combine_kwargs = self.combine_kwargs
        output_splits = range(self.split_out)
        split_name = "split-" + self._name
        tree_node_name = "tree_node-" + self._name

        # Calculate reduction-tree properties
        parts = self.frame.npartitions
        widths = [parts]
        while parts > 1:
            parts = math.ceil(parts / self.split_every)
            widths.append(int(parts))
        height = len(widths)

        # Get each split
        dsk = {}
        keys = self.frame.__dask_keys__()
        for s in output_splits:
            for p, key in enumerate(keys):
                dsk[(split_name, p, s)] = (operator.getitem, key, s)

        if height >= 2:
            # Loop over output splits
            for s in output_splits:
                # Loop over reduction levels
                for depth in range(1, height):
                    # Loop over reduction groups
                    for group in range(widths[depth]):
                        # Calculate inputs for the current group
                        p_max = widths[depth - 1]
                        lstart = self.split_every * group
                        lstop = min(lstart + self.split_every, p_max)
                        if depth == 1:
                            # Input nodes are from input layer
                            keys = [(split_name, p, s) for p in range(lstart, lstop)]
                        else:
                            # Input nodes are tree-reduction nodes
                            keys = [
                                (tree_node_name, p, depth - 1, s)
                                for p in range(lstart, lstop)
                            ]

                        # Define task
                        if depth == height - 1:
                            # Final Node (Use fused `self.tree_finalize` task)
                            assert (
                                group == 0
                            ), f"group = {group}, not 0 for final tree reduction task"
                            dsk[(self._name, s)] = (
                                apply,
                                aggregate,
                                [keys],
                                aggregate_kwargs,
                            )

                        else:
                            # Intermediate Node
                            if combine_kwargs:
                                dsk[(tree_node_name, group, depth, s)] = (
                                    apply,
                                    combine,
                                    [keys],
                                    combine_kwargs,
                                )
                            else:
                                dsk[(tree_node_name, group, depth, s)] = (combine, keys)

        else:
            # Deal with single-partition case
            for s in output_splits:
                keys = [(split_name, 0, s)]
                dsk[(self._name, s)] = (apply, aggregate, [keys], aggregate_kwargs)

        return dsk

        # d = {}
        # j = 1
        # keys = self.frame.__dask_keys__()
        # # apply combine to batches of intermediate results
        # while len(keys) > 1:
        #     new_keys = []
        #     for i, batch in enumerate(
        #         toolz.partition_all(self.split_every or len(keys), keys)
        #     ):
        #         batch = list(batch)
        #         if combine_kwargs:
        #             d[(self._name, j, i)] = (apply, combine, [batch], combine_kwargs)
        #         else:
        #             d[(self._name, j, i)] = (combine, batch)
        #         new_keys.append((self._name, j, i))
        #     j += 1
        #     keys = new_keys

        # # apply aggregate to the final result
        # d[(self._name, 0)] = (apply, aggregate, [keys], aggregate_kwargs)

        # return d


class Reduction(ApplyConcatApply):
    """A common pattern of apply concat apply

    Common reductions like sum/min/max/count/... have some shared code around
    `_concat` and so on.  This class inherits from `ApplyConcatApply` in order
    to leverage this shared structure.

    I wouldn't be surprised if there was a way to merge them both into a single
    abstraction in the future.

    This class implements `{chunk,combine,aggregate}` methods of
    `ApplyConcatApply` by depending on `reduction_{chunk,combine,aggregate}`
    methods.
    """

    reduction_chunk = None
    reduction_combine = None
    reduction_aggregate = None

    @functools.cached_property
    def split_out(self):
        # Require split_out == 1 for Reductions
        if "split_out" in self._parameters and self.operand("split_out") != 1:
            raise ValueError("`split_out` must be 1 for a Reduction")
        return 1

    @classmethod
    def chunk(cls, df, **kwargs):
        out = cls.reduction_chunk(df, **kwargs)
        # Return a dataframe so that the concatenated version is also a dataframe
        return out.to_frame().T if is_series_like(out) else out

    @classmethod
    def combine(cls, inputs: list, **kwargs):
        func = cls.reduction_combine or cls.reduction_aggregate or cls.reduction_chunk
        df = _concat(inputs)
        out = func(df, **kwargs)
        # Return a dataframe so that the concatenated version is also a dataframe
        return out.to_frame().T if is_series_like(out) else out

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        func = cls.reduction_aggregate or cls.reduction_chunk
        df = _concat(inputs)
        return func(df, **kwargs)

    def __dask_postcompute__(self):
        return toolz.first, ()

    def _divisions(self):
        return (None, None)

    def __str__(self):
        params = {param: self.operand(param) for param in self._parameters[1:]}
        s = ", ".join(
            k + "=" + repr(v) for k, v in params.items() if v != self._defaults.get(k)
        )
        base = str(self.frame)
        if " " in base:
            base = "(" + base + ")"
        return f"{base}.{self.__class__.__name__.lower()}({s})"


class Sum(Reduction):
    _parameters = ["frame", "skipna", "numeric_only", "min_count"]
    _defaults = {
        "skipna": True,
        "numeric_only": None,
        "min_count": 0,
    }
    reduction_chunk = M.sum

    @property
    def chunk_kwargs(self):
        return dict(
            skipna=self.skipna,
            numeric_only=self.numeric_only,
            min_count=self.min_count,
        )

    @property
    def _meta(self):
        return self.frame._meta.sum(**self.chunk_kwargs)

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return self.frame[parent.operand("columns")].sum(*self.operands[1:])


class Max(Reduction):
    _parameters = ["frame", "skipna", "numeric_only"]
    _defaults = {"skipna": True, "numeric_only": None}
    reduction_chunk = M.max

    @property
    def chunk_kwargs(self):
        return dict(
            skipna=self.skipna,
        )

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return self.frame[parent.operand("columns")].max(skipna=self.skipna)


class Len(Reduction):
    reduction_chunk = staticmethod(len)
    reduction_aggregate = sum

    def _simplify_down(self):
        if isinstance(self.frame, Elemwise):
            child = max(self.frame.dependencies(), key=lambda expr: expr.npartitions)
            return Len(child)


class Size(Reduction):
    reduction_chunk = staticmethod(lambda df: df.size)
    reduction_aggregate = sum

    def _simplify_down(self):
        if is_dataframe_like(self.frame) and len(self.frame.columns) > 1:
            return len(self.frame.columns) * Len(self.frame)
        else:
            return Len(self.frame)


class Mean(Reduction):
    _parameters = ["frame", "skipna", "numeric_only"]
    _defaults = {"skipna": True, "numeric_only": None}

    @property
    def _meta(self):
        return (
            self.frame._meta.sum(skipna=self.skipna, numeric_only=self.numeric_only) / 2
        )

    def _simplify_down(self):
        return (
            self.frame.sum(skipna=self.skipna, numeric_only=self.numeric_only)
            / self.frame.count()
        )


class Count(Reduction):
    _parameters = ["frame", "numeric_only"]
    _defaults = {"numeric_only": None}
    split_every = 16
    reduction_chunk = M.count

    @classmethod
    def reduction_aggregate(cls, df):
        return df.sum().astype("int64")


class Min(Max):
    reduction_chunk = M.min


class Mode(ApplyConcatApply):
    """

    Mode was a bit more complicated than class reductions, so we retreat back
    to ApplyConcatApply
    """

    _parameters = ["frame", "dropna", "split_out"]
    _defaults = {"dropna": True, "split_out": 1}
    chunk = M.value_counts
    split_every = 16

    @classmethod
    def combine(cls, results: list[pd.Series]):
        df = _concat(results)
        out = df.groupby(df.index).sum()
        out.name = results[0].name
        return out

    @classmethod
    def aggregate(cls, results: list[pd.Series], dropna=None):
        df = _concat(results) if len(results) > 1 else results[0]
        max = df.max(skipna=dropna)
        out = df[df == max].index.to_series().sort_values().reset_index(drop=True)
        out.name = results[0].name
        return out

    @property
    def chunk_kwargs(self):
        return {"dropna": self.dropna}

    @property
    def aggregate_kwargs(self):
        return {"dropna": self.dropna}
