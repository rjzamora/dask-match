from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

from dask.dataframe.dispatch import make_meta
from dask.dataframe.utils import check_meta
from dask.delayed import Delayed, delayed

from dask_expr import new_collection
from dask_expr._expr import Expr, PartitionsFiltered
from dask_expr.io import BlockwiseIO

if TYPE_CHECKING:
    import distributed


class _DelayedExpr(Expr):
    # Wraps a Delayed object to make it an Expr for now. This is hacky and we should
    # integrate this properly...
    # TODO

    def __init__(self, obj):
        self.obj = obj
        self.operands = [obj]

    @property
    def _name(self):
        return self.obj.key

    def _layer(self) -> dict:
        return self.obj.dask.to_dict()

    def _divisions(self):
        return (None, None)


class FromDelayed(PartitionsFiltered, BlockwiseIO):
    _parameters = ["meta", "user_divisions", "verify_meta", "_partitions"]
    _defaults = {
        "meta": None,
        "_partitions": None,
        "user_divisions": None,
        "verify_meta": True,
    }

    def dependencies(self):
        return self.dfs

    @functools.cached_property
    def dfs(self):
        return self.operands[len(self._parameters) :]

    @functools.cached_property
    def _meta(self):
        if self.operand("meta") is not None:
            return self.operand("meta")

        return delayed(make_meta)(self.dfs[0]).compute()

    def _divisions(self):
        if self.operand("user_divisions") is not None:
            return self.operand("user_divisions")
        else:
            return (None,) * (len(self.dfs) + 1)

    def _filtered_task(self, index: int):
        key = self.dfs[index]._name
        if self.verify_meta:
            return (
                functools.partial(check_meta, meta=self._meta, funcname="from_delayed"),
                key,
            )
        else:
            return identity, key


def identity(x):
    return x


def from_delayed(
    dfs: Delayed | distributed.Future | Iterable[Delayed | distributed.Future],
    meta=None,
    divisions: tuple | None = None,
    verify_meta: bool = True,
):
    """Create Dask DataFrame from many Dask Delayed objects

    Parameters
    ----------
    dfs :
        A ``dask.delayed.Delayed``, a ``distributed.Future``, or an iterable of either
        of these objects, e.g. returned by ``client.submit``. These comprise the
        individual partitions of the resulting dataframe.
        If a single object is provided (not an iterable), then the resulting dataframe
        will have only one partition.
    $META
    divisions :
        Partition boundaries along the index.
        For tuple, see https://docs.dask.org/en/latest/dataframe-design.html#partitions
        If None, then won't use index information
    prefix :
        Prefix to prepend to the keys.
    verify_meta :
        If True check that the partitions have consistent metadata, defaults to True.
    """
    if isinstance(dfs, Delayed) or hasattr(dfs, "key"):
        dfs = [dfs]

    if len(dfs) == 0:
        raise TypeError("Must supply at least one delayed object")

    if divisions == "sorted":
        raise NotImplementedError(
            "divisions='sorted' not supported, please calculate the divisions "
            "yourself."
        )

    dfs = [
        delayed(df) if not isinstance(df, Delayed) and hasattr(df, "key") else df
        for df in dfs
    ]

    for item in dfs:
        if not isinstance(item, Delayed):
            raise TypeError("Expected Delayed object, got %s" % type(item).__name__)

    dfs = [_DelayedExpr(df) for df in dfs]

    return new_collection(
        FromDelayed(make_meta(meta), divisions, verify_meta, None, *dfs)
    )
