from __future__ import annotations

import operator
from functools import cached_property

from dask_expr.expr import EQ, GE, GT, LE, LT, NE, Expr, Filter, Projection
from dask_expr.io import BlockwiseIO, PartitionsFiltered


def _list_columns(columns):
    # Simple utility to convert columns to list
    if isinstance(columns, (str, int)):
        columns = [columns]
    elif isinstance(columns, tuple):
        columns = list(columns)
    return columns


class ReadParquet(PartitionsFiltered, BlockwiseIO):
    """Read a parquet dataset"""

    _parameters = [
        "path",
        "columns",
        "filters",
        "categories",
        "index",
        "storage_options",
        "calculate_divisions",
        "ignore_metadata_file",
        "metadata_task_size",
        "split_row_groups",
        "blocksize",
        "aggregate_files",
        "parquet_file_extension",
        "filesystem",
        "kwargs",
        "_partitions",
        "_series",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "categories": None,
        "index": None,
        "storage_options": None,
        "calculate_divisions": False,
        "ignore_metadata_file": False,
        "metadata_task_size": None,
        "split_row_groups": "infer",
        "blocksize": "default",
        "aggregate_files": None,
        "parquet_file_extension": (".parq", ".parquet", ".pq"),
        "filesystem": "fsspec",
        "kwargs": None,
        "_partitions": None,
        "_series": False,
    }

    @cached_property
    def _ddf(self):
        # Leverage dd.read_parquet for now to
        # simplify development
        import dask.dataframe as dd

        return dd.read_parquet(
            self.path,
            columns=self.operand("columns"),
            filters=self.filters,
            categories=self.categories,
            index=self.operand("index"),
            storage_options=self.storage_options,
            calculate_divisions=self.calculate_divisions,
            ignore_metadata_file=self.ignore_metadata_file,
            metadata_task_size=self.metadata_task_size,
            split_row_groups=self.split_row_groups,
            blocksize=self.blocksize,
            aggregate_files=self.aggregate_files,
            parquet_file_extension=self.parquet_file_extension,
            filesystem=self.filesystem,
            **self.kwargs,
        )

    @property
    def _meta(self):
        meta = self._ddf._meta
        if self._series:
            column = _list_columns(self.operand("columns"))[0]
            return meta[column]
        return meta

    def _divisions(self):
        return self._ddf.divisions

    @cached_property
    def _tasks(self):
        return list(self._ddf.dask.to_dict().values())

    def _filtered_task(self, index: int):
        if self._series:
            return (operator.getitem, self._tasks[index], self.columns[0])
        return self._tasks[index]

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            operands = list(self.operands)
            operands[self._parameters.index("columns")] = _list_columns(
                parent.operand("columns")
            )
            if isinstance(parent.operand("columns"), (str, int)):
                operands[self._parameters.index("_series")] = True
            return ReadParquet(*operands)

        if isinstance(parent, Filter) and isinstance(
            parent.predicate, (LE, GE, LT, GT, EQ, NE)
        ):
            kwargs = dict(zip(self._parameters, self.operands))
            if (
                isinstance(parent.predicate.left, ReadParquet)
                and parent.predicate.left.path == self.path
                and not isinstance(parent.predicate.right, Expr)
            ):
                op = parent.predicate._operator_repr
                column = parent.predicate.left.columns[0]
                value = parent.predicate.right
                kwargs["filters"] = (kwargs["filters"] or tuple()) + (
                    (column, op, value),
                )
                return ReadParquet(**kwargs)
            if (
                isinstance(parent.predicate.right, ReadParquet)
                and parent.predicate.right.path == self.path
                and not isinstance(parent.predicate.left, Expr)
            ):
                # Simple dict to make sure field comes first in filter
                flip = {LE: GE, LT: GT, GE: LE, GT: LT}
                op = parent.predicate
                op = flip.get(op, op)._operator_repr
                column = parent.predicate.right.columns[0]
                value = parent.predicate.left
                kwargs["filters"] = (kwargs["filters"] or tuple()) + (
                    (column, op, value),
                )
                return ReadParquet(**kwargs)
