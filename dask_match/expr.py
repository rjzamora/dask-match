import functools
import numbers
import operator
import os
from collections import defaultdict
from collections.abc import Iterator

import dask
import pandas as pd
import toolz
from dask.base import normalize_token, tokenize
from dask.core import ishashable
from dask.dataframe import methods
from dask.dataframe.core import (
    _get_divisions_map_partitions,
    _get_meta_map_partitions,
    apply_and_enforce,
    is_dataframe_like,
)
from dask.utils import M, apply, funcname, import_required
from matchpy import Arity, Operation, ReplacementRule, replace_all
from matchpy.expressions.expressions import _OperationMeta

replacement_rules = []

no_default = "__no_default__"


class _ExprMeta(_OperationMeta):
    """Metaclass to determine Operation behavior

    Matchpy overrides `__call__` so that `__init__` doesn't behave as expected.
    This is gross, but has some logic behind it.  We need to enforce that
    expressions can be easily replayed.  Eliminating `__init__` is one way to
    do this.  It forces us to compute things lazily rather than at
    initialization.

    We motidify Matchpy's implementation so that we can handle keywords and
    default values more cleanly.

    We also collect replacement rules here.
    """

    seen = set()

    def __call__(cls, *args, variable_name=None, **kwargs):
        # Collect optimization rules for new classes
        if cls not in _ExprMeta.seen:
            _ExprMeta.seen.add(cls)
            for rule in cls._replacement_rules():
                replacement_rules.append(rule)

        # Grab keywords and manage default values
        operands = list(args)
        for parameter in cls._parameters[len(operands) :]:
            try:
                operands.append(kwargs.pop(parameter))
            except KeyError:
                operands.append(cls._defaults[parameter])
        assert not kwargs

        # Defer up to matchpy
        return super().__call__(*operands, variable_name=None)


_defer_to_matchpy = False


class Expr(Operation, metaclass=_ExprMeta):
    """Primary class for all Expressions

    This mostly includes Dask protocols and various Pandas-like method
    definitions to make us look more like a DataFrame.
    """

    commutative = False
    associative = False
    _parameters = []
    _defaults = {}

    @functools.cached_property
    def ndim(self):
        meta = self._meta
        try:
            return meta.ndim
        except AttributeError:
            return 0

    @classmethod
    def _replacement_rules(cls) -> Iterator[ReplacementRule]:
        """Rules associated to this class that are useful for optimization

        See also:
            optimize
            _ExprMeta
        """
        yield from []

    def __str__(self):
        s = ", ".join(
            str(param) + "=" + str(operand)
            for param, operand in zip(self._parameters, self.operands)
            if operand != self._defaults.get(param)
        )
        return f"{type(self).__name__}({s})"

    def __repr__(self):
        return str(self)

    def _tree_repr_lines(self, indent=0):
        header = funcname(type(self)) + ":"
        lines = []
        for i, op in enumerate(self.operands):
            if isinstance(op, Expr):
                lines.extend(op._tree_repr_lines(2))
            else:
                try:
                    param = self._parameters[i]
                    default = self._defaults[param]
                except (IndexError, KeyError):
                    param = self._parameters[i] if i < len(self._parameters) else ""
                    default = "--no-default--"

                if isinstance(op, pd.core.base.PandasObject):
                    op = "<pandas>"

                elif repr(op) != repr(default):
                    if param:
                        header += f" {param}={repr(op)}"
                    else:
                        header += repr(op)
        lines = [header] + lines
        lines = [" " * indent + line for line in lines]

        return lines

    def tree_repr(self):
        return os.linesep.join(self._tree_repr_lines())

    def pprint(self):
        for line in self._tree_repr_lines():
            print(line)

    def __hash__(self):
        return hash(self._name)

    def __reduce__(self):
        return type(self), tuple(self.operands)

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as err:
            # Allow operands to be accessed as attributes
            # as long as the keys are not already reserved
            # by existing methods/properties
            _parameters = type(self)._parameters
            if key in _parameters:
                idx = _parameters.index(key)
                return self.operands[idx]
            if is_dataframe_like(self._meta) and key in self._meta.columns:
                return self[key]
            raise err

    def operand(self, key):
        # Access an operand unambiguously
        # (e.g. if the key is reserved by a method/property)
        return self.operands[type(self)._parameters.index(key)]

    def dependencies(self):
        # Dependencies are `Expr` operands only
        return [operand for operand in self.operands if isinstance(operand, Expr)]

    def _task(self, index: int):
        """The task for the i'th partition

        Parameters
        ----------
        index:
            The index of the partition of this dataframe

        Examples
        --------
        >>> class Add(Expr):
        ...     def _task(self, i):
        ...         return (operator.add, (self.left._name, i), (self.right._name, i))

        Returns
        -------
        task:
            The Dask task to compute this partition

        See Also
        --------
        Expr._layer
        """
        raise NotImplementedError(
            "Expressions should define either _layer (full dictionary) or _task"
            " (single task).  This expression type defines neither"
        )

    def _layer(self) -> dict:
        """The graph layer added by this expression

        Examples
        --------
        >>> class Add(Expr):
        ...     def _layer(self):
        ...         return {
        ...             (self._name, i): (operator.add, (self.left._name, i), (self.right._name, i))
        ...             for i in range(self.npartitions)
        ...         }

        Returns
        -------
        layer: dict
            The Dask task graph added by this expression

        See Also
        --------
        Expr._task
        Expr.__dask_graph__
        """

        return {(self._name, i): self._task(i) for i in range(self.npartitions)}

    def simplify(self):
        """Simplify expression

        This leverages the ``._simplify_down`` method defined on each class

        Returns
        -------
        expr:
            output expression
        changed:
            whether or not any change occured
        """
        expr = self

        while True:
            # Simplify this node
            out = expr._simplify_down()
            if out is None:
                out = expr
            if not isinstance(out, Expr):
                return out
            if out._name != expr._name:
                expr = out
                continue

            # Allow children to simplify their parents
            _continue = False
            for child in expr.dependencies():
                out = child._simplify_up(expr)
                if out is None:
                    out = expr
                if not isinstance(out, Expr):
                    return out
                if out is not expr and out._name != expr._name:
                    expr = out
                    _continue = True
                    break

            if _continue:
                continue

            # Simplify all of the children
            new_operands = []
            changed = False
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    new = operand.simplify()
                    if new._name != operand._name:
                        changed = True
                else:
                    new = operand
                new_operands.append(new)

            if changed:
                expr = type(expr)(*new_operands)
                continue
            else:
                break

        return expr

    def _simplify_down(self):
        return

    def _simplify_up(self, parent):
        return

    def optimize(self, **kwargs):
        return optimize(self, **kwargs)

    @property
    def index(self):
        return Index(self)

    @property
    def size(self):
        return Size(self)

    def __getitem__(self, other):
        if isinstance(other, Expr):
            return Filter(self, other)  # df[df.x > 1]
        else:
            return Projection(self, other)  # df[["a", "b", "c"]]

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __lt__(self, other):
        return LT(self, other)

    def __rlt__(self, other):
        return LT(other, self)

    def __gt__(self, other):
        return GT(self, other)

    def __rgt__(self, other):
        return GT(other, self)

    def __le__(self, other):
        return LE(self, other)

    def __rle__(self, other):
        return LE(other, self)

    def __ge__(self, other):
        return GE(self, other)

    def __rge__(self, other):
        return GE(other, self)

    def __eq__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__eq__(self, other)
        else:
            return EQ(other, self)

    def __ne__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__ne__(self, other)
        else:
            return NE(other, self)

    def sum(self, skipna=True, numeric_only=None, min_count=0):
        return Sum(self, skipna, numeric_only, min_count)

    def mean(self, skipna=True, numeric_only=None, min_count=0):
        return Mean(self, skipna=skipna, numeric_only=numeric_only)

    def max(self, skipna=True, numeric_only=None, min_count=0):
        return Max(self, skipna, numeric_only, min_count)

    def mode(self, dropna=True):
        return Mode(self, dropna=dropna)

    def min(self, skipna=True, numeric_only=None, min_count=0):
        return Min(self, skipna, numeric_only, min_count)

    def count(self, numeric_only=None):
        return Count(self, numeric_only)

    def astype(self, dtypes):
        return AsType(self, dtypes)

    def apply(self, function, *args, **kwargs):
        return Apply(self, function, args, kwargs)

    @functools.cached_property
    def divisions(self):
        return tuple(self._divisions())

    def _divisions(self):
        raise NotImplementedError()

    @property
    def known_divisions(self):
        """Whether divisions are already known"""
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self):
        if "npartitions" in self._parameters:
            idx = self._parameters.index("npartitions")
            return self.operands[idx]
        else:
            return len(self.divisions) - 1

    @functools.cached_property
    def _name(self):
        return funcname(type(self)).lower() + "-" + tokenize(*self.operands)

    @property
    def columns(self):
        return self._meta.columns

    @property
    def dtypes(self):
        return self._meta.dtypes

    @property
    def _meta(self):
        raise NotImplementedError()

    def __dask_graph__(self):
        """Traverse expression tree, collect layers"""
        stack = [self]
        seen = set()
        layers = []
        while stack:
            expr = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            layers.append(expr._layer())
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)

        return toolz.merge(layers)

    def __dask_keys__(self):
        return [(self._name, i) for i in range(self.npartitions)]

    def substitute(self, substitutions: dict) -> "Expr":
        """Substitute specific `Expr` instances within `self`

        Parameters
        ----------
        substitutions:
            mapping old terms to new terms. Note that using
            non-`Expr` keys may produce unexpected results,
            and substituting boolean values is not allowed.

        Examples
        --------
        >>> (df + 10).substitute({10: 20})
        df + 20
        """
        if not substitutions:
            return self

        if self in substitutions:
            return substitutions[self]

        new = []
        update = False
        for operand in self.operands:
            if (
                not isinstance(operand, bool)
                and ishashable(operand)
                and operand in substitutions
            ):
                new.append(substitutions[operand])
                update = True
            elif isinstance(operand, Expr):
                val = operand.substitute(substitutions)
                if operand._name != val._name:
                    update = True
                new.append(val)
            else:
                new.append(operand)

        if update:  # Only recreate if something changed
            return type(self)(*new)
        return self


    def _to_graphviz(
        self,
        data_attributes=None,
        function_attributes=None,
        rankdir="BT",
        graph_attr=None,
        node_attr=None,
        edge_attr=None,
        **kwargs,
    ):
        from dask.dot import label, name

        graphviz = import_required(
            "graphviz",
            "Drawing dask graphs with the graphviz visualization engine requires the `graphviz` "
            "python library and the `graphviz` system library.\n\n"
            "Please either conda or pip install as follows:\n\n"
            "  conda install python-graphviz     # either conda install\n"
            "  python -m pip install graphviz    # or pip install and follow installation instructions",
        )

        data_attributes = data_attributes or {}
        function_attributes = function_attributes or {}
        graph_attr = graph_attr or {}
        node_attr = node_attr or {}
        edge_attr = edge_attr or {}

        graph_attr["rankdir"] = rankdir
        node_attr["shape"] = "box"
        node_attr["fontname"] = "helvetica"

        graph_attr.update(kwargs)
        g = graphviz.Digraph(
            graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr
        )

        # n_tasks = {}
        # for layer in hg.dependencies:
        #     n_tasks[layer] = len(hg.layers[layer])

        # min_tasks = min(n_tasks.values())
        # max_tasks = max(n_tasks.values())

        cache = {}

        color = kwargs.get("color")
        # if color == "layer_type":
        #     layer_colors = {
        #         "DataFrameIOLayer": ["#CCC7F9", False],  # purple
        #         "ShuffleLayer": ["#F9CCC7", False],  # rose
        #         "SimpleShuffleLayer": ["#F9CCC7", False],  # rose
        #         "ArrayOverlayLayer": ["#FFD9F2", False],  # pink
        #         "BroadcastJoinLayer": ["#D9F2FF", False],  # blue
        #         "Blockwise": ["#D9FFE6", False],  # green
        #         "BlockwiseLayer": ["#D9FFE6", False],  # green
        #         "MaterializedLayer": ["#DBDEE5", False],  # gray
        #     }


        exprs = {}
        dependencies = {}
        stack = [self]
        seen = set()
        while stack:
            expr = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            exprs[expr._name] = expr
            dependencies[expr._name] = set([dep._name for dep in self.dependencies()])
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)



        for expr in dependencies:
            expr_name = name(expr)
            attrs = data_attributes.get(expr, {})

            node_label = label(expr, cache=cache)
            node_size = (
                20
                #if max_tasks == min_tasks
                #else int(20 + ((n_tasks[layer] - min_tasks) / (max_tasks - min_tasks)) * 20)
            )

            # expr_type = str(type(exprs[expr]).__name__)
            # node_tooltips = (
            #     f"A {expr_type.replace('Layer', '')} Layer with {n_tasks[layer]} Tasks.\n"
            # )

            attrs.setdefault("label", str(node_label))
            attrs.setdefault("fontsize", str(node_size))
            #attrs.setdefault("tooltip", str(node_tooltips))

            g.node(expr_name, **attrs)

        for expr, deps in dependencies.items():
            expr_name = name(expr)
            for dep in deps:
                dep_name = name(dep)
                g.edge(dep_name, expr_name)

        return g


    def visualize(self, filename="dask-hlg.svg", format=None, **kwargs):
        """
        Visualize this dask high level graph.
        Requires ``graphviz`` to be installed.
        Parameters
        ----------
        filename : str or None, optional
            The name of the file to write to disk. If the provided `filename`
            doesn't include an extension, '.png' will be used by default.
            If `filename` is None, no file will be written, and the graph is
            rendered in the Jupyter notebook only.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file. Default is 'svg'.
        color : {None, 'layer_type'}, optional (default: None)
            Options to color nodes.
            - None, no colors.
            - layer_type, color nodes based on the layer type.
        **kwargs
           Additional keyword arguments to forward to ``to_graphviz``.
        Examples
        --------
        >>> x.dask.visualize(filename='dask.svg')  # doctest: +SKIP
        >>> x.dask.visualize(filename='dask.svg', color='layer_type')  # doctest: +SKIP
        Returns
        -------
        result : IPython.diplay.Image, IPython.display.SVG, or None
            See dask.dot.dot_graph for more information.
        See Also
        --------
        dask.dot.dot_graph
        dask.base.visualize # low level variant
        """

        from dask.dot import graphviz_to_file

        g = self._to_graphviz(**kwargs)
        graphviz_to_file(g, filename, format)
        return g


class Blockwise(Expr):
    """Super-class for block-wise operations

    This is fairly generic, and includes definitions for `_meta`, `divisions`,
    `_layer` that are often (but not always) correct.  Mostly this helps us
    avoid duplication in the future.

    Note that `Fused` expressions rely on every `Blockwise`
    expression defining a proper `_task` method.
    """

    operation = None

    @functools.cached_property
    def _meta(self):
        return self.operation(
            *[arg._meta if isinstance(arg, Expr) else arg for arg in self.operands]
        )

    @property
    def _kwargs(self):
        return {}

    def _broadcast_dep(self, dep: Expr):
        # Checks if a dependency should be broadcasted to
        # all partitions of this `Blockwise` operation
        return dep.npartitions == 1 and dep.ndim < self.ndim

    def _divisions(self):
        # This is an issue.  In normal Dask we re-divide everything in a step
        # which combines divisions and graph.
        # We either have to create a new Align layer (ok) or combine divisions
        # and graph into a single operation.
        dependencies = self.dependencies()
        for arg in dependencies:
            if not self._broadcast_dep(arg):
                assert arg.divisions == dependencies[0].divisions
        return dependencies[0].divisions

    @functools.cached_property
    def _name(self):
        if self.operation:
            head = funcname(self.operation)
        else:
            head = funcname(type(self)).lower()
        return head + "-" + tokenize(*self.operands)

    def _blockwise_arg(self, arg, i):
        """Return a Blockwise-task argument"""
        if isinstance(arg, Expr):
            # Make key for Expr-based argument
            if self._broadcast_dep(arg):
                return (arg._name, 0)
            else:
                return (arg._name, i)

        else:
            return arg

    def _task(self, index: int):
        """Produce the task for a specific partition

        Parameters
        ----------
        index:
            Partition index for this task.

        Returns
        -------
        task: tuple
        """
        args = tuple(self._blockwise_arg(op, index) for op in self.operands)
        if self._kwargs:
            return (apply, self.operation, args, self._kwargs)
        else:
            return (self.operation,) + args


class MapPartitions(Blockwise):
    _parameters = [
        "frame",
        "func",
        "meta",
        "enforce_metadata",
        "transform_divisions",
        "kwargs",
    ]

    def _broadcast_dep(self, dep: Expr):
        # Always broadcast single-partition dependencies in MapPartitions
        return dep.npartitions == 1

    @property
    def args(self):
        return [self.frame] + self.operands[len(self._parameters) :]

    @functools.cached_property
    def _meta(self):
        meta = self.operand("meta")
        args = [arg._meta if isinstance(arg, Expr) else arg for arg in self.args]
        return _get_meta_map_partitions(args, [], self.func, self.kwargs, meta, None)

    def _divisions(self):
        dfs = [arg for arg in self.args if isinstance(arg, Expr)]
        return _get_divisions_map_partitions(
            False,  # Partitions must already be "aligned"
            self.transform_divisions,
            dfs,
            self.func,
            self.args,
            self.kwargs,
        )

    def _task(self, index: int):
        args = [self._blockwise_arg(op, index) for op in self.args]
        if self.enforce_metadata:
            kwargs = self.kwargs.copy()
            kwargs.update(
                {
                    "_func": self.func,
                    "_meta": self._meta,
                }
            )
            return (apply, apply_and_enforce, args, kwargs)
        else:
            return (
                apply,
                self.func,
                args,
                self.kwargs,
            )


class Elemwise(Blockwise):
    """
    This doesn't really do anything, but we anticipate that future
    optimizations, like `len` will care about which operations preserve length
    """

    pass


class AsType(Elemwise):
    """A good example of writing a trivial blockwise operation"""

    _parameters = ["frame", "dtypes"]
    operation = M.astype


class Apply(Elemwise):
    """A good example of writing a less-trivial blockwise operation"""

    _parameters = ["frame", "function", "args", "kwargs"]
    _defaults = {"args": (), "kwargs": {}}
    operation = M.apply

    @property
    def _meta(self):
        return self.frame._meta.apply(self.function, *self.args, **self.kwargs)

    def _task(self, index: int):
        return (
            apply,
            M.apply,
            [
                (self.frame._name, index),
                self.function,
            ]
            + list(self.args),
            self.kwargs,
        )


class Assign(Elemwise):
    """Column Assignment"""

    _parameters = ["frame", "key", "value"]
    operation = staticmethod(methods.assign)


class Filter(Blockwise):
    _parameters = ["frame", "predicate"]
    operation = operator.getitem

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return self.frame[parent.operand("columns")][self.predicate]


class Projection(Elemwise):
    """Column Selection"""

    _parameters = ["frame", "columns"]
    operation = operator.getitem

    @property
    def columns(self):
        if isinstance(self.operand("columns"), list):
            return pd.Index(self.operand("columns"))
        else:
            return self.operand("columns")

    def __str__(self):
        base = str(self.frame)
        if " " in base:
            base = "(" + base + ")"
        return f"{base}[{repr(self.columns)}]"

    def _simplify_down(self):
        if isinstance(self.frame, Projection):
            # df[a][b]
            a = self.frame.operand("columns")
            b = self.operand("columns")

            if not isinstance(a, list):
                assert a == b
            elif isinstance(b, list):
                assert all(bb in a for bb in b)
            else:
                assert b in a

            return self.frame.frame[b]


class Index(Elemwise):
    """Column Selection"""

    _parameters = ["frame"]
    operation = getattr

    @property
    def _meta(self):
        return self.frame._meta.index

    def _task(self, index: int):
        return (
            getattr,
            (self.frame._name, index),
            "index",
        )


class Head(Expr):
    """Take the first `n` rows of the first partition"""

    _parameters = ["frame", "n"]
    _defaults = {"n": 5}

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        return self.frame.divisions[:2]

    def _task(self, index: int):
        raise NotImplementedError()

    def _simplify_down(self):
        if isinstance(self.frame, Elemwise):
            operands = [
                Head(op, self.n) if isinstance(op, Expr) else op
                for op in self.frame.operands
            ]
            return type(self.frame)(*operands)
        if not isinstance(self, BlockwiseHead):
            # Lower to Blockwise
            return BlockwiseHead(Partitions(self.frame, [0]), self.n)
        if isinstance(self.frame, Head):
            return Head(self.frame.frame, min(self.n, self.frame.n))


class BlockwiseHead(Head, Blockwise):
    """Take the first `n` rows of every partition

    Typically used after `Partition(..., [0])` to take
    the first `n` rows of an entire collection.
    """

    def _divisions(self):
        return self.frame.divisions

    def _task(self, index: int):
        return (M.head, (self.frame._name, index), self.n)


class Binop(Elemwise):
    _parameters = ["left", "right"]
    arity = Arity.binary

    def __str__(self):
        return f"{self.left} {self._operator_repr} {self.right}"

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            if isinstance(self.left, Expr):
                left = self.left[
                    parent.operand("columns")
                ]  # TODO: filter just the correct columns
            else:
                left = self.left
            if isinstance(self.right, Expr):
                right = self.right[parent.operand("columns")]
            else:
                right = self.right
            return type(self)(left, right)


class Add(Binop):
    operation = operator.add
    _operator_repr = "+"

    def _simplify_down(self):
        if (
            isinstance(self.left, Expr)
            and isinstance(self.right, Expr)
            and self.left._name == self.right._name
        ):
            return 2 * self.left


class Sub(Binop):
    operation = operator.sub
    _operator_repr = "-"


class Mul(Binop):
    operation = operator.mul
    _operator_repr = "*"

    def _simplify_down(self):
        if (
            isinstance(self.right, Mul)
            and isinstance(self.left, numbers.Number)
            and isinstance(self.right.left, numbers.Number)
        ):
            return (self.left * self.right.left) * self.right.right


class Div(Binop):
    operation = operator.truediv
    _operator_repr = "/"


class LT(Binop):
    operation = operator.lt
    _operator_repr = "<"


class LE(Binop):
    operation = operator.le
    _operator_repr = "<="


class GT(Binop):
    operation = operator.gt
    _operator_repr = ">"


class GE(Binop):
    operation = operator.ge
    _operator_repr = ">="


class EQ(Binop):
    operation = operator.eq
    _operator_repr = "=="


class NE(Binop):
    operation = operator.ne
    _operator_repr = "!="


class Partitions(Expr):
    """Select one or more partitions"""

    _parameters = ["frame", "partitions"]

    @property
    def _meta(self):
        return self.frame._meta

    def _divisions(self):
        divisions = []
        for part in self.partitions:
            divisions.append(self.frame.divisions[part])
        divisions.append(self.frame.divisions[part + 1])
        return tuple(divisions)

    def _task(self, index: int):
        return (self.frame._name, self.partitions[index])

    def _simplify_down(self):
        if isinstance(self.frame, Blockwise) and not isinstance(
            self.frame, BlockwiseIO
        ):
            operands = [
                Partitions(op, self.partitions)
                if (isinstance(op, Expr) and not self.frame._broadcast_dep(op))
                else op
                for op in self.frame.operands
            ]
            return type(self.frame)(*operands)
        elif isinstance(self.frame, PartitionsFiltered):
            if self.frame._partitions:
                partitions = [self.frame._partitions[p] for p in self.partitions]
            else:
                partitions = self.partitions
            # We assume that expressions defining a special "_partitions"
            # parameter can internally capture the same logic as `Partitions`
            operands = [
                partitions if self.frame._parameters[i] == "_partitions" else op
                for i, op in enumerate(self.frame.operands)
            ]
            return type(self.frame)(*operands)


class PartitionsFiltered(Expr):
    """Mixin class for partition filtering

    A `PartitionsFiltered` subclass must define a
    `_partitions` parameter. When `_partitions` is
    defined, the following expresssions must produce
    the same output for `cls: PartitionsFiltered`:
      - `cls(expr: Expr, ..., _partitions)`
      - `Partitions(cls(expr: Expr, ...), _partitions)`

    In order to leverage the default `Expr._layer`
    method, subclasses should define `_filtered_task`
    instead of `_task`.
    """

    @property
    def _filtered(self) -> bool:
        """Whether or not output partitions have been filtered"""
        return self.operand("_partitions") is not None

    @property
    def _partitions(self) -> list | tuple | range:
        """Selected partition indices"""
        if self._filtered:
            return self.operand("_partitions")
        else:
            return range(self.npartitions)

    @functools.cached_property
    def divisions(self):
        # Common case: Use self._divisions()
        full_divisions = super().divisions
        if not self._filtered:
            return full_divisions

        # Specific case: Specific partitions were selected
        new_divisions = []
        for part in self._partitions:
            new_divisions.append(full_divisions[part])
        new_divisions.append(full_divisions[part + 1])
        return tuple(new_divisions)

    @property
    def npartitions(self):
        if self._filtered:
            return len(self._partitions)
        return super().npartitions

    def _task(self, index: int):
        return self._filtered_task(self._partitions[index])

    def _filtered_task(self, index: int):
        raise NotImplementedError()


@normalize_token.register(Expr)
def normalize_expression(expr):
    return expr._name


def optimize(expr: Expr, fuse: bool = True) -> Expr:
    """High level query optimization

    This leverages three optimization passes:

    1.  Class based simplification using the ``_simplify`` function and methods
    2.  Replacement rules with matchpy
    3.  Blockwise fusion

    Parameters
    ----------
    expr:
        Input expression to optimize
    fuse:
        whether or not to turn on blockwise fusion

    See Also
    --------
    simplify
    matchpy
    optimize_blockwise_fusion
    """
    expr = expr.simplify()
    expr = optimize_matchpy(expr)

    if fuse:
        expr = optimize_blockwise_fusion(expr)

    return expr


def optimize_matchpy(expr: Expr) -> Expr:
    last = None
    global _defer_to_matchpy

    _defer_to_matchpy = True  # take over ==/!= when optimizing
    try:
        while last is None or expr._name != last._name:
            last = expr
            expr = replace_all(expr, replacement_rules)
    finally:
        _defer_to_matchpy = False

    return expr


## Utilites for Expr fusion


def optimize_blockwise_fusion(expr):
    """Traverse the expression graph and apply fusion"""

    def _fusion_pass(expr):
        # Full pass to find global dependencies
        seen = set()
        stack = [expr]
        dependents = defaultdict(set)
        dependencies = {}
        while stack:
            next = stack.pop()

            if next._name in seen:
                continue
            seen.add(next._name)

            if isinstance(next, Blockwise):
                dependencies[next] = set()
                if next not in dependents:
                    dependents[next] = set()

            for operand in next.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)
                    if isinstance(operand, Blockwise):
                        if next in dependencies:
                            dependencies[next].add(operand)
                        dependents[operand].add(next)

        # Traverse each "root" until we find a fusable sub-group.
        # Here we use root to refer to a Blockwise Expr node that
        # has no Blockwise dependents
        roots = [
            k
            for k, v in dependents.items()
            if v == set() or all(not isinstance(_expr, Blockwise) for _expr in v)
        ]
        while roots:
            root = roots.pop()
            seen = set()
            stack = [root]
            group = []
            while stack:
                next = stack.pop()

                if next._name in seen:
                    continue
                seen.add(next._name)

                group.append(next)
                for dep in dependencies[next]:
                    if not (dependents[dep] - set(stack) - set(group)):
                        # All of deps dependents are contained
                        # in the local group (or the local stack
                        # of expr nodes that we know we will be
                        # adding to the local group)
                        stack.append(dep)
                    elif dep not in roots and dependencies[dep]:
                        # Couldn't fuse dep, but we may be able to
                        # use it as a new root on the next pass
                        roots.append(dep)

            # Replace fusable sub-group
            if len(group) > 1:
                group_deps = []
                local_names = [_expr._name for _expr in group]
                for _expr in group:
                    group_deps += [
                        operand
                        for operand in _expr.dependencies()
                        if operand._name not in local_names
                    ]
                to_replace = {group[0]: Fused(group, *group_deps)}
                return expr.substitute(to_replace), not roots

        # Return original expr if no fusable sub-groups were found
        return expr, True

    while True:
        original_name = expr._name
        expr, done = _fusion_pass(expr)
        if done or expr._name == original_name:
            break

    return expr


class Fused(Blockwise):
    """Fused ``Blockwise`` expression

    A ``Fused`` corresponds to the fusion of multiple
    ``Blockwise`` expressions into a single ``Expr`` object.
    Before graph-materialization time, the behavior of this
    object should be identical to that of the first element
    of ``Fused.exprs`` (i.e. the top-most expression in
    the fused group).

    Parameters
    ----------
    exprs : List[Expr]
        Group of original ``Expr`` objects being fused together.
    *dependencies:
        List of external ``Expr`` dependencies. External-``Expr``
        dependencies correspond to any ``Expr`` operand that is
        not already included in ``exprs``. Note that these
        dependencies should be defined in the order of the ``Expr``
        objects that require them (in ``exprs``). These
        dependencies do not include literal operands, because those
        arguments should already be captured in the fused subgraphs.
    """

    _parameters = ["exprs"]

    @functools.cached_property
    def _meta(self):
        return self.exprs[0]._meta

    def __str__(self):
        names = [expr._name.split("-")[0] for expr in self.exprs]
        if len(names) > 3:
            names = [names[0], f"{len(names) - 2}", names[-1]]
        descr = "-".join(names)
        return f"Fused-{descr}"

    @functools.cached_property
    def _name(self):
        return f"{str(self)}-{tokenize(self.exprs)}"

    def _divisions(self):
        return self.exprs[0]._divisions()

    def _task(self, index):
        graph = {self._name: (self.exprs[0]._name, index)}
        for _expr in self.exprs:
            if isinstance(_expr, Fused):
                (_, subgraph, name) = _expr._task(index)
                graph.update(subgraph)
                graph[(name, index)] = name
            else:
                graph[(_expr._name, index)] = _expr._task(index)

        for i, dep in enumerate(self.dependencies()):
            graph[self._blockwise_arg(dep, index)] = "_" + str(i)

        return (
            Fused._execute_task,
            graph,
            self._name,
        ) + tuple(self._blockwise_arg(dep, index) for dep in self.dependencies())

    @staticmethod
    def _execute_task(graph, name, *deps):
        for i, dep in enumerate(deps):
            graph["_" + str(i)] = dep
        return dask.core.get(graph, name)


from dask_match.io import BlockwiseIO
from dask_match.reductions import Count, Max, Mean, Min, Mode, Size, Sum
