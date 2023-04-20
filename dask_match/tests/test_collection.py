import operator
import pickle

import pandas as pd
import pytest

import dask
from dask.dataframe.utils import assert_eq
from dask.utils import M

from dask_match import expr, from_pandas, optimize


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": range(100)})
    pdf["y"] = pdf.x * 10.0
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=10)


def test_del(pdf, df):
    pdf = pdf.copy()

    # Check __delitem__
    del pdf["x"]
    del df["x"]
    assert_eq(pdf, df)


def test_setitem(pdf, df):
    pdf = pdf.copy()

    df["z"] = df.x + df.y

    assert "z" in df.columns
    assert_eq(df, df)


def test_meta_divisions_name():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    df = 2 * from_pandas(a, npartitions=2)
    assert list(df.columns) == list(a.columns)
    assert df.npartitions == 2

    assert df.x.sum()._meta == 0
    assert df.x.sum().npartitions == 1

    assert "mul" in df._name
    assert "sum" in df.sum()._name


def test_meta_blockwise():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    b = pd.DataFrame({"z": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})

    aa = from_pandas(a, npartitions=2)
    bb = from_pandas(b, npartitions=2)

    cc = 2 * aa - 3 * bb
    assert set(cc.columns) == {"x", "y", "z"}


def test_dask(pdf, df):
    assert (df.x + df.y).npartitions == 10
    z = (df.x + df.y).sum()

    assert assert_eq(z, (pdf.x + pdf.y).sum())


@pytest.mark.parametrize(
    "func",
    [
        M.max,
        M.min,
        M.sum,
        M.count,
        M.mean,
        pytest.param(
            lambda df: df.size,
            marks=pytest.mark.skip(reason="scalars don't work yet"),
        ),
    ],
)
def test_reductions(func, pdf, df):
    assert_eq(func(df), func(pdf))
    assert_eq(func(df.x), func(pdf.x))


def test_mode():
    pdf = pd.DataFrame({"x": [1, 2, 3, 1, 2]})
    df = from_pandas(pdf, npartitions=3)

    assert_eq(df.x.mode(), pdf.x.mode(), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.x > 10,
        lambda df: df.x + 20 > df.y,
        lambda df: 10 < df.x,
        lambda df: 10 <= df.x,
        lambda df: 10 == df.x,
        lambda df: df.x < df.y,
        lambda df: df.x > df.y,
        lambda df: df.x == df.y,
        lambda df: df.x != df.y,
    ],
)
def test_conditionals(func, pdf, df):
    assert_eq(func(pdf), func(df), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.astype(int),
        lambda df: df.apply(lambda row, x, y=10: row * x + y, x=2),
        lambda df: df[df.x > 5],
        lambda df: df.assign(a=df.x + df.y, b=df.x - df.y),
    ],
)
def test_blockwise(func, pdf, df):
    assert_eq(func(pdf), func(df))


def test_repr(df):
    assert "+ 1" in str(df + 1)
    assert "+ 1" in repr(df + 1)

    s = (df["x"] + 1).sum(skipna=False).expr
    assert '["x"]' in s or "['x']" in s
    assert "+ 1" in s
    assert "sum(skipna=False)" in s


def test_columns_traverse_filters(pdf, df):
    result = optimize(df[df.x > 5].y, fuse=False)
    expected = df.y[df.x > 5]

    assert str(result) == str(expected)


def test_broadcast(pdf, df):
    assert_eq(
        df + df.sum(),
        pdf + pdf.sum(),
    )
    assert_eq(
        df.x + df.x.sum(),
        pdf.x + pdf.x.sum(),
    )


def test_persist(pdf, df):
    a = df + 2
    b = a.persist()

    assert_eq(a, b)
    assert len(a.__dask_graph__()) > len(b.__dask_graph__())

    assert len(b.__dask_graph__()) == b.npartitions

    assert_eq(b.y.sum(), (pdf + 2).y.sum())


def test_index(pdf, df):
    assert_eq(df.index, pdf.index)
    assert_eq(df.x.index, pdf.x.index)


def test_head(pdf, df):
    assert_eq(df.head(compute=False), pdf.head())
    assert_eq(df.head(compute=False, n=7), pdf.head(n=7))

    assert df.head(compute=False).npartitions == 1


def test_head_down(df):
    result = (df.x + df.y + 1).head(compute=False)
    optimized = optimize(result)

    assert_eq(result, optimized)

    assert not isinstance(optimized.expr, expr.Head)


def test_projection_stacking(df):
    result = df[["x", "y"]]["x"]
    optimized = optimize(result, fuse=False)
    expected = df["x"]

    assert optimized._name == expected._name


def test_substitute(df):
    pdf = pd.DataFrame(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
        }
    )
    df = from_pandas(pdf, npartitions=3)
    df = df.expr

    result = (df + 1).substitute({1: 2})
    expected = df + 2
    assert result._name == expected._name

    result = df["a"].substitute({df["a"]: df["b"]})
    expected = df["b"]
    assert result._name == expected._name

    result = (df["a"] - df["b"]).substitute({df["b"]: df["c"]})
    expected = df["a"] - df["c"]
    assert result._name == expected._name

    result = df["a"].substitute({3: 4})
    expected = from_pandas(pdf, npartitions=4).a
    assert result._name == expected._name

    result = (df["a"].sum() + 5).substitute({df["a"]: df["b"], 5: 6})
    expected = df["b"].sum() + 6
    assert result._name == expected._name


def test_from_pandas(pdf):
    df = from_pandas(pdf, npartitions=3)
    assert df.npartitions == 3
    assert "pandas" in df._name


def test_copy(pdf, df):
    original = df.copy()
    columns = tuple(original.columns)

    df["z"] = df.x + df.y

    assert tuple(original.columns) == columns
    assert "z" not in original.columns


def test_partitions(pdf, df):
    assert_eq(df.partitions[0], pdf.iloc[:10])
    assert_eq(df.partitions[1], pdf.iloc[10:20])
    assert_eq(df.partitions[1:3], pdf.iloc[10:30])
    assert_eq(df.partitions[[3, 4]], pdf.iloc[30:50])
    assert_eq(df.partitions[-1], pdf.iloc[90:])

    out = (df + 1).partitions[0].simplify()
    assert isinstance(out.expr, expr.Add)
    assert isinstance(out.expr.left, expr.Partitions)


@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("npartitions", [3, 12])
@pytest.mark.parametrize("max_branch", [32, 8])
def test_task_shuffle(ignore_index, npartitions, max_branch):
    pdf = pd.DataFrame({"x": list(range(20)) * 5, "y": range(100)})
    df = from_pandas(pdf, npartitions=10)
    df2 = df.shuffle(
        "x",
        npartitions=npartitions,
        ignore_index=ignore_index,
        max_branch=max_branch,
    )

    # Check that the output partition count is correct
    assert df2.npartitions == (npartitions or df.npartitions)

    # Check the computed (re-ordered) result
    assert_eq(df, df2, check_index=not ignore_index)

    # Check that df was really partitioned by "x"
    unique = []
    for part in dask.compute(list(df2["x"].partitions))[0]:
        unique.extend(part.unique().tolist())
    # If any values of "x" can be found in multiple
    # partitions, then `len(unique)` will be >20
    assert sorted(unique) == list(range(20))


@pytest.mark.parametrize("npartitions", [3, 12])
@pytest.mark.parametrize("max_branch", [32, 8])
def test_task_shuffle_index(npartitions, max_branch):
    pdf = pd.DataFrame({"x": list(range(20)) * 5, "y": range(100)}).set_index("x")
    df = from_pandas(pdf, npartitions=10)
    df2 = df.shuffle(
        "x",
        npartitions=npartitions,
        max_branch=max_branch,
    )

    # Check that the output partition count is correct
    assert df2.npartitions == (npartitions or df.npartitions)

    # Check the computed (re-ordered) result
    assert_eq(df, df2)

    # Check that df was really partitioned by "x"
    unique = []
    for part in dask.compute(list(df2.index.partitions))[0]:
        unique.extend(part.unique().tolist())
    # If any values of "x" can be found in multiple
    # partitions, then `len(unique)` will be >20
    assert sorted(unique) == list(range(20))


def test_task_shuffle_p2p():
    from distributed import Client, LocalCluster

    pdf = pd.DataFrame({"x": list(range(20)) * 5, "y": range(100)}).set_index("x")
    df = from_pandas(pdf, npartitions=10)
    df2 = df.shuffle("x", backend="p2p")

    with Client(LocalCluster(n_workers=2)) as client:
        df2.compute()
        # Check the computed (re-ordered) result
        #assert_eq(df, df2)


def test_column_getattr(df):
    df = df.expr
    assert df.x._name == df["x"]._name

    with pytest.raises(AttributeError):
        df.foo


def test_serialization(pdf, df):
    before = pickle.dumps(df)

    assert len(before) < 200 + len(pickle.dumps(pdf))

    part = df.partitions[0].compute()
    assert (
        len(pickle.dumps(df.__dask_graph__()))
        < 1000 + len(pickle.dumps(part)) * df.npartitions
    )

    after = pickle.dumps(df)

    assert before == after  # caching doesn't affect serialization

    assert pickle.loads(before)._name == pickle.loads(after)._name
    assert_eq(pickle.loads(before), pickle.loads(after))


def test_size_optimized(df):
    expr = (df.x + 1).apply(lambda x: x).size
    out = optimize(expr)
    expected = optimize(df.x.size)
    assert out._name == expected._name

    expr = (df + 1).apply(lambda x: x).size
    out = optimize(expr)
    expected = optimize(df.size)
    assert out._name == expected._name


def test_tree_repr(df):
    from dask_match.datasets import timeseries

    df = timeseries()
    expr = ((df.x + 1).sum(skipna=False) + df.y.mean()).expr
    s = expr.tree_repr()

    assert "Sum" in s
    assert "Add" in s
    assert "1" in s
    assert "True" not in s
    assert "None" not in s
    assert "skipna=False" in s
    assert str(df.seed) in s.lower()


def test_simple_graphs(df):
    expr = (df + 1).expr
    graph = expr.__dask_graph__()

    assert graph[(expr._name, 0)] == (operator.add, (df.expr._name, 0), 1)
