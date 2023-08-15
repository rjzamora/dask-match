from dask.dataframe.utils import assert_eq

from dask_expr import new_collection
from dask_expr._expr import Lengths
from dask_expr.datasets import Timeseries, timeseries


def test_timeseries():
    df = timeseries(freq="360 s", start="2000-01-01", end="2000-01-02")
    assert_eq(df, df)


def test_optimization():
    df = timeseries(dtypes={"x": int, "y": float}, seed=123)
    result = df[["x"]].optimize()
    assert result.expr._name == df.expr.substitute_parameters({"columns": ["x"]})._name

    result = df["x"].optimize(fuse=False)
    assert (
        result.expr._name
        == df.expr.substitute_parameters({"columns": ["x"], "_series": True})._name
    )


def test_column_projection_deterministic():
    df = timeseries(freq="1H", start="2000-01-01", end="2000-01-02", seed=123)
    result_id = df[["id"]].optimize()
    result_id_x = df[["id", "x"]].optimize()
    assert_eq(result_id["id"], result_id_x["id"])


def test_timeseries_culling():
    df = timeseries(dtypes={"x": int, "y": float}, seed=123)
    pdf = df.compute()
    offset = len(df.partitions[0].compute())
    df = (df[["x"]] + 1).partitions[1]
    df2 = df.optimize()

    # All tasks should be fused for the single output partition
    assert df2.npartitions == 1
    assert len(df2.dask) == df2.npartitions
    expected = pdf.iloc[offset : 2 * offset][["x"]] + 1
    assert_eq(df2, expected)


def test_persist():
    df = timeseries(freq="1H", start="2000-01-01", end="2000-01-02", seed=123)
    a = df["x"]
    b = a.persist()

    assert_eq(a, b)
    assert len(a.dask) == len(b.dask)
    assert len(b.dask) == b.npartitions


def test_lengths():
    df = timeseries(freq="1H", start="2000-01-01", end="2000-01-03", seed=123)
    assert len(df) == sum(new_collection(Lengths(df.expr).optimize()).compute())


def test_timeseries_empty_projection():
    ts = timeseries(end="2000-01-02", dtypes={})
    expected = timeseries(end="2000-01-02")
    assert len(ts) == len(expected)


def test_combine_similar(tmpdir):
    df = timeseries(end="2000-01-02")
    pdf = df.compute()
    got = df[df["name"] == "a"][["id"]]

    expected = pdf[pdf["name"] == "a"][["id"]]
    assert_eq(got, expected)
    assert_eq(got.optimize(fuse=False), expected)
    assert_eq(got.optimize(fuse=True), expected)

    # We should only have one Timeseries node, and
    # it should not include "z" in the dtypes
    timeseries_nodes = list(got.optimize(fuse=False).find_operations(Timeseries))
    assert len(timeseries_nodes) == 1
    assert set(timeseries_nodes[0].dtypes.keys()) == {"id", "name"}
