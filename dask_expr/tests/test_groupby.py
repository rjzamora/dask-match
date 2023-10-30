import pytest

from dask_expr import from_pandas
from dask_expr._reductions import TreeReduce
from dask_expr.tests._util import _backend_library, assert_eq, xfail_gpu

# Set DataFrame backend for this module
lib = _backend_library()


@pytest.fixture
def pdf():
    pdf = lib.DataFrame({"x": list(range(10)) * 10, "y": range(100), "z": 1})
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=4)


@pytest.mark.xfail(reason="Cannot group on a Series yet")
def test_groupby_unsupported_by(pdf, df):
    assert_eq(df.groupby(df.x).sum(), pdf.groupby(pdf.x).sum())


@pytest.mark.parametrize(
    "api", ["sum", "mean", "min", "max", "prod", "first", "last", "var", "std"]
)
@pytest.mark.parametrize(
    "numeric_only",
    [
        pytest.param(True, marks=xfail_gpu("numeric_only not supported by cudf")),
        False,
    ],
)
def test_groupby_numeric(pdf, df, api, numeric_only):
    if not numeric_only and api in {"var", "std"}:
        pytest.xfail("not implemented")
    g = df.groupby("x")
    agg = getattr(g, api)(numeric_only=numeric_only)

    expect = getattr(pdf.groupby("x"), api)(numeric_only=numeric_only)
    assert_eq(agg, expect)

    g = df.groupby("x")
    agg = getattr(g, api)(numeric_only=numeric_only)["y"]

    expect = getattr(pdf.groupby("x"), api)(numeric_only=numeric_only)["y"]
    assert_eq(agg, expect)


@pytest.mark.parametrize(
    "func",
    [
        "count",
        pytest.param(
            "value_counts", marks=xfail_gpu("value_counts not supported by cudf")
        ),
        "size",
    ],
)
def test_groupby_no_numeric_only(pdf, func):
    pdf = pdf.drop(columns="z")
    df = from_pandas(pdf, npartitions=10)
    g = df.groupby("x")
    agg = getattr(g, func)()

    expect = getattr(pdf.groupby("x"), func)()
    assert_eq(agg, expect)


def test_groupby_mean_slice(pdf, df):
    g = df.groupby("x")
    agg = g.y.mean()

    expect = pdf.groupby("x").y.mean()
    assert_eq(agg, expect)


def test_groupby_series(pdf, df):
    pdf_result = pdf.groupby(pdf.x).sum()
    result = df.groupby(df.x).sum()
    assert_eq(result, pdf_result)
    result = df.groupby("x").sum()
    assert_eq(result, pdf_result)

    df2 = from_pandas(lib.DataFrame({"a": [1, 2, 3]}))

    with pytest.raises(ValueError, match="DataFrames columns"):
        df.groupby(df2.a)


@pytest.mark.parametrize(
    "spec",
    [
        {"x": "count"},
        {"x": ["count"]},
        {"x": ["count"], "y": "mean"},
        {"x": ["sum", "mean"]},
        ["min", "mean"],
        "sum",
    ],
)
def test_groupby_agg(pdf, df, spec):
    g = df.groupby("x")
    agg = g.agg(spec)

    expect = pdf.groupby("x").agg(spec)
    assert_eq(agg, expect)


def test_groupby_getitem_agg(pdf, df):
    assert_eq(df.groupby("x").y.sum(), pdf.groupby("x").y.sum())
    assert_eq(df.groupby("x")[["y"]].sum(), pdf.groupby("x")[["y"]].sum())


def test_groupby_agg_column_projection(pdf, df):
    g = df.groupby("x")
    agg = g.agg({"x": "count"}).simplify()

    assert list(agg.frame.columns) == ["x"]
    expect = pdf.groupby("x").agg({"x": "count"})
    assert_eq(agg, expect)


def test_groupby_split_every(pdf):
    df = from_pandas(pdf, npartitions=16)
    query = df.groupby("x").sum()
    tree_reduce_node = list(query.optimize(fuse=False).find_operations(TreeReduce))
    assert len(tree_reduce_node) == 1
    assert tree_reduce_node[0].split_every == 8

    query = df.groupby("x").aggregate({"y": "sum"})
    tree_reduce_node = list(query.optimize(fuse=False).find_operations(TreeReduce))
    assert len(tree_reduce_node) == 1
    assert tree_reduce_node[0].split_every == 8


def test_groupby_index(pdf):
    pdf = pdf.set_index("x")
    df = from_pandas(pdf, npartitions=10)
    result = df.groupby(df.index).sum()
    expected = pdf.groupby(pdf.index).sum()
    assert_eq(result, expected)
    assert_eq(result["y"], expected["y"])

    result = df.groupby(df.index).var()
    expected = pdf.groupby(pdf.index).var()
    assert_eq(result, expected)
    assert_eq(result["y"], expected["y"])

    result = df.groupby(df.index).agg({"y": "sum"})
    expected = pdf.groupby(pdf.index).agg({"y": "sum"})
    assert_eq(result, expected)


@pytest.mark.parametrize("api", ["sum", "mean", "min", "max", "prod", "var", "std"])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("split_out", [1, 2])
def test_groupby_single_agg_split_out(pdf, df, api, sort, split_out):
    g = df.groupby("x", sort=sort)
    agg = getattr(g, api)(split_out=split_out)

    expect = getattr(pdf.groupby("x", sort=sort), api)()
    assert_eq(agg, expect, sort_results=not sort)


@pytest.mark.parametrize(
    "spec",
    [
        {"x": "count"},
        {"x": ["count"]},
        {"x": ["count"], "y": "mean"},
        {"x": ["sum", "mean"]},
        ["min", "mean"],
        "sum",
    ],
)
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("split_out", [1, 2])
def test_groupby_agg_split_out(pdf, df, spec, sort, split_out):
    g = df.groupby("x", sort=sort)
    agg = g.agg(spec, split_out=split_out)

    expect = pdf.groupby("x", sort=sort).agg(spec)
    assert_eq(agg, expect, sort_results=not sort)


def test_groupby_reduction_shuffle(df, pdf):
    q = df.groupby("x").sum(split_out=True)
    assert q.optimize().npartitions == df.npartitions
    expected = pdf.groupby("x").sum()
    assert_eq(q, expected)


def test_groupby_projection_split_out(df, pdf):
    pdf_result = pdf.groupby("x")["y"].sum()
    result = df.groupby("x")["y"].sum(split_out=2)
    assert_eq(result, pdf_result)

    pdf_result = pdf.groupby("y")["x"].sum()
    df = from_pandas(pdf, npartitions=50)
    result = df.groupby("y")["x"].sum(split_out=2)
    assert_eq(result, pdf_result)
