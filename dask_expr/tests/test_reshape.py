import pytest

from dask_expr import from_pandas
from dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
lib = _backend_library()


@pytest.fixture
def pdf():
    pdf = lib.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": lib.Series([4, 5, 8, 6, 1, 4], dtype="category"),
            "z": [4, 15, 8, 16, 1, 14],
            "a": 1,
        }
    )
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=3)


@pytest.mark.parametrize("aggfunc", ["first", "last", "sum", "mean", "count"])
def test_pivot_table(df, pdf, aggfunc):
    assert_eq(
        df.pivot_table(index="x", columns="y", values="z", aggfunc=aggfunc),
        pdf.pivot_table(index="x", columns="y", values="z", aggfunc=aggfunc),
        check_dtype=aggfunc != "count",
    )

    assert_eq(
        df.pivot_table(index="x", columns="y", values=["z", "a"], aggfunc=aggfunc),
        pdf.pivot_table(index="x", columns="y", values=["z", "a"], aggfunc=aggfunc),
        check_dtype=aggfunc != "count",
    )


def test_pivot_table_fails(df):
    with pytest.raises(ValueError, match="must be the name of an existing column"):
        df.pivot_table(index="aaa", columns="y", values="z")
    with pytest.raises(ValueError, match="must be the name of an existing column"):
        df.pivot_table(index=["a"], columns="y", values="z")

    with pytest.raises(ValueError, match="must be the name of an existing column"):
        df.pivot_table(index="a", columns="xxx", values="z")
    with pytest.raises(ValueError, match="must be the name of an existing column"):
        df.pivot_table(index="a", columns=["x"], values="z")

    with pytest.raises(ValueError, match="'columns' must be category dtype"):
        df.pivot_table(index="a", columns="x", values="z")

    df2 = df.copy()
    df2["y"] = df2.y.cat.as_unknown()
    with pytest.raises(ValueError, match="'columns' categories must be known"):
        df2.pivot_table(index="a", columns="y", values="z")

    with pytest.raises(
        ValueError, match="'values' must refer to an existing column or columns"
    ):
        df.pivot_table(index="x", columns="y", values="aaa")
