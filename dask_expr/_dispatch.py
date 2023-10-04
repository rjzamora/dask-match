from __future__ import annotations

from dask.utils import Dispatch


class DefaultDispatch(Dispatch):
    __default = None

    def set_default(self, val):
        if isinstance(val, type):
            self.__default = lambda _: val
        else:
            self.__default = val

    def dispatch(self, cls):
        try:
            return super().dispatch(cls)
        except TypeError as err:
            if self.__default is not None:
                return self.__default
            raise err


# Define dispatchable functions

get_dataframe_class = DefaultDispatch("get_dataframe_class")
get_series_class = DefaultDispatch("get_series_class")
get_index_class = DefaultDispatch("get_index_class")
get_groupby_class = DefaultDispatch("get_groupby_class")
