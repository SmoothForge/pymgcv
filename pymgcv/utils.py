from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro

from pymgcv.rlibs import rbase, rutils
from pymgcv.rpy_utils import to_py


def load_rdata_dataframe_from_url(url: str) -> pd.DataFrame:
    """Load an RData (.rda) file from URL expecting a single dataframe, without R console output."""
    # Backup the original R console print callback
    original_consolewrite_print = rcb.consolewrite_print

    # Override with a dummy function to suppress output
    rcb.consolewrite_print = lambda x: None  # ignore any output
    rcb.consolewrite_warnerror = lambda x: None  # also ignore warnings if needed

    try:
        temp_file = rbase.tempfile(fileext=".RData")
        rutils.download_file(url, destfile=temp_file, mode="wb")
        loaded_names = rbase.load(temp_file)
        loaded_names_py = list(loaded_names)

        if len(loaded_names_py) != 1:
            raise ValueError(
                f"Expected one object, got {len(loaded_names_py)}: {loaded_names_py}",
            )

        data_r = ro.r[loaded_names_py[0]]
        data = to_py(data_r)
        rbase.file_remove(temp_file)

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a dataframe, got {type(data)}")

        return data

    finally:
        # Restore original R console print callback
        rcb.consolewrite_print = original_consolewrite_print
        rcb.consolewrite_warnerror = rcb.consolewrite_print


def get_data(name: str) -> Any:
    """Get built-in R dataset.

    This generally assumes the dataset is a dataframe, and may not work consistently
    for other types of data.

    """
    with ro.local_context() as lc:
        rutils.data(ro.rl(name), envir=lc)
        data = lc[name]

        if isinstance(data, ro.ListVector) and not isinstance(
            data,
            ro.vectors.DataFrame,
        ):
            data = to_py(data)
            names = np.asarray(list(data.names()))
            vals = list(data.values())
            return names, vals 

        return to_py(data)


def data_len(data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray]) -> int:
    """Get the length of the data.

    If the data is a dictionary, returns the maximum value of the shape along axis 0.
    """
    if isinstance(data, pd.DataFrame):
        return len(data)
    return max([d.shape[0] for d in data.values()])
