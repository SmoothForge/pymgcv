"""Data conversion utilities for Python-R interoperability.

This module provides convenient functions for converting data between Python
and R representations, particularly for use with rpy2. It handles the conversion
of pandas DataFrames to R data frames and various other Python objects to their
R equivalents, with proper handling of numpy arrays and pandas-specific features.

The conversions are essential for seamless integration with R's mgcv library
while maintaining pythonic data structures on the Python side.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from pandas.api.types import is_integer_dtype
from rpy2.robjects import numpy2ri, pandas2ri

from pymgcv.rlibs import rbase


def to_rpy(x, *, as_vector: bool = False):
    """Convert Python object to R object using rpy2.

    Handles automatic conversion of pandas DataFrames, numpy arrays, and
    other Python objects to their R equivalents using the appropriate
    rpy2 converters.

    Args:
        x: Python object to convert (DataFrame, array, list, etc.)
        as_vector: If True, 1-D numpy arrays are returned as plain R vectors. By
            default they are converted to 1-D R arrays (with a ``dim`` attribute),
            which R will not multiply elementwise with a matrix, e.g. as
            ``mgcv::bam`` does with the weights.

    Returns:
        R object equivalent of the input, ready for use in R function calls

    Examples:
        ```python
        import pandas as pd
        import numpy as np

        # Convert DataFrame
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        r_df = to_rpy(df)

        # Convert numpy array
        arr = np.array([1, 2, 3])
        r_arr = to_rpy(arr)  # 1-D R array
        r_vec = to_rpy(arr, as_vector=True)  # plain R vector
        ```
    """
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        r_obj = ro.conversion.get_conversion().py2rpy(x)
    if as_vector and isinstance(x, np.ndarray) and x.ndim == 1:
        return rbase.as_vector(r_obj)
    return r_obj


def to_py(x) -> Any:
    """Convert R object to Python object using rpy2.

    Handles automatic conversion of R data structures back to their Python
    equivalents, including R data frames to pandas DataFrames and R vectors
    to numpy arrays.

    Args:
        x: R object to convert (data.frame, vector, list, etc.)

    Returns:
        Python object equivalent of the input

    Examples:
        ```python
        # Convert R vector to numpy array
        r_vector = ro.IntVector([1, 2, 3])
        py_array = to_py(r_vector)  # Returns numpy array

        # Convert R data frame to pandas DataFrame
        r_df = robjects.r('data.frame(x=1:3, y=4:6)')
        py_df = to_py(r_df)  # Returns pandas DataFrame
        ```
    """
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        py_obj = ro.conversion.get_conversion().rpy2py(x)

    if isinstance(py_obj, pd.DataFrame):  # Handle R nan integer encoding
        r_nan = -2147483648
        for col in py_obj.columns:
            if is_integer_dtype(py_obj[col]) and any(py_obj[col] == r_nan):
                py_obj[col] = py_obj[col].replace(r_nan, np.nan).astype(pd.Int64Dtype())
    return py_obj


def data_to_rdf(
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
    *,
    include: Iterable[str] | Literal["all"],
) -> ro.vectors.DataFrame:
    """Convert pandas DataFrame or dictionary to R data.frame for use with mgcv.

    Args:
        data: DataFrame or dictionary containing all variables referenced in the model.
            Note, using a dictionary is required when passing matrix-valued variables.
        include: The variables to include. We force passing this because it promotes
            being explicit which columns to convert, and avoid trying to convert e.g.
            unused object columns, which will lead to warnings from rpy2. Elements
            not present in the data will be ignored.

    Returns:
        R data.frame object ready for use with mgcv functions.
    """
    if include != "all":
        if isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data[[col for col in data.columns if col in include]])
        else:
            data = {k: v for k, v in data.items() if k in include}

    multidim_data = {}
    if isinstance(data, dict):
        multidim_data = {
            k: rbase.I(to_rpy(v)) for k, v in data.items() if np.ndim(v) > 1
        }
        other_data = {k: v for k, v in data.items() if np.ndim(v) == 1}
        data = pd.DataFrame(other_data)

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    rpy_df = to_rpy(data)
    multidim_data = rbase.data_frame(**multidim_data)

    if rpy_df.nrow == 0:
        return multidim_data
    if multidim_data.nrow == 0:
        return rpy_df
    return rbase.cbind(rpy_df, multidim_data)


def is_null(object) -> bool:
    """Check if an object is NULL.

    We use this to avoid the possible confusion with ``rbase.is_null`` returning
    a boolean vector, which e.g. acts as a "truthy" value regardless of its
    contents.
    """
    return rbase.is_null(object)[0]
