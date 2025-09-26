"""Formula utilities."""

# R formulas are hard to work with programmatically. We want to be able to pass
# variables (e.g. rpy2 objects) to R formulas. The two ways to do this (to my knowledge)
# are by converting to a string representation of R code, or assigning them to an R
# and using the variable name. Both have downsides e.g. we don't want to convert large
# arrays to strings representations, and overreliance on assigning R variables makes
# the formulas less readable. Hence, we currently use a combination of both approaches.
from dataclasses import dataclass
from typing import Any

from rpy2 import robjects as ro

from pymgcv.rlibs import rbase
from pymgcv.rpy_utils import to_rpy


@dataclass
class _Var:
    name: str


def _to_r_constructor_string(arg: Any) -> str:
    """Converts an object to R string representation.

    _Var acts as a placeholder for a variable name in R.

    Args:
        arg: The object to convert. If not already an RObject, it is first passed to
            `to_rpy`.
    """
    if isinstance(arg, _Var):
        return arg.name

    if not isinstance(arg, ro.RObject):
        arg = to_rpy(arg)

    connection = rbase.textConnection("__r_obj_str", "w")
    rbase.dput(arg, file=connection)
    rbase.close(connection)
    result = ro.r["__r_obj_str"]
    assert len(result) == 1  # type: ignore
    return result[0]  # type: ignore
