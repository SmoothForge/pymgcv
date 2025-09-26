from functools import partial

import numpy as np
import pandas as pd
import pytest

from pymgcv.gam import GAM
from pymgcv.qq import qq_simulate, qq_transform
from pymgcv.terms import L, S

from .gam_test_cases import GAMTestCase, get_test_cases


@pytest.mark.parametrize(
    "qq_fun",
    [
        partial(qq_transform, transform_to="uniform"),
        partial(qq_transform, transform_to="normal"),
        qq_simulate,
    ],
)
def test_qq_functions(qq_fun):
    """Test that qq_uniform runs without error and returns expected structure."""
    rng = np.random.default_rng(42)
    n = 100
    x0, x1, x2, x3 = [rng.uniform(-1, 1, n) for _ in range(4)]
    y = (
        0.5 * x0
        + np.sin(np.pi * x1)
        + np.cos(np.pi * x2) * np.sin(np.pi * x3)
        + rng.normal(0, 0.3, n)
    )
    data = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3, "y": y})

    gam = GAM({"y": L("x0") + S("x1") + S("x2", "x3")})
    gam.fit(data)

    result = qq_fun(gam)
    for arr in [result.theoretical, result.residuals]:
        assert len(arr) == n
        assert np.all(np.isfinite(arr))
        assert np.all(np.isfinite(arr))


test_cases = get_test_cases()


@pytest.mark.parametrize(
    "test_case",
    [test_cases["GauLSS"]],
)  # TODO add others.
@pytest.mark.parametrize("qq_fun", [qq_simulate])
def test_qq_functions_lss(test_case: GAMTestCase, qq_fun):
    gam = test_case.gam_model
    gam.fit(test_case.data)
    qq_fun(gam)
