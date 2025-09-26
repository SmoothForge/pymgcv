from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymgcv.gam import BAM, GAM, AbstractGAM
from pymgcv.rpy_utils import to_py
from pymgcv.terms import L
from pymgcv.utils import data_len

from .gam_test_cases import GAMTestCase, get_test_cases, smooth_1d_by_numeric_gam

test_cases = get_test_cases()


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_pymgcv_mgcv_equivilance(test_case: GAMTestCase):
    pymgcv_gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    mgcv_gam = test_case.mgcv_gam(test_case.data)
    assert (
        pytest.approx(
            expected=mgcv_gam.rx2["coefficients"],
        )
        == pymgcv_gam.coefficients().to_numpy()
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_partial_effects_colsum_matches_predict(test_case: GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    predictions = gam.predict(test_case.data)
    term_predictions = gam.partial_effects(test_case.data, compute_se=True)

    for target, pred in predictions.items():
        term_fit = term_predictions[target].fit
        assert pytest.approx(pred) == term_fit.sum(axis=1)


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_partial_effects(test_case: GAMTestCase):
    """Consistency check between partial_effect and partial_effects."""
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    partial_effects = gam.partial_effects(test_case.data, compute_se=True)

    predictors = gam.predictors

    for target, terms in predictors.items():
        for term in terms:
            effect = gam.partial_effect(
                term,
                target,
                test_case.data,
                compute_se=True,
            )
            name = term.label()
            expected_fit = pytest.approx(partial_effects[target].fit[name], abs=1e-6)
            expected_se = pytest.approx(partial_effects[target].se[name], abs=1e-6)

            assert expected_fit == effect.fit
            assert expected_se == effect.se


def test_invalid_type():
    rng = np.random.default_rng(1)
    gam = GAM({"y": L("x")})
    data = pd.DataFrame({"y": rng.normal(size=100), "x": rng.normal(size=100)})
    data["x"] = data["x"].astype(str)
    with pytest.raises(TypeError, match="is of unsupported type"):
        gam = gam.fit(data)

    data = {"x": np.asarray(data["x"]), "y": data["y"]}
    with pytest.raises(TypeError, match="is of unsupported type"):
        gam = gam.fit(data)


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_with_se_matches_without(test_case: GAMTestCase):
    gam = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)

    partial_effects_with_se = gam.partial_effects(compute_se=True)
    partial_effects_without = gam.partial_effects(compute_se=False)

    for target in gam.predictors.keys():
        assert (
            pytest.approx(partial_effects_with_se[target].fit)
            == partial_effects_without[target]
        )


abstract_method_test_cases = [
    "GAM - smooth_1d_gam",
    "BAM - smooth_1d_random_wiggly_curve_gam",
    "MVN",
    "GauLSS",
]


# check it gives something reasonable in both uni/multivariate, and bam case
@pytest.mark.parametrize(
    "test_case",
    [test_cases[k] for k in abstract_method_test_cases],
    ids=abstract_method_test_cases,
)
def test_abstract_methods(test_case: GAMTestCase):
    fit = test_case.gam_model.fit(test_case.data, **test_case.fit_kwargs)
    coef = fit.coefficients()
    cov = fit.covariance()
    assert cov.shape[0] == cov.shape[1]
    assert cov.shape[0] == coef.shape[0]
    assert np.all(coef.index == cov.index)
    assert isinstance(fit.aic(), float)

    residuals = fit.residuals()

    assert residuals.shape[0] == data_len(test_case.data)
    assert fit.fit_state is not None
    mgcv_gam = fit.fit_state.rgam
    resid_from_y_and_fit = fit.residuals_from_y_and_fit(
        y=to_py(mgcv_gam.rx2["y"]),
        fit=to_py(mgcv_gam.rx2["fitted.values"]),
        weights=to_py(mgcv_gam.rx2["prior.weights"]),
    )
    assert np.all(residuals == resid_from_y_and_fit)
    assert isinstance(fit.edf(), pd.Series)
    assert isinstance(fit.penalty_edf(), pd.Series)

    k_check = fit.check_k()
    assert isinstance(k_check, pd.DataFrame)
    expected_columns = ["term", "max_edf", "edf", "k_index", "p_value"]
    assert k_check.columns.to_list() == expected_columns


@pytest.mark.parametrize("gam_type", [GAM, BAM])
@pytest.mark.parametrize("add_nan_to", ["y", "x", "x1"])  # Response, predictor, by
@pytest.mark.parametrize("nan", [np.nan, pd.NA])
def test_errors_on_nans(gam_type: type[AbstractGAM], add_nan_to: str, nan: Any):
    # Erroring on nans seems a sensible default. Strict but avoids
    # possible issues (e.g. length mismatch if unexpectedly dropped).
    test_case: GAMTestCase = smooth_1d_by_numeric_gam(gam_type)

    data = deepcopy(test_case.data)
    gam = test_case.gam_model
    gam.fit(data)
    assert isinstance(data, pd.DataFrame)
    data[add_nan_to] = (
        (data[add_nan_to] * 100).round().astype(pd.Int64Dtype())
    )  # nullable int
    data.at[2, add_nan_to] = nan

    with pytest.raises(ValueError, match="NaN"):
        gam.fit(data)

    if add_nan_to != "y":
        with pytest.raises(ValueError, match="NaN"):
            gam.predict(data)

        with pytest.raises(ValueError, match="NaN"):
            gam.partial_effects(data)

        with pytest.raises(ValueError, match="NaN"):
            gam.partial_effect(gam.predictors["y"][0], data=data)
