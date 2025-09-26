from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import beta, norm

from pymgcv.families import SupportsCDF
from pymgcv.gam import AbstractGAM
from pymgcv.rlibs import rbase, rstats
from pymgcv.rpy_utils import is_null, to_py


@dataclass
class QQResult:
    """Results required for a qq plot."""

    theoretical: np.ndarray
    residuals: np.ndarray
    interval: tuple[np.ndarray, np.ndarray]  # lower, upper


def qq_simulate(
    gam: AbstractGAM,
    *,
    n_sim: int = 50,
    level: float = 0.9,
    type: Literal["deviance", "response", "pearson"] = "deviance",
) -> QQResult:
    """Generate data for qq-plot via simulation from the family.

    Args:
        gam: The fitted GAM object.
        n_sim: The number of simulations to perform.
        level: The level (i.e. 0.9 means 90% interval).
        type: The type of residuals to use.
    """
    if gam.fit_state is None:
        raise ValueError("GAM must be fitted before simulating quantiles.")
    if n_sim < 2:
        raise ValueError("n must be at least 2.")

    model = gam.fit_state.rgam
    fit = rstats.fitted(model)
    weights = rstats.weights(model, type="prior")
    sigma2 = model.rx2["sig2"]

    if is_null(sigma2):
        sigma2 = rbase.summary(model, re_test=False).rx2["dispersion"]

    fit, weights, sigma2 = to_py(fit), to_py(weights), to_py(sigma2)

    if gam.family.n_observed_predictors > 1:
        raise NotImplementedError(
            "Multivariate response families are not supported for qq_simulate.",
        )

    sims = []
    for _ in range(n_sim):
        ysim = gam.family.sample(mu=fit, wt=weights, scale=sigma2)
        res = gam.residuals_from_y_and_fit(
            y=ysim,
            fit=fit,
            weights=weights,
            type=type,
        )
        sim = np.sort(res)
        sims.append(sim)

    sims = np.stack(sims, axis=1)
    n_obs = len(fit)
    theoretical = np.quantile(
        sims,
        q=(np.arange(n_obs) + 0.5) / n_obs,
    )
    alpha = (1 - level) / 2
    interval = np.quantile(sims, q=(alpha, 1 - alpha), axis=1)
    residuals = gam.residuals(type=type)
    residuals = np.sort(residuals)

    return QQResult(
        theoretical=theoretical,
        residuals=residuals,
        interval=(interval[0], interval[1]),
    )


def qq_transform(
    gam: AbstractGAM,
    *,
    transform_to: Literal["normal", "uniform"] = "normal",
    level: float = 0.9,
) -> QQResult:
    """Generate a QQ-plot by transforming the data to a known distribution.

    This plots the theoretical quantiles against the transformed data. The data
    are transformed by 1) passing through the CDF implied by the model, and 2)
    passing to the quantile function of the distribution implied by ``transform_to``.

    !!! note

       Using ``transform_to="uniform"`` may hide outliers/tail behaviour issues,
       as the data is constrained to [0, 1], which can be hard to assess visually.

    Args:
        gam: The fitted GAM object. The family should support the CDF method,
            which can be checked with `isinstance(family, SupportsCDF)`.
        transform_to: The distribution to transform the residuals to.
        level: The confidence level for the interval.

    Returns:
        QQResult: The results required for a qq plot.
    """
    if gam.fit_state is None:
        raise ValueError("GAM has not been fit")

    if not isinstance(gam.family, SupportsCDF):
        raise TypeError("Family must support CDF method to use qq_transform.")

    class _Uniform:
        @staticmethod
        def quantile(probs):
            return probs

        @staticmethod
        def confidence_interval(alpha, n_obs: int):
            a = np.arange(n_obs)
            lower = beta.ppf(alpha, a, n_obs - a + 1)
            upper = beta.ppf(1 - alpha, a, n_obs - a + 1)
            return lower, upper

    class _Normal:
        @staticmethod
        def quantile(probs):
            q = norm.ppf(probs)
            # Below not perfect, but will make clear there is a problem
            return np.nan_to_num(q, nan=np.nan, posinf=100, neginf=-100)

        @staticmethod
        def confidence_interval(alpha, n_obs: int):
            a = (np.arange(n_obs) + 0.5) / n_obs
            q = norm.ppf(a)
            offset = norm.ppf(alpha) * np.sqrt(a * (1 - a) / n_obs) / norm.pdf(q)
            return q - offset, q + offset

    dist = {"uniform": _Uniform, "normal": _Normal}[transform_to]
    model = gam.fit_state.rgam
    fit = rstats.fitted(model)
    weights = rstats.weights(model, type="prior")
    sigma2 = model.rx2["sig2"]

    if is_null(sigma2):
        sigma2 = rbase.summary(model, re_test=False).rx2["dispersion"]

    fit, weights, sigma2 = to_py(fit), to_py(weights), to_py(sigma2)

    # Transform the data to uniform (CDF)
    probs = gam.family.cdf(
        to_py(gam.fit_state.rgam.rx2["y"]),
        mu=fit,
        wt=weights,
        scale=sigma2,
    )
    probs = np.clip(probs, 0, 1)  # Probably not needed but avoid nan risk
    probs = np.sort(probs)
    transformed = dist.quantile(probs)

    n_obs = len(probs)
    theoretical = dist.quantile((np.arange(n_obs) + 0.5) / n_obs)

    alpha = (1 - level) / 2
    lower, upper = dist.confidence_interval(alpha, n_obs)
    return QQResult(
        theoretical=theoretical,
        residuals=transformed,
        interval=(lower, upper),
    )
