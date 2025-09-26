"""Core GAM fitting and model specification functionality."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pprint import pformat
from typing import Literal, Self, overload

import numpy as np
import pandas as pd
import rpy2.rinterface as ri
import rpy2.robjects as ro
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from pymgcv.custom_types import FitAndSE
from pymgcv.families import AbstractFamily, Gaussian
from pymgcv.rlibs import rbase, rmgcv, rstats, rutils
from pymgcv.rpy_utils import data_to_rdf, to_py, to_rpy
from pymgcv.terms import AbstractTerm, Intercept
from pymgcv.utils import data_len

GAMFitMethods = Literal[
    "GCV.Cp",
    "GACV.Cp",
    "QNCV",
    "REML",
    "P-REML",
    "ML",
    "P-ML",
    "NCV",
]

BAMFitMethods = Literal[
    "fREML",
    "GCV.Cp",
    "GACV.Cp",
    "REML",
    "P-REML",
    "ML",
    "P-ML",
    "NCV",
]


@dataclass
class FitState:
    """The mgcv gam, and the data used for fitting.

    This gets set as an attribute fit_state on the AbstractGAM object after fitting.

    Attributes:
        rgam: The fitted mgcv gam object.
        data: The data used for fitting.
    """

    rgam: ro.ListVector
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series]

    def __repr__(self):
        """Simplified repr to show the structure which is sufficient for debugging."""
        return f"FitState(rgam=mgcv.gam, data={type(self.data).__name__})"


@dataclass
class AbstractGAM(ABC):
    """Abstract base class for GAM models.

    This class cannot be initialized but provides a common interface for fitting and
    predicting using different types of GAM models.
    """

    predictors: dict[str, list[AbstractTerm]]
    family: AbstractFamily
    add_intercepts: bool
    fit_state: FitState | None

    def __init__(
        self,
        predictors: Mapping[str, Iterable[AbstractTerm] | AbstractTerm],
        *,
        family: AbstractFamily | None = None,
        add_intercepts: bool = True,
    ):
        r"""Initialize the model.

        Args:
            predictors: Dictionary mapping target variable names to an iterable of
                [`AbstractTerm`][pymgcv.terms.AbstractTerm] objects used to predict
                $g([\mathbb{E}[Y])$.

                - For simple models, this will usually be a single
                    key-value pair:
                    ```python
                    {"y": S("x1") + S("x2")}
                    ```
                - For multivariate models, e.g. [`MVN`][pymgcv.families.MVN], the
                    dictionary will have multiple pairs:
                        ```python
                        {"y1": S("x1") + S("x2"), "y2": S("x2")}
                        ```
                - For multiparameter models, such as LSS-type models (e.g.
                [`GauLSS`][pymgcv.families.GauLSS]), the first key-value pair
                must correspond to the variable name in the data (usually modelling the
                location), and the subsequent dictionary elements model the other
                parameters **in the order as defined by the
                family** (e.g. scale and shape). The names of these extra parameters,
                can be anything, and are used as column names for prediction outputs.
                ```python
                {"y": S("x1") + S("x2"), "scale": S("x2")}
                ```

            family: Distribution family to use. See [Families](./families.md)
                for available options.
            add_intercepts: If False, intercept terms must be manually added to the
                formulae using [`Intercept`][pymgcv.terms.Intercept]. If True,
                automatically adds an intercept term to each formula. Intercepts are
                added as needed by methods, such that ``gam.predictors`` reflect the
                model as constructed (i.e. before adding intercepts).
        """
        predictors = dict(predictors).copy()
        family = Gaussian() if family is None else family

        predictors = {
            k: [v] if isinstance(v, AbstractTerm) else list(v)
            for k, v in predictors.items()
        }
        self.predictors = predictors
        self.family = family
        self.fit_state = None
        self.add_intercepts = add_intercepts
        self._check_init()

    def _check_init(self):
        # Perform some basic checks
        for terms in self.predictors.values():
            identifiers = set()
            labels = set()
            for term in terms:
                mgcv_id = term.mgcv_identifier()
                label = term.label()
                if mgcv_id in identifiers or label in labels:
                    raise ValueError(
                        f"Duplicate term with label '{label}' and mgcv_identifier "
                        f"'{mgcv_id}' found in formula. pymgcv does not support "
                        "duplicate terms. If this is intentional, consider duplicating "
                        "the corresponding variable in your data under a new name and "
                        "using it for one of the terms.",
                    )
                identifiers.add(mgcv_id)
                labels.add(label)

        if len(self.predictors) != self.family.n_predictors:
            raise ValueError(
                f"Expected {self.family.n_predictors} predictors, but received "
                f"{len(self.predictors)} predictors.",
            )

        disallowed = [
            "Intercept",
            ".xt",
            ".m",
        ]
        for var in self.referenced_variables:
            if rbase.make_names(var)[0] != var:
                raise ValueError(
                    f"Variable '{var}' is not a valid R variable name, and must be "
                    f"renamed.",
                )
            if any(var.startswith(d) for d in disallowed):
                raise ValueError(
                    f"Variable name '{var}' is reserved by pymgcv, so must be renamed.",
                )

    def __repr__(self):
        """Formats repr over multiple lines for readability."""
        parts = []
        for name, value in self.__dict__.items():
            parts.append(f"{name}={pformat(value)}")
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(parts) + ",\n)"

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *args,
        **kwargs,
    ) -> Self:
        """Fit the GAM model to the given data."""
        pass

    @overload
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *args,
        compute_se: Literal[False] = False,
        type: Literal["response", "link"] = "link",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        pass

    @overload
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *args,
        compute_se: Literal[True],
        type: Literal["response", "link"] = "link",
        **kwargs,
    ) -> dict[str, FitAndSE[np.ndarray]]:
        pass

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *args,
        compute_se: bool = False,
        type: Literal["response", "link"] = "link",
        **kwargs,
    ) -> dict[str, np.ndarray] | dict[str, FitAndSE[np.ndarray]]:
        """Predict the response variable(s) (link scale) for the given data."""
        pass

    @overload
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *args,
        compute_se: Literal[False] = False,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        pass

    @overload
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *args,
        compute_se: Literal[True],
        **kwargs,
    ) -> dict[str, FitAndSE[pd.DataFrame]]:
        pass

    @abstractmethod
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *args,
        compute_se: bool = False,
        **kwargs,
    ) -> dict[str, pd.DataFrame] | dict[str, FitAndSE[pd.DataFrame]]:
        """Compute the partial effects for the terms in the model."""
        pass

    @property
    def referenced_variables(self) -> list[str]:
        """List of variables referenced by the model required to be present in data."""
        vars = set(list(self.predictors.keys())[: self.family.n_observed_predictors])
        for predictor in self.predictors.values():
            for term in predictor:
                vars.update(term.varnames)
                if term.by is not None:
                    vars.add(term.by)
        return list(vars)

    def _check_data(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *,
        requires: Literal["all", "covariates"] | AbstractTerm,
    ) -> None:
        """Validate that data contains all variables required by the model.

        Checks that all variables required exist in the data, and are not
        strings/objects, and do not contain NaNs.

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model.
            requires: If "all", checks that all response variables and covariates are
               present and valid (e.g. as required for fitting). If "covariates",
               then the response variable does not have to be present (e.g. as
               required for prediction). If a term, only variables required for
               that term are required (i.e. for computing a partial effect).
        """
        match requires:
            case "all":
                required_vars = self.referenced_variables
            case "covariates":
                target = list(self.predictors.keys())[
                    : self.family.n_observed_predictors
                ]
                required_vars = [
                    v for v in self.referenced_variables if v not in target
                ]
            case AbstractTerm():
                required_vars = requires.varnames
                if requires.by is not None:
                    required_vars += (requires.by,)

        for var in required_vars:
            if var not in data:
                raise ValueError(
                    f"Variable {var} referenced in model not present in data.",
                )
            dtype = data[var].dtype
            is_cat = isinstance(dtype, pd.CategoricalDtype)
            if is_object_dtype(dtype) or is_string_dtype(dtype) and not is_cat:
                raise TypeError(
                    f"Variable {var} is of unsupported type {data[var].dtype}.",
                )
            if pd.isna(data[var]).any():  # type: ignore
                raise ValueError(
                    f"NaNs present in required variable {var}. "
                    "Either impute or drop rows containing NaNs.",
                )

    def _to_r_formula_strings(self) -> list[str] | str:
        """Convert the gam model formula into an mgcv-style formula string.

        Returns:
            A string for single-formula models, or a list of strings for
                multi-formula models.
        """
        formulae = []
        predictors = (
            self.predictors
            if not self.add_intercepts
            else _add_intercepts(self.predictors)
        )

        unobserved = list(predictors.keys())[self.family.n_observed_predictors :]

        for target, terms in predictors.items():
            if target in unobserved:  # e.g. lss scale
                target = ""  # no left hand side

            formula_str = f"{target}~{'+'.join(map(str, terms))}"
            if not any(isinstance(term, Intercept) for term in terms):
                formula_str += "-1"
            formulae.append(formula_str)
        return formulae if len(formulae) > 1 else formulae[0]  # type: ignore[return-value]

    def _to_r_formulae(self) -> ro.Formula | list[ro.Formula]:
        """Convert the model specification to R formula objects.

        Returns:
            Single Formula object for simple models, or list of Formula objects
            for multi-formula models.
        """
        formulae = self._to_r_formula_strings()
        if isinstance(formulae, str):
            return ro.Formula(formulae)
        return [ro.Formula(f) for f in formulae]

    def summary(self) -> str:
        """Generate an mgcv-style summary of the fitted GAM model."""
        if self.fit_state is None:
            raise RuntimeError("Cannot print summary of an unfitted model.")
        strvec = rutils.capture_output(rbase.summary(self.fit_state.rgam))
        return "\n".join(tuple(strvec))

    def check_k(self, subsample: int = 5000, n_rep: int = 400) -> pd.DataFrame:
        """Checking basis dimension choices (k).

        The default choices for ``k`` are relatively arbitrary. This function aids in
        assessing whether the chosen basis dimensions are appropriate. A low p-value can
        indicate that the chosen basis dimension is too low.

        The function works by constrasting a residual variance estimate based on near
        neighbour points (based on the covariates of a term), to the overall residual
        variance. The ``k_index`` is the ratio of the near neighbour estimate to the
        overall variance. The further below 1 the ``k_index`` is, the more likely it is
        that there exists missed patterns in the residuals. The p-value is generated
        using a randomization test to obtain the null distribution.

        For details, see section 5.9 of:

            Wood S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
            edition). Chapman and Hall/CRC Press.

        Args:
            subsample: The maximum number of points to use, above which a random
                subsample is used.
            n_rep: The number of re-shuffles to do to get the p-value.

        Returns:
            A dataframe with the following columns:

                - `term`: The mgcv-style name of the smooth term.
                - `max_edf`: The maximum possible edf (often ``k-1``).
                - `k_index`: The ratio between the nearest neighbour variance
                   residual variance estimate and the overall variance.
                - `p_value`: The p-value of the randomization test.
                - `max_edf`: The maximum effective degrees of freedom.
        """
        if self.fit_state is None:
            raise ValueError("Cannot run check_k on an unfitted model.")

        rgam = self.fit_state.rgam
        result = rmgcv.k_check(rgam, subsample=subsample, n_rep=n_rep)
        rownames, colnames = result.rownames, result.colnames
        df = pd.DataFrame(to_py(result), columns=colnames)
        df.insert(0, "term", rownames)
        return df.rename(
            columns={"k'": "max_edf", "p-value": "p_value", "k-index": "k_index"},
        )

    def coefficients(self) -> pd.Series:  # TODO consider returning as dict?
        """Extract model coefficients from the fitted GAM.

        Returns a series where the index if the mgcv-style name of the parameter.
        """
        if self.fit_state is None:
            raise RuntimeError("Cannot extract coefficients from an unfitted model.")
        coef = self.fit_state.rgam.rx2["coefficients"]
        names = coef.names
        return pd.Series(to_py(coef), index=names)

    def covariance(
        self,
        *,
        sandwich: bool = False,
        freq: bool = False,
        unconditional: bool = False,
    ) -> pd.DataFrame:
        """Extract the covariance matrix from the fitted GAM.

        Extracts the Bayesian posterior covariance matrix of the parameters or
        frequentist covariance matrix of the parameter estimators from the fitted GAM.

        Args:
            sandwich: If True, compute sandwich estimate of covariance matrix.
                Currently expensive for discrete bam fits.
            freq: If True, return the frequentist covariance matrix of the parameter
                estimators. If False, return the Bayesian posterior covariance matrix
                of the parameters. The latter option includes the expected squared bias
                according to the Bayesian smoothing prior.
            unconditional: If True (and freq=False), return the Bayesian smoothing
                parameter uncertainty corrected covariance matrix, if available.

        Returns:
            The covariance matrix as a pandas dataframe where the column names and index
            are the mgcv-style parameter names.

        """
        if self.fit_state is None:
            raise RuntimeError("Cannot extract covariance from an unfitted model.")

        if unconditional and freq:
            raise ValueError("Unconditional and freq cannot both be True")

        coef_names = self.fit_state.rgam.rx2["coefficients"].names
        cov = to_py(
            rstats.vcov(
                self.fit_state.rgam,
                sandwich=sandwich,
                freq=freq,
                unconditional=unconditional,
            ),
        )
        return pd.DataFrame(cov, index=coef_names, columns=coef_names)

    @overload
    def partial_effect(
        self,
        term: AbstractTerm | int,
        target: str | None = None,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def partial_effect(
        self,
        term: AbstractTerm | int,
        target: str | None = None,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: Literal[True],
    ) -> FitAndSE[np.ndarray]: ...

    def partial_effect(
        self,
        term: AbstractTerm | int,
        target: str | None = None,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: bool = False,
    ) -> np.ndarray | FitAndSE[np.ndarray]:
        """Compute the partial effect for a single model term.

        This method efficiently computes the contribution of one specific term
        to the model predictions.

        Args:
            term: The specific term to evaluate (must match a term used in the
                original model specification) or an integer index representing
                the position of the term in the target's predictor list
            target: Name of the target variable from the keys of ``gam.predictors``. If
                set to None, the single predictor is used if only one is present,
                otherwise an error is raised.
            data: DataFrame or dictionary containing the variables needed
                to compute the partial effect for the term.
            compute_se: Whether to compute and return standard errors
        """
        if self.fit_state is None:
            raise ValueError(
                "Cannot compute partial effect before fitting the model.",
            )

        if target is None:
            if len(self.predictors) > 1:
                raise ValueError(
                    "Target must be specified when multiple predictors are present.",
                )
            target = list(self.predictors.keys())[0]

        if isinstance(term, int):
            term = self.predictors[target][term]

        if data is not None:
            self._check_data(data, requires=term)

        data = data if data is not None else self.fit_state.data

        formula_idx = list(self.predictors.keys()).index(target)

        if compute_se:
            return term._partial_effect_with_se(
                data=data,
                rgam=self.fit_state.rgam,
                formula_idx=formula_idx,
            )
        return term._partial_effect(
            data=data,
            rgam=self.fit_state.rgam,
            formula_idx=formula_idx,
        )

    def edf(self) -> pd.Series:
        """Compute the effective degrees of freedom (EDF) for the model coefficients.

        Returns:
            A series of EDF values, with the mgcv-style coefficient names as the index.
        """
        if self.fit_state is None:
            raise ValueError("Model must be fit before computing the EDF.")
        edf = self.fit_state.rgam.rx2["edf"]
        return pd.Series(to_py(edf), index=to_py(edf.names))

    def penalty_edf(self):
        """Computed the effective degrees of freedom (EDF) associated with each penalty.

        Returns:
            A series of EDF values, with the index being the mgcv-style name of the
            penalty.
        """
        if self.fit_state is None:
            raise ValueError("Model must be fit before computing penalty EDFs.")
        edf = rmgcv.pen_edf(self.fit_state.rgam)
        return pd.Series(to_py(edf), index=to_py(edf.names))

    def partial_residuals(
        self,
        term: AbstractTerm | int,
        target: str | None = None,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        avoid_scaling: bool = False,
    ) -> np.ndarray:
        """Compute partial residuals for model diagnostic plots.

        Partial residuals combine the fitted values from a specific term with
        the overall model residuals. They're useful for assessing whether the
        chosen smooth function adequately captures the relationship, or if a
        different functional form might be more appropriate.

        Args:
            term: The model term to compute partial residuals for. If an integer,
                it is interpreted as the index of the term in the predictor of
                ``target``.
            target: Name of the target variable (response variable or family
                parameter name from the model specification). If set to None, an error
                is raised when multiple predictors are present; otherwise, the sole
                available target is used.
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            avoid_scaling: If True, and the term has a numeric by variable,
                the scaling by the by variable is not included in the term effect.
                This facilitates plotting the residuals, as the plots
                only show the smooth component (unscaled by the by variable).

        Returns:
            Series containing the partial residuals for the specified term
        """
        if self.fit_state is None:
            raise ValueError(
                "Cannot compute partial residuals before fitting the model.",
            )

        if target is None:
            if len(self.predictors) > 1:
                raise ValueError(
                    "Target must be specified when multiple predictors are present.",
                )
            target = list(self.predictors.keys())[0]

        if isinstance(term, int):
            term = self.predictors[target][term]

        link_fit = self.predict(data)[target]  # _check_data called within predict
        data = data if data is not None else self.fit_state.data
        data = deepcopy(data)

        if np.shape(link_fit) != np.shape(data[target]):
            # e.g. Binomial family with matrix of counts as input.
            raise ValueError(
                "Cannot compute partial residuals if the target variable shape does "
                "not match the shape of the fitted values.",
            )

        if (
            term.by is not None
            and is_numeric_dtype(data[term.by].dtype)
            and avoid_scaling
        ):
            data = dict(data) if isinstance(data, Mapping) else data
            data[term.by] = np.full(data_len(data), 1)

        term_effect = self.partial_effect(term=term, target=target, data=data)

        response_residual = data[target] - self.family.inverse_link(link_fit)

        # We want to transform residuals to link scale.
        # link(response) - link(response_fit) not sensible: poisson + log link -> log(0)
        # Instead use first order taylor expansion of link function around the fit
        d_mu_d_eta = self.family.dmu_deta(link_fit)
        d_mu_d_eta = np.maximum(d_mu_d_eta, 1e-6)  # Numerical stability

        # If ĝ is the first order approxmation to link, below is:
        # ĝ(response) - ĝ(response_fit)
        link_residual = response_residual / d_mu_d_eta
        return link_residual + term_effect

    @overload
    def _format_predictions(
        self,
        predictions,
        *,
        compute_se: Literal[False],
    ) -> dict[str, np.ndarray]: ...

    @overload
    def _format_predictions(
        self,
        predictions,
        *,
        compute_se: Literal[True],
    ) -> dict[str, FitAndSE[np.ndarray]]: ...

    def _format_predictions(
        self,
        predictions,
        *,
        compute_se: bool,
    ) -> dict[str, np.ndarray] | dict[str, FitAndSE[np.ndarray]]:
        """Formats output from mgcv predict."""
        all_targets = self.predictors.keys()
        if compute_se:
            fit_all = to_py(predictions.rx2["fit"]).reshape(-1, len(all_targets))
            se_all = to_py(predictions.rx2["se.fit"]).reshape(-1, len(all_targets))
            return {
                k: FitAndSE(fit=fit_all[:, i], se=se_all[:, i])
                for i, k in enumerate(all_targets)
            }
        fit_all = to_py(predictions).reshape(-1, len(all_targets))
        return {k: v for k, v in zip(all_targets, fit_all.T, strict=True)}

    def _format_partial_effects(
        self,
        predictions,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        *,
        compute_se: bool,
    ) -> dict[str, pd.DataFrame] | dict[str, FitAndSE[pd.DataFrame]]:
        """Formats output from mgcv predict with type="terms" (i.e. partial effects)."""
        if compute_se:
            fit_raw = pd.DataFrame(
                to_py(predictions.rx2["fit"]),
                columns=to_py(rbase.colnames(predictions.rx2["fit"])),
            )
            se_raw = pd.DataFrame(
                to_py(predictions.rx2["se.fit"]),
                columns=to_py(rbase.colnames(predictions.rx2["se.fit"])),
            )
        else:
            fit_raw = pd.DataFrame(
                to_py(predictions),
                columns=to_py(rbase.colnames(predictions)),
            )
            se_raw = None

        # Partition results based on formulas
        results = {}
        predictors = (
            self.predictors
            if not self.add_intercepts
            else _add_intercepts(self.predictors)
        )

        for i, (target, terms) in enumerate(predictors.items()):
            fit = {}
            se = {}

            for term in terms:
                label = term.label()
                identifier = term.mgcv_identifier(i)

                if term.by is not None and data[term.by].dtype == "category":
                    levels = pd.Series(data[term.by]).cat.categories.to_list()
                    cols = [f"{identifier}{lev}" for lev in levels]
                    fit[label] = fit_raw[cols].sum(axis=1)

                    if se_raw is not None:
                        se[label] = se_raw[cols].sum(axis=1)

                elif identifier in fit_raw.columns:
                    fit[label] = fit_raw[identifier]

                    if se_raw is not None:
                        se[label] = se_raw[identifier]

                else:  # Offset + Intercept
                    partial_effect = self.partial_effect(
                        term,
                        target,
                        data,
                        compute_se=compute_se,
                    )
                    if isinstance(partial_effect, FitAndSE):
                        fit[label] = partial_effect.fit
                        se[label] = partial_effect.se
                    else:
                        fit[label] = partial_effect

            if compute_se:
                results[target] = FitAndSE(
                    fit=pd.DataFrame(fit),
                    se=pd.DataFrame(se),
                )
            else:
                results[target] = pd.DataFrame(fit)
        return results

    def aic(self, k: float = 2) -> float:
        """Calculate Akaike's Information Criterion for fitted GAM models.

        Where possible (fitting [`GAM`][pymgcv.gam.GAM]/[`BAM`][pymgcv.gam.BAM]
        models with "ML" or "REML"), this uses the approach of Wood, Pya & Saefken 2016,
        which accounts for smoothing parameter uncertainty, without favouring
        overly simple models.

        Args:
            k: Penalty per parameter (default 2 for classical AIC).
        """
        if self.fit_state is None:
            raise ValueError("Cannot compute AIC before fitting.")
        res = rstats.AIC(self.fit_state.rgam, k=k)
        return res[0]  # type: ignore[index]

    def residuals(
        self,
        type: Literal[
            "deviance",
            "pearson",
            "scaled.pearson",
            "working",
            "response",
        ] = "deviance",
    ):
        r"""Compute the residuals for a fitted model.

        Args:
            type: Type of residuals to compute, one of:

                - **response**: Raw residuals $y - \mu$, where $y$ is the observed data
                    and $\mu$ is the model fitted value.
                - **pearson**: Pearson residuals — raw residuals divided by the square
                    root of the model's mean-variance relationship.
                    $$
                    \frac{y - \mu}{\sqrt{V(\mu)}}
                    $$
                - **scaled.pearson**: Raw residuals divided by the standard deviation of
                    the data according to the model mean variance relationship and
                    estimated scale parameter.
                - **deviance**: Deviance residuals as defined by the model’s family.
                - **working**: Working residuals are the residuals returned from
                    model fitting at convergence.
        """
        if self.fit_state is None:
            raise ValueError("Cannot compute residuals before fitting.")
        return to_py(rstats.residuals(self.fit_state.rgam, type=type))

    def residuals_from_y_and_fit(
        self,
        *,
        y: np.ndarray,
        fit: np.ndarray,
        weights: np.ndarray | None = None,
        type: Literal[
            "deviance",
            "pearson",
            "scaled.pearson",
            "working",
            "response",
        ] = "deviance",
    ):
        """Compute the residuals, from y, fit and optionally prior weights."""
        # For now just overwrite attributes and use residuals(gam)
        # From scanning gam.residuals code this looks like it should be reasonable?
        gam = deepcopy(self)
        if gam.fit_state is None:
            raise ValueError("Cannot compute residuals before fitting the model.")

        if weights is None:
            weights = np.ones_like(y)
        gam.fit_state.rgam.rx2["y"] = to_rpy(y)
        gam.fit_state.rgam.rx2["fitted.values"] = to_rpy(fit)
        gam.fit_state.rgam.rx2["prior.weights"] = to_rpy(weights)
        return gam.residuals(type)


@dataclass(init=False, repr=False)  # use AbstractGAM init and repr
class GAM(AbstractGAM):
    """Standard GAM Model."""

    predictors: dict[str, list[AbstractTerm]]
    family: AbstractFamily
    add_intercepts: bool
    fit_state: FitState | None

    def fit(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *,
        method: GAMFitMethods = "REML",
        weights: str | np.ndarray | pd.Series | None = None,
        optimizer: str | tuple[str, str] = ("outer", "newton"),
        scale: Literal["unknown"] | float | int | None = None,
        select: bool = False,
        gamma: float | int = 1,
        knots: dict[str, np.ndarray] | None = None,
        n_threads: int = 1,
    ) -> Self:
        """Fit the GAM.

        Args:
            data: DataFrame or dictionary containing all variables referenced in the
                model. Note, using a dictionary is required when passing matrix-valued
                variables.
            method: Method for smoothing parameter estimation, matching the mgcv
                options.
            weights: Observation weights. Either a string, matching a column name,
                or an array/series with length equal to the number of observations.
            optimizer: An string or length 2 tuple, specifying the numerical
                optimization method to use to optimize the smoothing parameter
                estimation criterion (given by method). "outer" for the direct nested
                optimization approach. "outer" can use several alternative optimizers,
                specified in the second element: "newton" (default), "bfgs", "optim" or
                "nlm". "efs" for the extended Fellner Schall method of Wood and Fasiolo
                (2017).
            scale: If a number is provided, it is treated as a known scale parameter.
                If left to None, the scale parameter is 1 for Poisson and binomial and
                unknown otherwise. Note that (RE)ML methods can only work with scale
                parameter 1 for the Poisson and binomial cases.
            select: If set to True then gam can add an extra penalty to each term so
                that it can be penalized to zero. This means that the smoothing
                parameter estimation during fitting can completely remove terms
                from the model. If the corresponding smoothing parameter is estimated as
                zero then the extra penalty has no effect. Use gamma to increase level
                of penalization.
            gamma: Increase this beyond 1 to produce smoother models. gamma multiplies
                the effective degrees of freedom in the GCV or UBRE/AIC. gamma can be
                viewed as an effective sample size in the GCV score, and this also
                enables it to be used with REML/ML. Ignored with P-RE/ML or the efs
                optimizer.
            knots: Dictionary mapping covariate names to knot locations.
                For most bases, the length of the knot locations should match with a
                user supplied `k` value. E.g. for `S("x", k=64)`, you could
                pass `knots={"x": np.linspace(0, 1, 64)}`. For multidimensional
                smooths, e.g. `S("x", "z", k=64)`, you could create a grid of
                coordinates:

                !!! example

                    ```python
                        import numpy as np
                        coords = np.linspace(0, 1, num=8)
                        X, Z = np.meshgrid(coords, coords)
                        knots = {"x": X.ravel(), "z": Z.ravel()}
                    ```
                Note if using
                [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline], this will
                avoid the eigen-decomposition used to find the basis, which although
                fast often leads to worse results. Different terms can use different
                numbers of knots, unless they share covariates.
            n_threads: Number of threads to use for fitting the GAM.
        """
        # TODO some missing options: control, sp, min.sp etc
        self._check_data(data, requires="all")
        if isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data[self.referenced_variables])
        else:
            data = {k: v for k, v in data.items() if k in self.referenced_variables}

        data = deepcopy(data)
        weights = data[weights] if isinstance(weights, str) else weights  # type: ignore
        r_knots = (
            ro.NULL
            if knots is None
            else ro.ListVector({k: to_rpy(pos) for k, pos in knots.items()})
        )
        rgam = rmgcv.gam(
            self._to_r_formulae(),
            data=data_to_rdf(data, include=self.referenced_variables),
            family=self.family.rfamily,
            method=method,
            weights=ro.NULL if weights is None else np.asarray(weights),
            optimizer=to_rpy(np.array(optimizer)),
            scale=0 if scale is None else (-1 if scale == "unknown" else scale),
            select=select,
            gamma=gamma,
            knots=r_knots,
            nthreads=n_threads,
        )
        self.fit_state = FitState(
            rgam=rgam,
            data=data,
        )
        return self

    @overload
    def predict(
        self,
        *,
        compute_se: Literal[False] = False,
        type: Literal["response", "link"] = "link",
        block_size: int | None = None,
    ) -> dict[str, np.ndarray]: ...

    @overload
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: Literal[True],
        type: Literal["response", "link"] = "link",
        block_size: int | None = None,
    ) -> dict[str, FitAndSE[np.ndarray]]: ...

    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: bool = False,
        type: Literal["response", "link"] = "link",
        block_size: int | None = None,
    ) -> dict[str, np.ndarray] | dict[str, FitAndSE[np.ndarray]]:
        """Compute model predictions with (optionally) uncertainty estimates.

        Makes predictions for new data using the fitted GAM model. Predictions
        are returned on the link scale (linear predictor scale), not the response
        scale. For response scale predictions, apply the appropriate inverse link
        function to the results.

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            compute_se: Whether to compute standard errors for predictions.
            type: Type of prediction to compute. Either "link" for linear predictor
                scale or "response" for response scale.
            block_size: Number of rows to process at a time.  If None then block size
                is 1000 if data supplied, and the number of rows in the model frame
                otherwise.

        Returns:
            A dictionary mapping the target variable names to a pandas DataFrame
            containing the predictions and standard errors if `se` is True.
        """
        if data is not None:
            self._check_data(data, requires="covariates")

        if self.fit_state is None:
            raise RuntimeError("Cannot call predict before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg
            if data is None
            else data_to_rdf(data, include=self.referenced_variables),
            se=compute_se,
            type=type,
            block_size=ri.MissingArg if block_size is None else block_size,
        )
        return self._format_predictions(
            predictions,
            compute_se=compute_se,
        )

    @overload
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: Literal[False] = False,
        block_size: int | None = None,
    ) -> dict[str, pd.DataFrame]: ...

    @overload
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: Literal[True],
        block_size: int | None = None,
    ) -> dict[str, FitAndSE[pd.DataFrame]]: ...

    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
        *,
        compute_se: bool = False,
        block_size: int | None = None,
    ) -> dict[str, pd.DataFrame] | dict[str, FitAndSE[pd.DataFrame]]:
        """Compute partial effects for all model terms.

        Calculates the contribution of each model term to the overall prediction on the
        link scale. The sum of all fit columns equals the total prediction (link scale).

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            compute_se: Whether to compute and return standard errors.
            block_size: Number of rows to process at a time.  If None then block size
                is 1000 if data supplied, and the number of rows in the model frame
                otherwise.
        """
        if data is not None:
            self._check_data(data, requires="covariates")

        if self.fit_state is None:
            raise RuntimeError("Cannot call partial_effects before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg
            if data is None
            else data_to_rdf(data, include=self.referenced_variables),
            se=compute_se,
            type="terms",
            newdata_gauranteed=True,
            block_size=ri.MissingArg if block_size is None else block_size,
        )
        return self._format_partial_effects(
            predictions,
            data if data is not None else self.fit_state.data,
            compute_se=compute_se,
        )


@dataclass(init=False, repr=False)  # use AbstractGAM init and repr
class BAM(AbstractGAM):
    """A big-data GAM (BAM) model."""

    predictors: dict[str, list[AbstractTerm]]
    family: AbstractFamily
    add_intercepts: bool
    fit_state: FitState | None

    def fit(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
        *,
        method: BAMFitMethods = "fREML",
        weights: str | np.ndarray | pd.Series | None = None,
        scale: Literal["unknown"] | float | int | None = None,
        select: bool = False,
        gamma: float | int = 1,
        knots: dict[str, np.ndarray] | None = None,
        chunk_size: int = 10000,
        discrete: bool = False,
        samfrac: float | int = 1,
        n_threads: int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> Self:
        """Fit the GAM.

        Args:
            data: DataFrame or dictionary containing all variables referenced in the
                model. Note, using a dictionary is required when passing matrix-valued
                variables.
            method: Method for smoothing parameter estimation, matching the mgcv,
                options.
            weights: Observation weights. Either a string, matching a column name,
                or a array/series with length equal to the number of observations.
            scale: If a number is provided, it is treated as a known scale parameter.
                If left to None, the scale parameter is 1 for Poisson and binomial and
                unknown otherwise. Note that (RE)ML methods can only work with scale
                parameter 1 for the Poisson and binomial cases.
            select: If set to True then gam can add an extra penalty to each term so
                that it can be penalized to zero. This means that the smoothing
                parameter estimation during fitting can completely remove terms
                from the model. If the corresponding smoothing parameter is estimated as
                zero then the extra penalty has no effect. Use gamma to increase level
                of penalization.
            gamma: Increase this beyond 1 to produce smoother models. gamma multiplies
                the effective degrees of freedom in the GCV or UBRE/AIC. gamma can be
                viewed as an effective sample size in the GCV score, and this also
                enables it to be used with REML/ML. Ignored with P-RE/ML or the efs
                optimizer.
            knots: Dictionary mapping covariate names to knot locations.
                For most bases, the length of the knot locations should match with a
                user supplied `k` value. E.g. for `S("x", k=64)`, you could
                pass `knots={"x": np.linspace(0, 1, 64)}`. For multidimensional
                smooths, e.g. `S("x", "z", k=64)`, you could create a grid of
                coordinates:

                !!! example

                    ```python
                        import numpy as np
                        coords = np.linspace(0, 1, num=8)
                        X, Z = np.meshgrid(coords, coords)
                        knots = {"x": X.ravel(), "z": Z.ravel()}
                    ```
                Note if using
                [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline], this will
                avoid the eigen-decomposition used to find the basis, which although
                fast often leads to worse results. Different terms can use different
                numbers of knots, unless they share covariates.
            chunk_size: The model matrix is created in chunks of this size, rather than
                ever being formed whole. Reset to 4*p if chunk.size < 4*p where p is the
                number of coefficients.
            discrete: if True and using method="fREML", discretizes covariates for
                storage and efficiency reasons.
            samfrac: If ``0<samfrac<1``, performs a fast preliminary fitting step using
                a subsample of the data to improve convergence speed.
            n_threads: Number of threads to use for fitting the GAM.
            gc_level: 0 uses R's garbage collector, 1 and 2 use progressively
                more frequent garbage collection, which takes time but reduces
                memory requirements.
        """
        # TODO some missing options: control, sp, min.sp, nthreads
        self._check_data(data, requires="all")
        if isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data[self.referenced_variables])
        else:
            data = {k: v for k, v in data.items() if k in self.referenced_variables}

        weights = data[weights] if isinstance(weights, str) else weights  # type: ignore
        r_knots = (
            ro.NULL
            if knots is None
            else ro.ListVector({k: to_rpy(pos) for k, pos in knots.items()})
        )
        self.fit_state = FitState(
            rgam=rmgcv.bam(
                self._to_r_formulae(),
                data=data_to_rdf(data, include=self.referenced_variables),
                family=self.family.rfamily,
                method=method,
                weights=ro.NULL if weights is None else np.asarray(weights),
                scale=0 if scale is None else (-1 if scale == "unknown" else scale),
                select=select,
                gamma=gamma,
                knots=r_knots,
                chunk_size=chunk_size,
                discrete=discrete,
                samfrac=samfrac,
                nthreads=n_threads,
                gc_level=gc_level,
            ),
            data=data,
        )
        return self

    @overload
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: Literal[False] = False,
        type: Literal["link", "response"] = "link",
        block_size: int = 50000,
        discrete: bool = True,
        n_threads: int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, np.ndarray]: ...

    @overload
    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: Literal[True],
        type: Literal["link", "response"] = "link",
        block_size: int = 50000,
        discrete: bool = True,
        n_threads: int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, FitAndSE[np.ndarray]]: ...

    def predict(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: bool = False,
        type: Literal["link", "response"] = "link",
        block_size: int = 50000,
        discrete: bool = True,
        n_threads: int = 1,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, FitAndSE[np.ndarray]] | dict[str, np.ndarray]:
        """Compute model predictions with uncertainty estimates.

        Makes predictions for new data using the fitted GAM model. Predictions
        are returned on the link scale (linear predictor scale), not the response
        scale. For response scale predictions, apply the appropriate inverse link
        function to the results.

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            compute_se: Whether to compute and return standard errors.
            type: Type of prediction to compute. Either "link" for linear predictor
                scale or "response" for response scale.
            block_size: Number of rows to process at a time.
            n_threads: Number of threads to use for computation.
            discrete: If True and the model was fitted with discrete=True, then
                uses discrete prediction methods in which covariates are
                discretized for efficiency for storage and efficiency reasons.
            gc_level: 0 uses R's garbage collector, 1 and 2 use progressively
                more frequent garbage collection, which takes time but reduces
                memory requirements.
        """
        if data is not None:
            self._check_data(data, requires="covariates")

        if self.fit_state is None:
            raise RuntimeError("Cannot call predict before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg
            if data is None
            else data_to_rdf(data, include=self.referenced_variables),
            se=compute_se,
            type=type,
            block_size=ro.NULL if block_size is None else block_size,
            discrete=discrete,
            n_threads=n_threads,
            gc_level=gc_level,
        )

        return self._format_predictions(
            predictions,
            compute_se=compute_se,
        )

    @overload
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: Literal[False] = False,
        block_size: int = 50000,
        n_threads: int = 1,
        discrete: bool = True,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, pd.DataFrame]: ...

    @overload
    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: Literal[True],
        block_size: int = 50000,
        n_threads: int = 1,
        discrete: bool = True,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, FitAndSE[pd.DataFrame]]: ...

    def partial_effects(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
        *,
        compute_se: bool = False,
        block_size: int = 50000,
        n_threads: int = 1,
        discrete: bool = True,
        gc_level: Literal[0, 1, 2] = 0,
    ) -> dict[str, pd.DataFrame] | dict[str, FitAndSE[pd.DataFrame]]:
        """Compute partial effects for all model terms.

        Calculates the contribution of each model term to the overall prediction.
        This decomposition is useful for understanding which terms contribute most
        to predictions and for creating partial effect plots. The sum of all fit columns
        equals the total prediction.

        Args:
            data: A dictionary or DataFrame containing all variables referenced in the
                model. Defaults to the data used to fit the model.
            compute_se: Whether to compute and return standard errors.
            block_size: Number of rows to process at a time. Higher is faster
                but more memory intensive.
            n_threads: Number of threads to use for computation.
            discrete: If True and the model was fitted with discrete=True, then
                uses discrete prediction methods in which covariates are
                discretized for efficiency for storage and efficiency reasons.
            gc_level: 0 uses R's garbage collector, 1 and 2 use progressively
                more frequent garbage collection, which takes time but reduces
                memory requirements.
        """
        if data is not None:
            self._check_data(data, requires="covariates")

        if self.fit_state is None:
            raise RuntimeError("Cannot call partial_effects before fitting.")

        predictions = rstats.predict(
            self.fit_state.rgam,
            ri.MissingArg
            if data is None
            else data_to_rdf(data, include=self.referenced_variables),
            se=compute_se,
            type="terms",
            newdata_gauranteed=True,
            block_size=ri.MissingArg if block_size is None else block_size,
            n_threads=n_threads,
            discrete=discrete,
            gc_level=gc_level,
        )
        return self._format_partial_effects(
            predictions,
            data=self.fit_state.data if data is None else data,
            compute_se=compute_se,
        )


def _add_intercepts(predictors: dict[str, list[AbstractTerm]]):
    result = predictors.copy()
    for target, terms in predictors.items():
        if not any(isinstance(t, Intercept) for t in terms):
            result[target].append(Intercept())
    return result
