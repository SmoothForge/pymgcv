"""The available terms for constructing GAM models."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import rpy2.robjects as ro

from pymgcv.basis_functions import AbstractBasis
from pymgcv.custom_types import FitAndSE
from pymgcv.formula_utils import _to_r_constructor_string, _Var
from pymgcv.rlibs import rbase, rmgcv, rstats
from pymgcv.rpy_utils import data_to_rdf, is_null, to_py
from pymgcv.utils import data_len

# TODO: Not supporting 'sp' or 'pc' basis types.


class AbstractTerm(ABC):
    """Abstract class defining the interface for GAM model terms.

    Attributes:
        varnames: Tuple of variable names used by this term. For univariate terms,
            this contains a single variable name. For multivariate terms (like
            tensor smooths), this contains multiple variable names.
        by: Optional name of a 'by' variable that scales this term.
    """

    varnames: tuple[str, ...]
    by: str | None  # TODO move to AbstractSmooth?

    @abstractmethod
    def __str__(self) -> str:
        """Convert the term to mgcv formula syntax."""
        pass

    @abstractmethod
    def label(self) -> str:
        """The label used by pymgcv for the term in plotting and columns.

        All labels must be unique in a formula. Labels should be implemented
        such that each unique label must map to a unique mgcv identifier (but
        not necessarily the other way around).
        """
        ...

    @abstractmethod
    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Generate the mgcv identifier for the term.

        When computing partial effects, we look for a column with this
        name from the mgcv output, and if not we fall back on using
        [`partial_effect`][pymgcv.gam.AbstractGAM.partial_effect].

        !!! example

            ```python
            from pymgcv.terms import S
            assert S("x1").mgcv_identifier(formula_idx=1) == "s.1(x1)"
            ```

        Args:
            formula_idx: Index of the formula in multi-formula models.
        """
        ...

    @abstractmethod
    def _partial_effect(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> np.ndarray:
        """Compute partial effects for this term.

        Args:
            data: DataFrame containing predictor variables.
            rgam: Fitted mgcv gam.
            formula_idx: The formula index of the term.
        """
        ...

    @abstractmethod
    def _partial_effect_with_se(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> FitAndSE[np.ndarray]:
        """Compute partial effects and standard errors for this term.

        Args:
            data: DataFrame containing predictor variables.
            rgam: Fitted mgcv gam.
            formula_idx: The formula index of the term.
        """
        ...

    def __add__(self, other) -> list:
        """Allow adding of terms to terms and lists."""
        if isinstance(other, list):
            return [self] + other
        if isinstance(other, AbstractTerm):
            return [self, other]
        return NotImplemented

    def __radd__(self, other):
        """Allow adding of terms to terms and lists."""
        if isinstance(other, list):
            return other + [self]
        if isinstance(other, AbstractTerm):
            return [other, self]
        return NotImplemented


class AbstractParametric(AbstractTerm):
    """Abstract class for parametric terms e.g. Linear and Interaction."""

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> np.ndarray:
        model_matrix, coefs = self._get_model_matrix_and_coefs(
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )
        return to_py(model_matrix @ coefs).squeeze(axis=-1)

    def _partial_effect_with_se(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> FitAndSE[np.ndarray]:
        model_matrix, coefs = self._get_model_matrix_and_coefs(
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )
        fitted_vals = to_py(model_matrix @ coefs).squeeze(axis=-1)
        cov = rgam.rx2["Vp"]
        cov.rownames = rstats.coef(rgam).names
        cov.colnames = rstats.coef(rgam).names
        subcov = cov.rx(coefs.names, coefs.names)
        se = rbase.sqrt(rbase.rowSums((model_matrix @ subcov).ro * model_matrix))
        return FitAndSE(fitted_vals, to_py(se))

    def _get_model_matrix_and_coefs(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> tuple[Any, Any]:
        if isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data[list(self.varnames)])
        else:
            data = {k: v for k, v in data.items() if k in self.varnames}
        predict_mat = rstats.model_matrix(
            ro.Formula(f"~{str(self)}-1"),
            data=data_to_rdf(data, include="all"),
        )
        post_fix = "" if formula_idx == 0 else f".{formula_idx}"
        predict_mat.colnames = rbase.paste0(predict_mat.colnames, post_fix)
        coef_names = rbase.intersect(predict_mat.colnames, rstats.coef(rgam).names)
        predict_mat = rbase.as_matrix(predict_mat.rx(True, coef_names))  # noqa: FBT003
        coefs = rstats.coef(rgam).rx(coef_names)
        return predict_mat, coefs


@dataclass
class L(AbstractParametric):
    """Linear (parametric) term with no basis expansion.

    - If the variable if continuous, the term will be included in the model matrix
      as is, with a single corresponding coefficient.
    - If the variable is a categorical variable, the term will be expanded (one-hot
      encoded into a set of dummy variables.

    Args:
        name: Name of the variable to include as a linear term.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        return self.varnames[0]

    def label(self) -> str:
        """Returns the label, e.g. 'L(x)'."""
        return f"L({self.varnames[0]})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return term identifier with optional formula index."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return self.varnames[0] + idx


@dataclass
class Interaction(AbstractParametric):
    """Parametric interaction term between multiple variables.

    Any categorical variables involved in an interaction are expanded into indicator
    variables representing all combinations at the specified interaction order.
    Numeric variables are incorporated by multiplication (i.e. with eachother and
    any indicator variables).

    Note, this does not automatically include main effects or lower order interactions.

    Args:
        *varnames: Variable names to include in the interaction. Can be any
            number of variables.

    !!! example

        ```python
        # Two-way interaction (multiplication if both numeric)
        from pymgcv.terms import Interaction
        age_income = Interaction('age', 'income')

        # Three-way interaction
        varnames = ['group0', 'group1', 'group2']
        three_way = Interaction(*varnames)

        # Generate all pairwise interactions
        from itertools import combinations
        pairs = [Interaction(*pair) for pair in combinations(varnames, 2)]
        ```

    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, *varnames: str):
        """Initialize an interaction term.

        Args:
            *varnames: Names of variables to include in the interaction.
                Must be 2 or more variables.
        """
        if len(varnames) < 2:
            raise ValueError("Interaction terms require at least 2 variables")
        self.varnames = tuple(varnames)
        self.by = None

    def __str__(self) -> str:
        """Return interaction in mgcv formula syntax (colon-separated)."""
        return ":".join(self.varnames)

    def label(self):
        """The label for the interaction term."""
        return f"Interaction({','.join(self.varnames)})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return interaction identifier with optional formula index."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        return ":".join(self.varnames) + idx


class AbstractSmooth(AbstractTerm):
    """Abstract class for smooth terms (including tensor smooths)."""

    bs: AbstractBasis | None

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> np.ndarray:
        result = self._partial_effect_shared(data, rgam, formula_idx, compute_se=False)
        assert isinstance(result, np.ndarray)
        return result

    def _partial_effect_with_se(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> FitAndSE[np.ndarray]:
        result = self._partial_effect_shared(data, rgam, formula_idx, compute_se=True)
        assert isinstance(result, FitAndSE)
        return result

    def _partial_effect_shared(
        self,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
        *,
        compute_se: bool,
    ) -> np.ndarray | FitAndSE[np.ndarray]:
        # Share an implementation with compute_se True/False
        data = deepcopy(data)
        smooth_name = self.mgcv_identifier(formula_idx)
        smooths = {s.rx2["label"][0]: s for s in rgam.rx2["smooth"]}

        is_categorical_by = self.by is not None and isinstance(
            data[self.by].dtype,
            pd.CategoricalDtype,
        )

        if not is_categorical_by:
            return self._mgcv_smooth_prediction(
                mgcv_smooth=smooths[smooth_name],
                data=data,
                rgam=rgam,
                compute_se=compute_se,
            )

        levels = data[self.by].cat.categories  # type: ignore
        n = data_len(data)

        if compute_se:
            result = FitAndSE(fit=np.empty(n), se=np.empty(n))
        else:
            result = np.empty(n)

        for lev in levels:
            assert self.by is not None
            is_lev = data[self.by] == lev
            if isinstance(data, pd.DataFrame):
                data_lev = pd.DataFrame(data[is_lev])
            else:
                data_lev = {k: v[is_lev] for k, v in data.items()}
            if len(data_lev) == 0:
                continue
            smooth = smooths[smooth_name + lev]
            level_prediction = self._mgcv_smooth_prediction(
                mgcv_smooth=smooth,
                data=data_lev,
                rgam=rgam,
                compute_se=compute_se,
            )
            if isinstance(result, FitAndSE):
                assert isinstance(level_prediction, FitAndSE)
                result.fit[is_lev] = level_prediction.fit
                result.se[is_lev] = level_prediction.se
            else:
                assert isinstance(level_prediction, np.ndarray)
                result[is_lev] = level_prediction

        return result

    def _mgcv_smooth_prediction(
        self,
        *,
        mgcv_smooth: Any,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        compute_se: bool,
    ) -> np.ndarray | FitAndSE:
        """Compute prediction and standard error for a smooth given data."""
        include = list(self.varnames)

        if self.by is not None:
            include.append(self.by)

        predict_mat = rmgcv.PredictMat(mgcv_smooth, data_to_rdf(data, include=include))
        first = round(mgcv_smooth.rx2["first.para"][0])
        last = round(mgcv_smooth.rx2["last.para"][0])
        coefs = rstats.coef(rgam)[(first - 1) : last]
        pred = to_py(rbase.drop(predict_mat @ coefs))
        if compute_se:
            cov = rbase.as_matrix(
                rgam.rx2["Vp"].rx(rbase.seq(first, last), rbase.seq(first, last)),
            )
            se = rbase.sqrt(rbase.rowSums((predict_mat @ cov).ro * predict_mat))
            se = to_py(rbase.pmax(0, se))
            return FitAndSE(pred, se)
        return pred

    def _add_xt_and_m_to_renv(self) -> list[str]:
        """Adds xt and m to the r env and returns a list string arguments."""
        args = []
        if self.bs is not None:
            xt, m = self.bs._get_xt(), self.bs._get_m()
            for k, v in {"xt": xt, "m": m}.items():
                if v is not None:
                    varname = f".{k}_{id(self)}"  # id to avoid conflicts
                    ro.globalenv[varname] = xt
                    args.append(f"{k}={varname}")
        return args


@dataclass
class S(AbstractSmooth):
    """Smooth term.

    For all the arguments, passing None will use the mgcv defaults.

    !!! Note

        For multiple variables, this creates an **isotropic** smooth, meaning all
        variables are treated on the same scale. If variables have very different scales
        or units, consider using [`T`][pymgcv.terms.T].

    Args:
        *varnames: Names of variables to smooth over. For single variables,
            creates a univariate smooth. For multiple variables, creates an
            isotropic multi-dimensional smooth.
        k: The dimension of the basis used to represent the smooth term. The
            default depends on the basis and number of variables that the smooth is a
            function of. For choosing k, see [`check_k`][pymgcv.gam.AbstractGAM.check_k].
        bs: Basis function. For available options see
            [Basis Functions](./basis_functions.md). If left to none, uses
            [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline].
        by: variable name used to scale the smooth. If it's a numeric vector, it
            scales the smooth, and the "by" variable should not be included as a
            seperate main effect (as the smooth is usually not centered). If the "by" is
            a categorical variable, a separate smooth is created for each factor level.
            In this case the smooths are centered, so the categorical variable should
            be included as a main effect.
        id: Identifier for grouping smooths with shared penalties. If using a
            categorical by variable, providing an id will ensure a shared smoothing
            parameter for each level.
        fx: Indicates whether the term is a fixed d.f. regression spline (True) or a
            penalized regression spline (False). Defaults to False.
    """

    varnames: tuple[str, ...]
    by: str | None
    k: int | None
    bs: AbstractBasis | None
    id: int | None
    fx: bool

    def __init__(
        self,
        *varnames: str,
        by: str | None = None,
        k: int | None = None,
        bs: AbstractBasis | None = None,
        id: int | None = None,
        fx: bool = False,
    ):
        if len(varnames) == 0:
            raise ValueError("S terms require at least one variable")
        self.varnames = varnames
        self.k = k
        self.bs = bs
        self.by = by
        self.id = id
        self.fx = fx

    def __str__(self) -> str:
        """Convert smooth term to mgcv formula syntax."""
        kwargs = asdict(self)
        kwargs["fx"] = self.fx if self.fx else None
        varnames = kwargs.pop("varnames")
        if self.bs is not None:
            kwargs = kwargs | {"xt": self.bs._get_xt(), "m": self.bs._get_m()}
        kwargs["by"] = _Var(self.by) if self.by is not None else None
        kwargs["bs"] = str(self.bs) if self.bs is not None else None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwarg_strings = [
            f"{k}={_to_r_constructor_string(v)}" for k, v in kwargs.items()
        ]
        kwarg_string = "" if len(kwarg_strings) == 0 else f",{','.join(kwarg_strings)}"
        return f"s({','.join(varnames)}{kwarg_string})"

    def label(self) -> str:
        """The label for the term, e.g. S(x,y,by=group)."""
        by = f",by={self.by}" if self.by else ""
        return f"S({','.join(self.varnames)}{by})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Generate simplified smooth identifier."""
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        mgcv_identifier = f"s{idx}({','.join(self.varnames)})"

        if self.by is not None:
            mgcv_identifier += f":{self.by}"

        return mgcv_identifier


# A far more intuitive interface would be to specify a list of smooths.
# But that deviates pretty far from the mgcv way.
@dataclass
class T(AbstractSmooth):
    """Tensor product smooth for scale-invariant multi-dimensional smoothing.

    Tensor smooths create smooth functions of multiple variables using marginal
    smooths in order to be robust to variables on different scales.
    For the sequence arguments, the length must match the number of variables if
    ``d`` is not provided, else they must match the length of ``d``.

    Args:
        *varnames: Names of variables for the tensor smooth.
        k: The basis dimension for each marginal smooth. If an integer, all
            marginal smooths will have the same basis dimension.
            For choosing k, see [`check_k`][pymgcv.gam.AbstractGAM.check_k]
        bs: basis type to use, or an iterable of basis types for each marginal
            smooth. Defaults to [`CubicSpline`][pymgcv.basis_functions.CubicSpline]
        d: Sequence specifying the dimension of each variable's smooth. For example,
            (2, 1) would specify to use one two dimensional marginal smooth and one
            1 dimensional marginal smooth, where three variables are provided. This
            is useful for space-time smooths (2 dimensional space and 1 time
            dimension).
        by: Variable name for 'by' variable scaling the tensor smooth, or creating
            a smooth for each level of a categorical by variable.
        id: Identifier for sharing penalties across multiple tensor smooths.
        fx: indicates whether the term is a fixed d.f. regression spline (True) or
            a penalized regression spline (False). Defaults to False.
        np: If False, use a single penalty for the tensor product.
            If True (default), use separate penalties for each marginal. Defaults
            to True.
        mc: If `interaction_only` is `True`, this specifies the marginals which should
            have centering constraints applied. The length should match the number of
            marginals.
        interaction_only: If True, creates ti() instead of te() - interaction only,
            excluding main effects of individual variables.
    """

    varnames: tuple[str, ...]
    by: str | None
    k: int | tuple[int, ...] | None
    bs: AbstractBasis | tuple[AbstractBasis, ...] | None
    d: tuple[int, ...] | None
    id: int | None
    fx: bool
    np: bool
    interaction_only: bool
    mc: list[bool] | None

    def __init__(
        self,
        *varnames: str,
        by: str | None = None,
        k: int | Iterable[int] | None = None,
        bs: AbstractBasis | Iterable[AbstractBasis] | None = None,
        d: Iterable[int] | None = None,
        id: int | None = None,
        fx: bool = False,
        np: bool = True,
        mc: Iterable[bool] | None = None,
        interaction_only: bool = False,
    ):
        if len(varnames) < 2:
            raise ValueError("Tensor smooths require at least 2 variables")

        if mc is not None and not interaction_only:
            raise ValueError("mc can only be specified when interaction_only is True.")

        self.varnames = varnames
        self.k = tuple(k) if isinstance(k, Iterable) else k
        self.bs = tuple(bs) if isinstance(bs, Iterable) else bs
        self.d = tuple(d) if d is not None else d
        self.by = by
        self.id = id
        self.fx = fx
        self.np = np
        self.mc = list(mc) if mc is not None else None
        self.interaction_only = interaction_only

    def __str__(self) -> str:
        """Convert tensor smooth to mgcv formula syntax.

        Returns:
            String in mgcv te() or ti() syntax, e.g., "te(x1,x2,k=c(10,15))"
        """

        def _handle_bs(bs):  # Returns bs, m, xt compatible with R
            match bs:
                case AbstractBasis():
                    from_bases = {"xt": bs._get_xt(), "m": bs._get_m()}

                    return str(bs), from_bases.get("m"), from_bases.get("xt")
                case tuple():
                    from_bases = [{"xt": b._get_xt(), "m": b._get_m()} for b in bs]
                    from_bases = [
                        {k: v for k, v in d.items() if v is not None}
                        for d in from_bases
                    ]
                    bs_vec = ro.StrVector([str(b) for b in bs])
                    m = [bs.get("m", ro.NA_Logical) for bs in from_bases]
                    if all(el == ro.NA_Logical for el in m):
                        m = None
                    xt = [bs.get("xt", ro.NULL) for bs in from_bases]
                    if all(is_null(el) for el in xt):
                        xt = None
                    return bs_vec, m, xt
                case _:
                    return None, None, None

        bs, m, xt = _handle_bs(self.bs)
        kwargs = {
            "by": _Var(self.by) if self.by is not None else None,
            "k": ro.IntVector(self.k) if isinstance(self.k, tuple) else self.k,
            "bs": bs,
            "m": m,
            "d": ro.IntVector(self.d) if self.d is not None else None,
            "id": self.id if self.id is not None else None,
            "fx": self.fx if self.fx is True else None,
            "np": self.np if self.np is False else None,
            "xt": xt,
            "mc": None if self.mc is None else np.asarray(self.mc),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwarg_strings = [
            f"{k}={_to_r_constructor_string(v)}" for k, v in kwargs.items()
        ]
        kwarg_string = "" if len(kwarg_strings) == 0 else f",{','.join(kwarg_strings)}"
        prefix = "ti" if self.interaction_only else "te"
        return f"{prefix}({','.join(self.varnames)}{kwarg_string})"

    def label(self) -> str:
        """Label, e.g. "T(x1,x2,by=x3)"."""
        by = f",by={self.by}" if self.by else ""
        return f"T({','.join(self.varnames)}{by})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Generate simplified tensor smooth identifier.

        Returns:
            Simplified identifier like "te(x1,x2)" or "ti.1(x1,x2):by_var"
        """
        idx = "" if formula_idx == 0 else f".{formula_idx}"
        prefix = "ti" if self.interaction_only else "te"
        mgcv_identifier = f"{prefix}{idx}({','.join(self.varnames)})"
        if self.by is not None:
            mgcv_identifier += ":" + self.by
        return mgcv_identifier


@dataclass
class Offset(AbstractTerm):
    """Offset term, added to the linear predictor as is.

    This means:

    - For log-link models: offset induces a multiplicative effect on the response scale
    - For identity-link models: an offset induces an additive effect on the response
        scale

    Args:
        name: Name of the variable to use as an offset. Must be present in the modeling
            data.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self, name: str):
        self.varnames = (name,)
        self.by = None

    def __str__(self) -> str:
        """Return offset in mgcv formula syntax."""
        return f"offset({self.varnames[0]})"

    def label(self) -> str:
        """Label, e.g. "Offset(x)"."""
        return f"Offset({self.varnames[0]})"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return offset identifier.

        Note: mgcv doesn't include offsets as parametric terms in predictions,
        so formula_idx is not used and offsets are handled specially.
        """
        return f"offset({self.varnames[0]})"

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> np.ndarray:
        """Compute offset partial effects.

        For offset terms, the partial effect is simply the offset variable
        values, with zero standard errors.
        """
        return np.asarray(data[self.varnames[0]])

    def _partial_effect_with_se(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> FitAndSE[np.ndarray]:
        """Compute offset partial effects.

        For offset terms, the partial effect is simply the offset variable
        values, with zero standard errors.
        """
        effect = np.asarray(data[self.varnames[0]])
        return FitAndSE(effect, np.zeros_like(effect))


@dataclass
class Intercept(AbstractTerm):
    """Intercept term.

    By default, this is added to all formulas in the model. If you want to control
    the intercept term, then [`GAM`][pymgcv.gam.GAM] should have ``add_intercepts``
    set to `False`, in which case, only intercepts explicitly added will be included
    in the model.
    """

    varnames: tuple[str, ...]
    by: str | None

    def __init__(self):
        self.varnames = ()
        self.by = None

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        return "1"

    def label(self) -> str:
        """The label, "Intercept"."""
        return "Intercept"

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        """Return term identifier with optional formula index."""
        return f"(Intercept){'' if formula_idx == 0 else f'.{formula_idx}'}"

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> np.ndarray:
        n = data_len(data)
        coef = rstats.coef(rgam)
        intercept_name = self.mgcv_identifier(formula_idx)
        return np.full(n, to_py(coef.rx2[intercept_name]).item())

    def _partial_effect_with_se(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> FitAndSE[np.ndarray]:
        intercept = self._partial_effect(data=data, rgam=rgam, formula_idx=formula_idx)
        intercept_name = self.mgcv_identifier(formula_idx)
        cov = rgam.rx2("Vp")
        names = rstats.coef(rgam).names
        cov.rownames, cov.colnames = names, names
        var = cov.rx2(intercept_name, intercept_name)
        se = np.full_like(intercept, np.sqrt(var).item())
        return FitAndSE(intercept, se)


@dataclass
class _FactorSmoothToByInterface(AbstractTerm):
    """This wraps a term using a FactorSmooth basis to a term with a by variable.

    This isn't a real usable term, but provides a consistent interface for plotting
    between categorical by variables and random wiggly curve terms.
    """

    factor_smooth_term: S | T
    by: str | None

    def __init__(self, factor_smooth_term: S | T):
        # TODO edge case of by being set and "fs" basis being used!
        self.factor_smooth_term = factor_smooth_term
        self.by = factor_smooth_term.varnames[-1]
        self.varnames = factor_smooth_term.varnames[:-1]

    def __str__(self) -> str:
        """Return variable name for mgcv formula."""
        raise NotImplementedError()

    def label(self) -> str:
        return self.factor_smooth_term.label()

    def mgcv_identifier(self, formula_idx: int = 0) -> str:
        raise NotImplementedError()

    def _partial_effect(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> np.ndarray:
        return self.factor_smooth_term._partial_effect(
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )

    def _partial_effect_with_se(
        self,
        *,
        data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray],
        rgam: Any,
        formula_idx: int,
    ) -> FitAndSE[np.ndarray]:
        return self.factor_smooth_term._partial_effect_with_se(
            data=data,
            rgam=rgam,
            formula_idx=formula_idx,
        )
