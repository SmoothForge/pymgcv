"""A collection of GAM test cases."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import rpy2.robjects as ro

import pymgcv.basis_functions as bs
from pymgcv import families as fam
from pymgcv import terms
from pymgcv.families import Poisson
from pymgcv.gam import BAM, GAM, AbstractGAM
from pymgcv.rpy_utils import data_to_rdf, to_py
from pymgcv.terms import Intercept, L, Offset, S, T


def get_method_default(gam_type: type[AbstractGAM]):
    """Returns the pymgcv default fitting method for the gams."""
    sig = inspect.signature(gam_type.fit)
    return sig.parameters["method"].default


@cache
def get_test_data() -> pd.DataFrame:
    """Simple toy test dataset for testing most models.

    Used as default if data is not specified in test cases.
    """
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    x1 = x + rng.normal(loc=2, scale=1.5, size=n)
    group = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    group1 = pd.Categorical(rng.choice(["E", "F", "G"], size=n))
    group_effect = pd.Series(group).map({"A": 0.0, "B": 1.0, "C": -1.0}).to_numpy()
    y = 2 + 0.5 * x - 0.3 * x1 + group_effect + rng.normal(scale=0.5, size=n)
    y1 = np.sin(x) + 0.1 * x1**2 + group_effect + rng.normal(scale=0.3, size=n)
    pos_int = rng.poisson(lam=np.exp(0.2 * x + 0.1 * x1 + 0.3 * group_effect))
    pos_float = (
        np.exp(
            1 + 0.4 * x - 0.2 * x1 + 0.5 * group_effect + rng.normal(scale=0.2, size=n),
        )
    ) + 0.1
    binary = rng.binomial(1, p=(1 + np.tanh(x)) / 2)
    prob = rng.beta(a=1, b=2, size=n)
    ordered_cat = pd.Categorical(
        rng.choice(["Low", "Medium", "High"], size=n),
        categories=["Low", "Medium", "High"],
        ordered=True,
    )
    zero_to_two_int = rng.integers(low=0, high=3, size=n)
    one_to_three_int = rng.integers(low=1, high=4, size=n)

    obj = [("a", "b")] * n
    return pd.DataFrame(
        {
            "y": y,
            "y1": y1,
            "x": x,
            "x1": x1,
            "pos_int": pos_int,
            "pos_float": pos_float,
            "group": group,
            "group1": group1,
            "binary": binary,
            "prob": prob,
            "ordered_cat": ordered_cat,
            "zero_to_two_int": zero_to_two_int,
            "one_to_three_int": one_to_three_int,
            "unused": obj,  # Check object column doesn't prevent conversion to R df
        },
    )


@dataclass(kw_only=True)
class GAMTestCase:  # GAM/BAM test cases
    """Test cases for GAMs.

    Note the fitting method will often need to be specified in `mgcv_args`, as pymgcv
    changes the default to `REML` for `GAM`. Note `data=data` is automatically included
    in `mgcv_args`.
    """

    mgcv_args: str
    gam_model: AbstractGAM
    add_to_r_env: dict[str, ro.RObject] = field(default_factory=dict)
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] = field(
        default_factory=get_test_data,
    )
    fit_kwargs: dict[str, Any] = field(default_factory=dict)

    def mgcv_gam(
        self,
        data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
    ) -> ro.ListVector:
        """Returns the mgcv gam object."""
        with ro.local_context() as env:
            env["data"] = data_to_rdf(data, include=self.gam_model.referenced_variables)
            for k, v in self.add_to_r_env.items():
                env[k] = v
            result = ro.r(self.mgcv_call)
            assert isinstance(result, ro.ListVector)
            return result

    @property
    def mgcv_call(self):
        """Returns the mgcv gam call as a string."""
        return (
            f"{self.gam_model.__class__.__name__.lower()}({self.mgcv_args},data=data)"
        )


# Factory functions for test cases
def linear_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    gam = gam_type({"y": L("x")})
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~x, method='{method}'",
        gam_model=gam,
    )


def categorical_linear_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~group, method='{method}'",
        gam_model=gam_type({"y": L("group")}),
    )


def smooth_1d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x), method='{method}'",
        gam_model=gam_type({"y": S("x")}),
    )


def smooth_with_specified_knots(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    # We need exact same floating representation
    knots = to_py(ro.r("(0:4)/4"))
    return GAMTestCase(
        mgcv_args=f"y~s(x, k=5), method='{method}',knots=list(x=(0:4)/4)",
        gam_model=gam_type({"y": S("x", k=5)}),
        fit_kwargs={"knots": {"x": knots}},
    )


def smooth_2d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x, x1), method='{method}'",
        gam_model=gam_type({"y": S("x", "x1")}),
    )


def smooth_2d_m_and_xt(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    basis = bs.ThinPlateSpline(max_knots=3, m=2)
    return GAMTestCase(
        mgcv_args=f"y~s(x,x1,m=2, xt=list(max.knots=3)), method='{method}'",
        gam_model=gam_type({"y": S("x", "x1", bs=basis)}),
    )


def tensor_2d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~te(x, x1), method='{method}'",
        gam_model=gam_type({"y": T("x", "x1")}),
    )


def tensor_interaction_2d_gam_with_mc(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~ti(x, x1, mc=c(TRUE, FALSE)), method='{method}'",
        gam_model=gam_type(
            {"y": T("x", "x1", mc=[True, False], interaction_only=True)},
        ),
    )


def random_effect_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x) + s(group, bs='re'), method='{method}'",
        gam_model=gam_type({"y": S("x") + S("group", bs=bs.RandomEffect())}),
    )


def categorical_interaction_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~group:group1, method='{method}'",
        gam_model=gam_type({"y": terms.Interaction("group", "group1")}),
    )


def offset_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x) + offset(x1), method='{method}'",
        gam_model=gam_type({"y": S("x") + Offset("x1")}),
    )


def smooth_1d_by_categorical_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x, by=group), method='{method}'",
        gam_model=gam_type({"y": S("x", by="group")}),
    )


def smooth_1d_by_numeric_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x, by=x1), method='{method}'",
        gam_model=gam_type({"y": S("x", by="x1")}),
    )


def tensor_2d_by_categorical_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~te(x,x1, by=group), method='{method}'",
        gam_model=gam_type({"y": T("x", "x1", by="group")}),
    )


def tensor_2d_by_numeric_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~te(x,x1,by=pos_int), method='{method}'",
        gam_model=gam_type({"y": T("x", "x1", by="pos_int")}),
    )


def smooth_1d_random_wiggly_curve_gam(
    gam_type: type[AbstractGAM] = GAM,
) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~s(x,group,bs='fs',xt=list(bs='cr')),method='{method}'",
        gam_model=gam_type(
            {"y": S("x", "group", bs=bs.FactorSmooth(bs.CubicSpline()))},
        ),
    )


def tensor_2d_random_wiggly_curve_gam(
    gam_type: type[AbstractGAM] = GAM,
) -> GAMTestCase:
    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y~t(x,x1,group,bs='fs'),method='{method}'",
        gam_model=gam_type({"y": T("x", "x1", "group", bs=bs.FactorSmooth())}),
    )


def markov_random_field_polys(gam_type: type[AbstractGAM]) -> GAMTestCase:
    countries = ["a", "b", "c", "d", "e", "f", "g"]
    polys: dict[str, np.ndarray] = {
        "a": np.array([[0, 2], [1, 2], [1, 1], [0, 1]]),
        "b": np.array([[1, 2], [2, 2], [2, 1], [1, 1]]),
        "c": np.array([[2, 2], [3, 2], [3, 1], [2, 1]]),
        "d": np.array([[0, 1], [1, 1], [1, 0], [0, 0]]),
        "e": np.array([[1, 1], [2, 1], [2, 0], [1, 0]]),
        "f": np.array([[2, 1], [3, 1], [3, 0], [2, 0]]),
        "g": np.array([[1, 0], [2, 0], [2, -1], [1, -1]]),
    }

    r_polys = (
        "list(a=matrix(c(0,2,1,2,1,1,0,1),ncol=2,byrow=T),"
        "b=matrix(c(1,2,2,2,2,1,1,1),ncol=2,byrow=T),"
        "c=matrix(c(2,2,3,2,3,1,2,1),ncol=2,byrow=T),"
        "d=matrix(c(0,1,1,1,1,0,0,0),ncol=2,byrow=T),"
        "e=matrix(c(1,1,2,1,2,0,1,0),ncol=2,byrow=T),"
        "f=matrix(c(2,1,3,1,3,0,2,0),ncol=2,byrow=T),"
        "g=matrix(c(1,0,2,0,2,-1,1,-1),ncol=2,byrow=T))"
    )

    rng = np.random.default_rng(1)
    countries = pd.Categorical(rng.choice(countries, size=100), categories=countries)

    data = {
        "country": countries,
        "y": rng.normal(size=100),
    }

    return GAMTestCase(
        mgcv_args=f"y~s(country,bs='mrf',xt=list(polys={r_polys})),method='REML'",
        gam_model=gam_type(
            {
                "y": S("country", bs=bs.MarkovRandomField(polys=polys)),  # type: ignore
            },
        ),
        data=data,
    )


def markov_random_field_neighbours(gam_type: type[AbstractGAM]) -> GAMTestCase:
    countries = ["a", "b", "c", "d", "e", "f", "g"]
    neighbours = {
        "a": ["b", "c"],
        "b": ["a", "c", "d"],
        "c": ["a", "b", "d", "e"],
        "d": ["b", "c", "e", "f"],
        "e": ["c", "d", "f", "g"],
        "f": ["d", "e", "g"],
        "g": ["e", "f"],
    }
    r_nb = (
        "list(a=c('b','c'),b=c('a','c','d'),c=c('a','b','d','e'),"
        "d=c('b','c','e','f'),e=c('c','d','f','g'),f=c('d','e','g'),g=c('e','f'))"
    )
    rng = np.random.default_rng(1)
    countries = pd.Categorical(rng.choice(countries, size=100), categories=countries)
    data = {
        "country": pd.Series(countries),
        "y": rng.normal(size=100),
    }

    return GAMTestCase(
        mgcv_args=f"y~s(country,bs='mrf',xt=list(nb={r_nb})),method='REML'",
        gam_model=gam_type(
            {
                "y": S("country", bs=bs.MarkovRandomField(neighbours=neighbours)),
            },
        ),
        data=data,
    )


# TODO: Test integer too?


def linear_functional_smooth_1d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    rng = np.random.default_rng(123)
    n = 200
    n_hours = 24
    hourly_x = rng.lognormal(size=(n, n_hours))
    y = sum(np.sqrt(col) for col in hourly_x.T) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "hourly_x": hourly_x}
    gam = gam_type({"y": S("hourly_x")})

    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y ~ s(hourly_x), method='{method}'",
        gam_model=gam,
        data=data,
    )


def linear_functional_tensor_2d_gam(gam_type: type[AbstractGAM]) -> GAMTestCase:
    rng = np.random.default_rng(123)
    n = 200
    n_times = 4
    x0 = rng.lognormal(size=(n, n_times))
    x1 = rng.lognormal(size=(n, n_times))

    def _true_fn(x0, x1):
        return np.sqrt(x0) + np.sqrt(x1)

    y = sum(
        _true_fn(x0_col, x1_col) for x0_col, x1_col in zip(x0.T, x1.T, strict=True)
    ) + rng.normal(scale=0.1, size=n)
    data = {"y": y, "x0": x0, "x1": x1}
    gam = gam_type({"y": T("x0", "x1")})

    method = get_method_default(gam_type)
    return GAMTestCase(
        mgcv_args=f"y ~ te(x0, x1), method='{method}'",
        gam_model=gam,
        data=data,
    )


def spline_test_cases() -> dict[str, GAMTestCase]:  # TODO maybe add other basis types
    bases = {
        "bs='tp'": bs.ThinPlateSpline(),
        "bs='cr'": bs.CubicSpline(),
        "bs='ds', m=c(1,0)": bs.DuchonSpline(m=1),
        "bs='bs'": bs.BSpline(),
        "bs='bs', m=c(3,2,1)": bs.BSpline(degree=3, penalty_orders=[2, 1]),
        "bs='ps', m=c(2,2)": bs.PSpline(degree=3, penalty_order=2),
    }
    test_cases = {}
    for k, v in bases.items():
        test_cases[k] = GAMTestCase(
            mgcv_args=f"y ~ s(x, {k}), method='REML'",
            gam_model=GAM({"y": S("x", bs=v)}),
        )
    return test_cases


def family_test_cases() -> dict[str, GAMTestCase]:
    """Mostly for testing families, so we just test with GAM."""
    # Gaussian excluded as used extensively as default
    rng = np.random.default_rng(1)
    two_counts_data = {  # for binomial which allows array input
        "counts": rng.poisson(10, size=(100, 2)),
        "x": rng.normal(size=100),
    }
    return {
        "Binomial - binary": GAMTestCase(
            mgcv_args="binary ~ s(x), family='binomial',method='REML'",
            gam_model=GAM({"binary": S("x")}, family=fam.Binomial()),
        ),
        "Binomial - counts": GAMTestCase(
            mgcv_args="counts ~ s(x), family='binomial',method='REML'",
            gam_model=GAM({"counts": S("x")}, family=fam.Binomial()),
            data=two_counts_data,
        ),
        "Gamma": GAMTestCase(
            mgcv_args="pos_float ~ s(x), family='Gamma',method='REML'",
            gam_model=GAM({"pos_float": S("x")}, family=fam.Gamma()),
        ),
        "InverseGaussian": GAMTestCase(
            mgcv_args="pos_float~s(x),family='inverse.gaussian',method='REML'",
            gam_model=GAM({"pos_float": S("x")}, family=fam.InverseGaussian()),
        ),
        "Poisson": GAMTestCase(
            mgcv_args="pos_int~s(x), family=poisson, method='REML'",
            gam_model=GAM({"pos_int": S("x")}, family=Poisson()),
        ),
        "Quasi": GAMTestCase(
            mgcv_args="pos_float~s(x),family=quasi(link='log',variance='mu^2'),method='REML'",
            gam_model=GAM(
                {"pos_float": S("x")},
                family=fam.Quasi(link="log", variance="mu^2"),
            ),
        ),
        "QuasiBinomial": GAMTestCase(
            mgcv_args="binary ~ s(x), family=quasibinomial,method='REML'",
            gam_model=GAM({"binary": S("x")}, family=fam.QuasiBinomial()),
        ),
        "QuasiPoisson": GAMTestCase(
            mgcv_args="pos_int ~ s(x), family=quasipoisson,method='REML'",
            gam_model=GAM({"pos_int": S("x")}, family=fam.QuasiPoisson()),
        ),
        # # mgcv families
        "MVN": GAMTestCase(
            mgcv_args="list(y ~ s(x), y1 ~ s(x1)), family=mvn(d=2),method='REML'",
            gam_model=GAM({"y": S("x"), "y1": S("x1")}, family=fam.MVN(d=2)),
        ),
        "GauLSS": GAMTestCase(
            mgcv_args="list(y~s(x),~s(x1)), family=gaulss,method='REML'",
            gam_model=GAM(
                {
                    "y": S("x"),
                    "scale": S("x1"),
                },
                family=fam.GauLSS(),
            ),
        ),
        "Betar": GAMTestCase(
            mgcv_args="prob ~ s(x), family=betar(1, eps=1e-10),method='REML'",
            gam_model=GAM({"prob": S("x")}, family=fam.Betar(phi=1)),
        ),
        "GumbLS": GAMTestCase(
            mgcv_args="list(prob~s(x),~s(x1)), family=gumbls,method='REML'",
            gam_model=GAM(
                {
                    "prob": S("x"),
                    "log_scale": S("x1"),
                },
                family=fam.GumbLS(),
            ),
        ),
        "NegativeBinomial": GAMTestCase(
            mgcv_args="pos_int~s(x), family=nb(theta=1),method='REML'",
            gam_model=GAM(
                {"pos_int": S("x")},
                family=fam.NegativeBinomial(theta=1, theta_fixed=True),
            ),
        ),
        "OCat": GAMTestCase(
            mgcv_args="one_to_three_int~s(x), family=ocat(R=3),method='REML'",
            gam_model=GAM(
                {"one_to_three_int": S("x")},
                family=fam.OCat(num_categories=3),
            ),
        ),
        "Scat": GAMTestCase(
            mgcv_args="pos_float~s(x), family=scat,method='REML'",
            gam_model=GAM({"pos_float": S("x")}, family=fam.Scat()),
        ),
        "Tweedie": GAMTestCase(
            mgcv_args=(
                "pos_float~s(x), family=Tweedie(1.5, link=power(0.1)),method='REML'"
            ),
            gam_model=GAM({"pos_float": S("x")}, family=fam.Tweedie(p=1.5, link=0.1)),
        ),
        "Tw": GAMTestCase(
            mgcv_args="pos_float~s(x), family=tw,method='REML'",
            gam_model=GAM({"pos_float": S("x")}, family=fam.Tw()),
        ),
        "ZIP": GAMTestCase(
            mgcv_args="pos_int~s(x), family=ziP,method='REML'",
            gam_model=GAM({"pos_int": S("x")}, family=fam.ZIP()),
        ),
        "GammaLS": GAMTestCase(
            mgcv_args="list(pos_float~s(x),~s(x1)), family=gammals,method='REML'",
            gam_model=GAM(
                {
                    "pos_float": S("x"),
                    "log_scale": S("x1"),
                },
                family=fam.GammaLS(),
            ),
        ),
        "GevLSS": GAMTestCase(
            mgcv_args="list(pos_float~s(x),~s(x),~s(x1)), family=gevlss,method='REML'",
            gam_model=GAM(
                {
                    "pos_float": S("x"),
                    "log_scale": S("x"),
                    "shape": S("x1"),
                },
                family=fam.GevLSS(),
            ),
        ),
        "Multinom": GAMTestCase(  # This is a bit clunky
            mgcv_args=(
                "list(zero_to_two_int~s(x),~s(x)), family=multinom(K=2),method='REML'"
            ),
            gam_model=GAM(
                {
                    "zero_to_two_int": S("x"),
                    "eta2": S("x"),
                },
                family=fam.Multinom(k=2),
            ),
        ),
        "Shash": GAMTestCase(
            mgcv_args="list(y~s(x),~s(x),~1,~1), family=shash,method='REML'",
            gam_model=GAM(
                {
                    "y": S("x"),
                    "log_scale": S("x"),
                    "shape": Intercept(),
                    "kurtosis": Intercept(),
                },
                family=fam.Shash(),
            ),
        ),
    }


def get_test_cases() -> dict[str, GAMTestCase]:
    supported_types_and_cases = [
        (
            (GAM, BAM),
            [
                linear_gam,
                categorical_linear_gam,
                smooth_1d_gam,
                smooth_2d_gam,
                smooth_2d_m_and_xt,
                smooth_with_specified_knots,
                tensor_2d_gam,
                tensor_interaction_2d_gam_with_mc,
                random_effect_gam,
                smooth_1d_random_wiggly_curve_gam,
                categorical_interaction_gam,
                offset_gam,
                smooth_1d_by_categorical_gam,
                smooth_1d_by_numeric_gam,
                tensor_2d_by_categorical_gam,
                tensor_2d_by_numeric_gam,
                linear_functional_smooth_1d_gam,
                linear_functional_tensor_2d_gam,
                markov_random_field_neighbours,
                markov_random_field_polys,
            ],
        ),
    ]

    test_cases = {}
    for gam_types, cases in supported_types_and_cases:
        for gam_type in gam_types:
            for case in cases:
                test_cases[f"{gam_type.__name__} - {case.__name__}"] = case(gam_type)

    return test_cases | spline_test_cases() | family_test_cases()
