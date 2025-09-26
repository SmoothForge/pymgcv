"""Basis functions for smooth terms in GAM models.

This module provides various basis function types that can be used with smooth
terms to control the shape and properties of the estimated smooth functions.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import rpy2.robjects as ro

from pymgcv.formula_utils import _to_r_constructor_string, _Var
from pymgcv.rpy_utils import to_rpy

# The design here is a bit strange and warrants some explanation. We want an intuitive
# interface. For this reason, we handle xt and m within the basis, whereas mgcv uses
# these as arguments to the s function (meaning different things depending on the
# basis). These arguments need to be internally forwared to s, within a formula context.
# The two most obvious solutions are converting to a string representation of R code,
# or assigning them to an R variable. We choose to do the latter, which is probably
# more robust/efficient/simple for complex cases, (e.g. where xt is a list of arrays).
# Here however, we just provide the methods for getting the xt/m as the appropriate R
# type, then we handle them in smooths


class AbstractBasis(ABC):
    """Abstract class defining the interface for GAM basis functions.

    All basis function classes must implement this protocol to be usable
    with smooth terms. The protocol ensures basis functions can be converted
    to appropriate mgcv R syntax and provide any additional parameters needed.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Convert basis to mgcv string identifier.

        Returns:
            String identifier used by mgcv (e.g., 'tp', 'cr', 'bs')
        """
        ...

    @abstractmethod
    def _get_xt(self) -> None | ro.ListVector | _Var:
        """Get the xt argument for the smooth (or None if not needed)."""
        ...

    @abstractmethod
    def _get_m(self) -> int | float | ro.Vector | None:
        """Get the m argument for the smooth (or None if not needed)."""
        ...

    def _m_and_xt_args(self) -> list[str]:
        """Arguments to pass to s for this basis.

        Each argument is a string respresentation, e.g. ["m=1", "xt=c(1,2)"].
        """
        args = []
        for k, v in {"xt": self._get_xt(), "m": self._get_m()}.items():
            if v is not None:
                args.append(f"{k}={_to_r_constructor_string(v)}")
        return args


@dataclass
class RandomEffect(AbstractBasis):
    """Random effect basis for correlated grouped data.

    This can be used with any mixture of numeric or categorical variables. Acts
    similarly to an [`Interaction`][pymgcv.terms.Interaction] but penalizes
    the corresponding coefficients with a multiple of the identity matrix (i.e. a ridge
    penalty), corresponding to an assumption of i.i.d. normality of the parameters.

    !!! warning

        Numeric variables (int/float), will be treated as a linear term with a single
        penalized slope parameter. Do not use an integer variable to encode
        categorical groups!

    !!! example

        For an example, see the
        [supplement vs placebo example](../examples/supplement_vs_placebo.ipynb).

    """

    def __str__(self) -> str:
        """Return mgcv identifier for random effects."""
        return "re"

    def _get_xt(self) -> None | ro.ListVector | _Var:
        return None

    def _get_m(self) -> int | float | ro.IntVector | ro.FloatVector | None:
        return None


@dataclass(kw_only=True, frozen=True)
class ThinPlateSpline(AbstractBasis):
    """Thin plate regression spline basis.

    Args:
        shrinkage: If True, the penalty is modified so that the term is shrunk to zero
            for a high enough smoothing parameter.
        m: The order of the derivative in the thin plate spline penalty. If $d$ is the
            number of covariates for the smooth term, this must satisfy $m>(d+1)/2$. If
            left to None, the smallest value satisfying $m>(d+1)/2$ will be used, which
            creates "visually smooth" functions.
        max_knots: The maximum number of knots to use. Defaults to 2000.
    """

    shrinkage: bool | None = False
    m: int | None = None
    max_knots: int | None = None

    def __str__(self) -> str:
        """Return mgcv identifier: 'ts' for shrinkage, 'tp' for standard."""
        return "ts" if self.shrinkage else "tp"

    def _get_xt(self) -> ro.ListVector | None | _Var:
        if self.max_knots is not None:
            return ro.ListVector({"max.knots": self.max_knots})
        return None

    def _get_m(self):
        return self.m


@dataclass
class FactorSmooth(AbstractBasis):
    """S for each level of a categorical variable.

    When using this basis, the first variable of the smooth should
    be a numeric variable, and the second should be a categorical variable.

    Unlike using a categorical by variable e.g. `S(x, by="group")`:

    - The terms share a smoothing parameter.
    - The terms are fully penalized, with seperate penalties on each null space
        component (e.g. intercepts). The terms are non-centered, and can
        be used with an intercept without introducing indeterminacy, due to the
        penalization.

    Args:
        bs: Any singly penalized basis function. Defaults to
            `ThinPlateSpline`. Only the type of the basis is passed
            to mgcv (i.e. what is returned by `str(bs)`). This is a limitation
            of mgcv (e.g. you cannot do )
            mgcv provides no way to pass more details for setting up the
            basis function.
    """

    bs: AbstractBasis = field(default_factory=ThinPlateSpline)

    def __str__(self) -> str:
        """Return mgcv identifier for random effects."""
        return "fs"

    def _get_xt(self) -> None | ro.ListVector | _Var:
        listvec = ro.ListVector({"bs": str(self.bs)})
        xt = self.bs._get_xt()
        if xt is not None:
            if isinstance(xt, _Var):
                xt_value = ro.globalenv[xt.name]
                combined = xt_value + listvec
                varname = f".xt_{id(self)}"
                ro.globalenv[varname] = combined
                return _Var(varname)
            return xt + listvec
        return listvec

    def _get_m(self) -> int | float | ro.Vector | None:
        return self.bs._get_m()


@dataclass(kw_only=True)
class CubicSpline(AbstractBasis):
    """Cubic regression spline basis.

    Cubic splines use piecewise cubic polynomials with knots placed throughout
    the data range. They tend to be computationally efficient, but often
    performs slightly worse than thin plate splines and are limited to
    univariate smooths. Note the limitation of being restricted to
    one-dimensional smooths does not imply they cannot be used for
    multivariate [`T`][pymgcv.terms.T] smooths,
    which are constructed from marginal bases.

    Args:
        cyclic: If True, creates a cyclic spline where the function values
            and derivatives match at the boundaries. Use for periodic data
            like time of day, angles, or seasonal patterns. Default is False.
        shrinkage: If True, adds penalty to the null space (linear component).
            Helps with model selection and identifiability. Default is False.
            Cannot be used with cyclic=True.

    Raises:
        ValueError: If both cyclic and shrinkage are True (incompatible options)
    """

    shrinkage: bool = False
    cyclic: bool = False

    def __post_init__(self):
        """Validate cubic spline configuration."""
        if self.cyclic and self.shrinkage:
            raise ValueError("Cannot use both cyclic and shrinkage simultaneously.")

    def __str__(self) -> str:
        """Return mgcv identifier: 'cs', 'cc', or 'cr'."""
        return "cs" if self.shrinkage else "cc" if self.cyclic else "cr"

    def _get_m(self) -> int | float | ro.Vector | None:
        return None

    def _get_xt(self) -> None | ro.ListVector | _Var:
        return None


@dataclass(kw_only=True)
class DuchonSpline(AbstractBasis):
    """Duchon spline basis - a generalization of thin plate splines.

    These smoothers allow the use of lower orders of derivative in the penalty than
    conventional thin plate splines, while still yielding continuous functions.

    The description, adapted from mgcv is as follows: Duchon’s (1977) construction
    generalizes the usual thin plate spline penalty as follows. The usual thin plate
    spline penalty is given by the integral of the squared Euclidian norm of a vector of
    mixed partial $m$-th order derivatives of the function w.r.t. its arguments. Duchon
    re-expresses this penalty in the Fourier domain, and then weights the squared norm
    in the integral by the Euclidean norm of the fourier frequencies, raised to the
    power $2s$, where $s$ is a user selected constant.

    If $d$ is the number of arguments of the smooth:

    - It is required that $-d/2 < s < d/2$.
    - If $s=0$ then the usual thin plate spline is recovered.
    - To obtain continuous functions we further require that $m + s > d/2$.

    For example, ``DuchonSpline(m=1, s=d/2)`` can be used in order to use first
    derivative penalization for any $d$, and still yield continuous functions.

    Args:
        m : Order of derivative to penalize.
        s : $s$ as described above, should be an integer divided by 2.
    """

    m: int = 2
    s: float | int = 0

    def __str__(self) -> str:
        """Return mgcv identifier for Duchon splines."""
        return "ds"

    def _get_m(self) -> int | float | ro.Vector | None:
        return ro.FloatVector([self.m, self.s])

    def _get_xt(self) -> None | ro.ListVector | _Var:
        return None


@dataclass(kw_only=True)
class SplineOnSphere(AbstractBasis):
    """Isotropic smooth for data on a sphere (latitude/longitude coordinates).

    This should be used with exactly two variables, where the first represents latitude
    on the interval [-90, 90] and the second represents longitude on the interval [-180,
    180].

    Args:
        m : An integer in [-1, 4]. Setting `m=-1` uses
            [`DuchonSpline(m=2,s=1/2)`](`pymgcv.basis_functions.DuchonSpline`). Setting
            `m=0` signals to use the 2nd order spline on the sphere, computed by
            Wendelberger’s (1981) method. For m>0, (m+2)/2 is the penalty order, with
            m=2 equivalent to the usual second derivative penalty.
    """

    m: int = 0

    def __str__(self) -> str:
        """Return mgcv identifier for splines on sphere."""
        return "sos"

    def _get_m(self) -> int | float | ro.Vector | None:
        return self.m

    def _get_xt(self) -> None | ro.ListVector | _Var:
        return None


@dataclass
class BSpline(AbstractBasis):
    """B-spline basis with derivative-based penalties.

    These are univariate (but note univariate smooths can be used for multivariate
    smooths constructed with [`T`][pymgcv.terms.T]).
    ``BSpline(degree=3, penalty_orders=[2])`` constructs a conventional cubic spline.

    Args:
        degree: The degree of the B-spline basis (e.g. 3 for a cubic spline).
        penalty_orders: The derivative orders to penalize. Default to [degree - 1].
    """

    degree: int
    penalty_orders: list[int]

    def __init__(self, *, degree: int = 3, penalty_orders: Iterable[int] | None = None):
        if penalty_orders is None:
            penalty_orders = [degree - 1]
        self.degree = degree
        self.penalty_orders = list(penalty_orders)

    def __str__(self) -> str:
        """Return mgcv identifier for B-splines."""
        return "bs"

    def _get_m(self) -> int | float | ro.Vector | None:
        return ro.IntVector([self.degree] + self.penalty_orders)

    def _get_xt(self) -> None | ro.ListVector | _Var:
        return None


@dataclass(kw_only=True)
class PSpline(AbstractBasis):
    """P-spline (penalized spline) basis as proposed by Eilers and Marx (1996).

    Uses B-spline bases penalized by discrete penalties applied directly to the basis
    coefficients. Note for most use cases splines with derivative-based penalties (e.g.
    [`ThinPlateSpline`][pymgcv.basis_functions.ThinPlateSpline] or
    [`CubicSpline`][pymgcv.basis_functions.CubicSpline]) tend to yield better
    MSE performance. ``BSpline(degree=3, penalty_order=2)`` is
    cubic-spline-like.

    Args:
        degree: Degree of the B-spline basis (e.g. 3 for cubic).
        penalty_order: The difference order to penalize. 0-th order is ridge penalty.
            Default to `degree-1`.
    """

    cyclic: bool = False
    degree: int
    penalty_order: int

    def __init__(self, *, degree: int = 3, penalty_order: int | None = None):
        self.degree = degree
        self.penalty_order = penalty_order if penalty_order is not None else degree - 1

    def __str__(self) -> str:
        """Return mgcv identifier: 'cp' for cyclic, 'ps' for standard."""
        return "cp" if self.cyclic else "ps"

    def _get_m(self) -> int | float | ro.Vector | None:
        # Note (unlike b-splines) seems mgcv uses m[1] for the penalty order, not degree so subtract 1
        return ro.IntVector([self.degree - 1, self.penalty_order])

    def _get_xt(self) -> None | ro.ListVector | _Var:
        return None


class MarkovRandomField(AbstractBasis):
    """Intrinsic Gaussian Markov random field for discrete spatial data.

    The smoothing penalty encourages similar value in neighboring locations. The
    variable used in the corresponding smooth should be a categorical variable
    with strings represenging the area labels.

    Should be constructed using either polygons or neighborhood structure. For
    plotting, the polygon structure is required.

    !!! example

        For an example, see the
        [markov random field crime example](../examples/markov_random_field_crime.ipynb).

    Args:
        polys: Dictionary mapping levels of the categorical variable to the
            polygons structure arrays. Each array should have two columns,
            representing the coordinates of the vertices.
        neighbours: Dictionary mapping levels of the categorical variable to
            a list or numpy array of strings corresponding neighbours for that level.
    """

    polys: dict[str, np.ndarray] | None
    neighbours: dict[str, np.ndarray] | None

    def __init__(
        self,
        *,
        polys: dict[str, np.ndarray] | None = None,
        neighbours: dict[str, np.ndarray] | dict[str, list[str]] | None = None,
    ):
        neither = polys is None and neighbours is None
        both = polys is not None and neighbours is not None

        if neither | both:
            raise ValueError("Exactly one of polys or neighbours must be provided.")

        self.polys = polys
        self.neighbours = (
            {k: np.asarray(v) for k, v in neighbours.items()}
            if neighbours is not None
            else None
        )

    def __str__(self) -> str:
        """Return mgcv identifier for Markov Random Fields."""
        return "mrf"

    def __repr__(self):
        """Simplified representation (avoiding displaying arrays)."""
        kw = "polys" if self.polys is not None else "neighbours"
        return f"MarkovRandomField({kw}=dict)"

    def _get_m(self) -> int | float | ro.Vector | None:
        return None

    def _get_xt(self) -> None | ro.ListVector | _Var:
        if self.polys is None:
            assert self.neighbours is not None
            kw = "nb"
            dict_ = self.neighbours
        else:
            kw = "polys"
            dict_ = self.polys
        polys_or_nb = ro.ListVector({k: to_rpy(v) for k, v in dict_.items()})
        polys_or_nb = ro.ListVector({kw: polys_or_nb})
        # Only really practical to assign to a variable here.
        varname = f".xt_{id(self)}"
        ro.globalenv[varname] = polys_or_nb
        return _Var(varname)
