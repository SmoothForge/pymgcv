"""Plotting utilities for visualizing GAM models."""

import types
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm

from pymgcv.basis_functions import FactorSmooth, MarkovRandomField, RandomEffect
from pymgcv.gam import AbstractGAM
from pymgcv.qq import QQResult, qq_simulate
from pymgcv.terms import (
    AbstractTerm,
    L,
    S,
    T,
    _FactorSmoothToByInterface,
)
from pymgcv.utils import data_len


def plot(
    gam: AbstractGAM,
    *,
    ncols: int = 2,
    scatter: bool = False,
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
    to_plot: type | types.UnionType | dict[str, list[AbstractTerm]] = AbstractTerm,
    kwargs_mapper: dict[Callable, dict[str, Any]] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot a gam model.

    Except for some specialised cases, this plots the partial effects of the terms.

    Args:
        gam: The fitted gam object to plot.
        ncols: The number of columns before wrapping axes.
        scatter: Whether to plot the residuals (where possible), and the overlayed
            datapoints on 2D plots. For more fine control, see `kwargs_mapper`. Defaults
            to False.
        data: The data to use for plotting partial residuals and scatter points.
            Will default to using the data used for fitting. Only relevant if
            `scatter=True`.
        to_plot: Which terms to plot. If a type, only plots terms
            of that type (e.g. ``to_plot = S | T`` to plot smooths).
            If a dictionary, it should map the target names to
            an iterable of terms to plot (similar to how models are specified).
        kwargs_mapper: Used to pass keyword arguments to the underlying `pymgcv.plot`
            functions. A dictionary mapping the plotting function to kwargs. For
            example, to disable the confidence intervals on the 1d plots, set
            ``kwargs_mapper`` to
            ```python
            import pymgcv.plot as gplt
            {gplt.continuous_1d: {"fill_between_kwargs": {"disable": True}}}
            ```
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")

    kwargs_mapper = {} if kwargs_mapper is None else kwargs_mapper
    kwargs_mapper.setdefault(categorical, {}).setdefault("residuals", scatter)
    kwargs_mapper.setdefault(continuous_1d, {}).setdefault("residuals", scatter)
    kwargs_mapper.setdefault(continuous_2d, {}).setdefault(
        "scatter_kwargs",
        {},
    ).setdefault("disable", not scatter)

    if isinstance(to_plot, type | types.UnionType):
        to_plot = {
            k: [v for v in terms if isinstance(v, to_plot)]
            for k, terms in gam.predictors.items()
        }

    plotters = []
    for target, terms in to_plot.items():
        for term in terms:
            try:
                plotter = _get_term_plotter(
                    term=term,
                    gam=gam,
                    target=target,
                    data=data,
                )
            except NotImplementedError:
                continue
            plotters.append(plotter)

    n_axs = sum(p.required_axes for p in plotters)
    if n_axs == 0:
        raise ValueError("Do not know how to plot any terms in the model.")

    ncols = min(n_axs, ncols)
    fig, axes = plt.subplots(
        nrows=ceil(n_axs / ncols),
        ncols=ncols,
        layout="constrained",
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    idx = 0
    for plotter in plotters:
        kwargs = kwargs_mapper.get(plotter.underlying_function, {})
        plotter.make_plot(axes[idx : (idx + plotter.required_axes)], **kwargs)

        # If multiple targets in the model, add target variable as title.
        if len(gam.predictors) > 1:
            for ax in axes[idx : (idx + plotter.required_axes)]:
                ax.set_title(plotter.target_variable)

        idx += plotter.required_axes

    # Hide unused axes
    for ax in axes[idx:]:
        ax.set_axis_off()

    return fig, axes


@dataclass
class _TermPlotter:
    make_plot: Callable
    underlying_function: Callable
    target_variable: str
    required_axes: int = 1


def _get_term_plotter(
    term: AbstractTerm,
    gam: AbstractGAM,
    target: str,
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
) -> _TermPlotter:
    """Utility for plotting a term in a model.

    Because some terms need multiple axes for plotting, this returns the number of axes
    required, and a function that applies the plotting to an iterable of axes, taking
    the axes and **kwargs passed to the plotting function. This allows us to setup the
    axes before plotting when plotting multiple terms.
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")
    data = data if data is not None else gam.fit_state.data
    data = deepcopy(data)

    if _is_random_wiggly(term):
        term = _FactorSmoothToByInterface(term)

    dtypes = {k: data[k].dtype for k in term.varnames}
    by_dtype = data[term.by].dtype if term.by is not None else None
    if isinstance(by_dtype, CategoricalDtype):
        levels = by_dtype.categories
    else:
        levels = [None]
    dim = len(term.varnames)

    def _all_numeric(dtypes: dict):
        return all(is_numeric_dtype(dtype) for dtype in dtypes.values())

    match (dim, term):
        case (1, L()) if isinstance(dtypes[term.varnames[0]], CategoricalDtype):

            def _plot_wrapper(axes: list[Axes], **kwargs: Any):
                axes[0] = categorical(
                    term=term,
                    gam=gam,
                    target=target,
                    data=data,
                    ax=axes[0],
                    **kwargs,
                )
                return axes

            return _TermPlotter(_plot_wrapper, categorical, target)

        case (1, S()) if isinstance(term.bs, RandomEffect):

            def _plot_wrapper(axes: list[Axes], **kwargs: Any):
                axes[0] = random_effect(
                    term=term,
                    gam=gam,
                    target=target,
                    ax=axes[0],
                    **kwargs,
                )
                return axes

            return _TermPlotter(_plot_wrapper, random_effect, target)

        case (1, S()) if (
            isinstance(term.bs, MarkovRandomField) and term.bs.polys is not None
        ):

            def _plot_wrapper(axes: list[Axes], **kwargs: Any):
                axes[0] = markov_random_field(
                    term=term,
                    gam=gam,
                    target=target,
                    ax=axes[0],
                    **kwargs,
                )
                return axes

            return _TermPlotter(_plot_wrapper, random_effect, target)

        case (1, AbstractTerm()) if _all_numeric(dtypes):

            def _plot_wrapper(axes: list[Axes], **kwargs: Any):
                for level in levels:
                    axes[0] = continuous_1d(
                        term=term,
                        gam=gam,
                        target=target,
                        data=data,
                        level=level,
                        ax=axes[0],
                        plot_kwargs={"label": level},
                        **kwargs,
                    )
                if isinstance(by_dtype, CategoricalDtype):
                    axes[0].legend()
                return axes

            return _TermPlotter(_plot_wrapper, continuous_1d, target)

        case (2, AbstractTerm()) if _all_numeric(dtypes):

            def _plot_wrapper(axes: list[Axes], **kwargs: Any):
                for i, level in enumerate(levels):
                    axes[i] = continuous_2d(
                        term=term,
                        gam=gam,
                        target=target,
                        data=data,
                        level=level,
                        ax=axes[i],
                        **kwargs,
                    )
                    if isinstance(by_dtype, CategoricalDtype):
                        axes[i].set_title(f"Level={level}")
                return axes

            return _TermPlotter(
                _plot_wrapper,
                continuous_2d,
                target,
                required_axes=len(levels),
            )

        case _:
            raise NotImplementedError(f"Did not know how to plot term {term}.")


def continuous_1d(
    *,
    term: AbstractTerm | int,
    gam: AbstractGAM,
    target: str | None = None,
    data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
    eval_density: int = 100,
    level: str | None = None,
    n_standard_errors: int | float = 2,
    residuals: bool = False,
    plot_kwargs: dict[str, Any] | None = None,
    fill_between_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot 1D smooth or linear terms with confidence intervals.

    !!! note

        - For terms with numeric "by" variables, the "by" variable is set to 1,
        showing the unscaled effect of the smooth.

    Args:
        term: The model term to plot. Must be a univariate term (single variable).
            If an integer is provided, it is assumed to be the index of the term
            in the predictor of ``target``.
        gam: GAM model containing the term to plot.
        target: Name of the target variable (response variable or family
            parameter name from the model specification). If set to None, an error
            is raised when multiple predictors are present; otherwise, the sole
            available target is used.
        data: DataFrame used for plotting partial residuals and determining
            axis limits. Defaults to the data used for training.
        eval_density: Number of evaluation points along the variable range
            for plotting the smooth curve. Higher values give smoother curves
            but increase computation time. Default is 100.
        level: Must be provided for smooths with a categorical "by" variable or a
            [`FactorSmooth`][pymgcv.basis_functions.FactorSmooth] basis.
            Specifies the level to plot.
        n_standard_errors: Number of standard errors for confidence intervals.
        residuals: Whether to plot partial residuals.
        plot_kwargs: Keyword arguments passed to ``matplotlib.pyplot.plot`` for
            the main curve.
        fill_between_kwargs: Keyword arguments passed to
            `matplotlib.pyplot.fill_between` for the confidence interval band.
            Pass `{"disable": True}` to disable the confidence interval band.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for partial residuals (ignored if `residuals=False`).
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot.
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")

    if target is None:
        if len(gam.predictors) > 1:
            raise ValueError(
                "Target must be specified when multiple predictors are present.",
            )
        target = list(gam.predictors.keys())[0]

    if isinstance(term, int):
        term = gam.predictors[target][term]

    data = data if data is not None else gam.fit_state.data
    data = deepcopy(data)
    term = _FactorSmoothToByInterface(term) if _is_random_wiggly(term) else term
    is_categorical_by = term.by and isinstance(data[term.by].dtype, CategoricalDtype)

    if len(term.varnames) != 1:
        raise ValueError(
            f"Expected varnames to be one continuous variable, got {term.varnames}",
        )
    if is_categorical_by and level is None:
        raise ValueError(
            "level must be provided for terms with 'by' variables, or FactorSmooths.",
        )

    if level is not None and term.by is not None:
        if isinstance(data, Mapping):
            data = {k: v[data[term.by] == level] for k, v in data.items()}
        else:
            data = data.loc[data[term.by] == level]
            assert isinstance(data, pd.DataFrame)

    x0_linspace = np.linspace(
        data[term.varnames[0]].min(),
        data[term.varnames[0]].max(),
        num=eval_density,
    )
    spaced_data = pd.DataFrame({term.varnames[0]: x0_linspace})

    if term.by is not None:
        if is_numeric_dtype(data[term.by].dtype):
            spaced_data[term.by] = 1
        else:
            spaced_data[term.by] = pd.Series(
                [level] * eval_density,
                dtype=data[term.by].dtype,
            )

    ax = plt.gca() if ax is None else ax
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    fill_between_kwargs = {} if fill_between_kwargs is None else fill_between_kwargs
    fill_between_kwargs.setdefault("alpha", 0.2)
    scatter_kwargs.setdefault("s", 0.05 * rcParams["lines.markersize"] ** 2)

    # Matching color, particularly nice for plotting categorical by smooths on same ax
    current_color = ax._get_lines.get_next_color()  # type: ignore Can't find reasonable alternative for now
    for kwargs in (plot_kwargs, fill_between_kwargs, scatter_kwargs):
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = current_color

    pred = gam.partial_effect(term, target, spaced_data, compute_se=True)

    # Add partial residuals
    if residuals and target in data and not _is_linear_functional(term, data):  # type: ignore
        partial_residuals = gam.partial_residuals(
            term,
            target=target,
            data=data,
            avoid_scaling=True,
        )

        ax.scatter(data[term.varnames[0]], partial_residuals, **scatter_kwargs)

    # Plot interval
    assert pred.se is not None
    _with_disable(ax.fill_between)(
        x0_linspace,
        pred.fit - n_standard_errors * pred.se,
        pred.fit + n_standard_errors * pred.se,
        **fill_between_kwargs,
    )

    ax.plot(x0_linspace, pred.fit, **plot_kwargs)
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.label())
    return ax


def continuous_2d(
    *,
    term: AbstractTerm | int,
    gam: AbstractGAM,
    target: str | None = None,
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series] | None = None,
    eval_density: int = 50,
    level: str | None = None,
    contour_kwargs: dict | None = None,
    contourf_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot 2D smooth surfaces as contour plots with data overlay.

    This function is essential for understanding bivariate relationships
    and interactions between two continuous variables.

    Args:
        term: The bivariate term to plot. Must have exactly two variables.
            Can be S('x1', 'x2') or T('x1', 'x2'). If an integer is provided,
            it is interpreted as the index of the term the list of predictors
            for ``target``.
        gam: GAM model containing the term to plot.
        target: Name of the target variable (response variable or family
            parameter name from the model specification). If set to None, an error
            is raised when multiple predictors are present; otherwise, the sole
            available target is used.
        data: DataFrame containing the variables for determining plot range
            and showing data points. Should typically be the training data.
        eval_density: Number of evaluation points along each axis, creating
            an eval_density × eval_density grid. Higher values give smoother
            surfaces but increase computation time. Default is 50.
        level: Must be provided for smooths with a categorical "by" variable or a
            [`FactorSmooth`][pymgcv.basis_functions.FactorSmooth] basis.
            Specifies the level to plot.
        contour_kwargs: Keyword arguments passed to `matplotlib.pyplot.contour`
            for the contour lines.
        contourf_kwargs: Keyword arguments passed to `matplotlib.pyplot.contourf`
            for the filled contours.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`
            for the data points overlay. Pass `{"disable": True}` to avoid plotting.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    Returns:
        The matplotlib Axes object with the plot, allowing further customization.

    Raises:
        ValueError: If the term doesn't have exactly two variables.
    """
    if gam.fit_state is None:
        raise ValueError("Cannot plot before fitting the model.")

    if target is None:
        if len(gam.predictors) > 1:
            raise ValueError(
                "Target must be specified when multiple predictors are present.",
            )
        target = list(gam.predictors.keys())[0]

    if isinstance(term, int):
        term = gam.predictors[target][term]

    data = data if data is not None else gam.fit_state.data
    data = deepcopy(data)
    term = _FactorSmoothToByInterface(term) if _is_random_wiggly(term) else term
    is_categorical_by = term.by and isinstance(data[term.by].dtype, CategoricalDtype)

    if len(term.varnames) != 2:
        raise ValueError(
            f"Expected varnames to be one continuous variable, got {term.varnames}",
        )

    if is_categorical_by and level is None:
        raise ValueError(
            "level must be provided for terms with 'by' variables, or FactorSmooths.",
        )

    if level is not None and term.by is not None:
        if isinstance(data, Mapping):
            data = {k: v[data[term.by] == level] for k, v in data.items()}
        else:
            data = data.loc[data[term.by] == level]
            assert isinstance(data, pd.DataFrame)

    x0_lims = (data[term.varnames[0]].min(), data[term.varnames[0]].max())
    x1_lims = (data[term.varnames[1]].min(), data[term.varnames[1]].max())
    x0_mesh, x1_mesh = np.meshgrid(
        np.linspace(*x0_lims, eval_density),
        np.linspace(*x1_lims, eval_density),
    )
    spaced_data = pd.DataFrame(
        {term.varnames[0]: x0_mesh.ravel(), term.varnames[1]: x1_mesh.ravel()},
    )
    if term.by is not None:
        if is_numeric_dtype(data[term.by].dtype):
            spaced_data[term.by] = 1
        else:
            spaced_data[term.by] = pd.Series(
                [level] * eval_density**2,
                dtype=data[term.by].dtype,
            )

    ax = plt.gca() if ax is None else ax
    contour_kwargs = {} if contour_kwargs is None else contour_kwargs
    contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs

    contour_kwargs.setdefault("levels", 14)
    contourf_kwargs.setdefault("levels", 14)
    contourf_kwargs.setdefault("alpha", 0.8)
    scatter_kwargs.setdefault("color", "black")
    scatter_kwargs.setdefault("s", 0.05 * rcParams["lines.markersize"] ** 2)
    scatter_kwargs.setdefault("zorder", 2)  # Ensures above contours

    pred = gam.partial_effect(
        term,
        target,
        data=spaced_data,
    )

    mesh = ax.contourf(
        x0_mesh,
        x1_mesh,
        pred.reshape(x0_mesh.shape),
        **contourf_kwargs,
    )
    _with_disable(ax.contour)(
        x0_mesh,
        x1_mesh,
        pred.reshape(x0_mesh.shape),
        **contour_kwargs,
    )
    color_bar = ax.figure.colorbar(mesh, ax=ax, pad=0)
    color_bar.set_label(term.label())
    _with_disable(ax.scatter)(
        np.asarray(data[term.varnames[0]]).ravel(),  # Ravel for linear functional terms
        np.asarray(data[term.varnames[1]]).ravel(),
        **scatter_kwargs,
    )
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.varnames[1])
    return ax


def categorical(
    *,
    term: L | int,
    gam: AbstractGAM,
    target: str | None = None,
    data: pd.DataFrame | Mapping[str, pd.Series | np.ndarray] | None = None,
    residuals: bool = False,
    n_standard_errors: int | float = 2,
    errorbar_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot categorical terms with error bars and partial residuals.

    Creates a plot showing:

    - The estimated effect of each category level as points.
    - Error bars representing confidence intervals.
    - Partial residuals as jittered scatter points.

    Args:
        term: The categorical term to plot. Must be a L term with a single
            categorical variable.
        gam: GAM model containing the term to plot.
        target: Name of the target variable (response variable or family
            parameter name from the model specification). If set to None, an error
            is raised when multiple predictors are present; otherwise, the sole
            available target is used.
        data: DataFrame (or dictionary) containing the categorical variable and
            response variable.
        residuals: Whether to plot partial residuals (jittered on x-axis).
        n_standard_errors: Number of standard errors for confidence intervals.
        errorbar_kwargs: Keyword arguments passed to `matplotlib.pyplot.errorbar`.
        scatter_kwargs: Keyword arguments passed to `matplotlib.pyplot.scatter`.
        ax: Matplotlib Axes object to plot on. If None, uses current axes.

    """
    if gam.fit_state is None:
        raise RuntimeError("The model must be fitted before plotting.")

    if target is None:
        if len(gam.predictors) > 1:
            raise ValueError(
                "Target must be specified when multiple predictors are present.",
            )
        target = list(gam.predictors.keys())[0]

    if isinstance(term, int):
        term = gam.predictors[target][term]  # type: ignore - checked below

    if not isinstance(term, L):
        raise TypeError("The term must be a linear term.")

    data = gam.fit_state.data if data is None else data

    errorbar_kwargs = {} if errorbar_kwargs is None else errorbar_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.05 * rcParams["lines.markersize"] ** 2)
    errorbar_kwargs.setdefault("capsize", 10)
    errorbar_kwargs.setdefault("fmt", ".")

    ax = plt.gca() if ax is None else ax
    vals = pd.Series(data[term.varnames[0]])

    if not isinstance(vals.dtype, CategoricalDtype):
        raise TypeError("The variable must be categorical in the data.")

    levels = pd.Series(
        vals.cat.categories,
        dtype=vals.dtype,
        name=term.varnames[0],
    )

    if residuals and target in data:
        partial_residuals = gam.partial_residuals(term, target, data)

        jitter = np.random.uniform(-0.25, 0.25, size=data_len(data))
        scatter_kwargs.setdefault("alpha", 0.2)

        ax.scatter(
            vals.cat.codes + jitter,
            partial_residuals,
            **scatter_kwargs,
        )

    ax.set_xticks(ticks=levels.cat.codes, labels=levels)

    pred = gam.partial_effect(
        term=term,
        target=target,
        data=pd.DataFrame(levels),
        compute_se=True,
    )

    assert pred.se is not None

    ax.errorbar(
        x=levels.cat.codes,
        y=pred.fit,
        yerr=n_standard_errors * pred.se,
        **errorbar_kwargs,
    )
    ax.set_xlabel(term.varnames[0])
    ax.set_ylabel(term.label())
    return ax


def random_effect(
    *,
    term: S | int,
    gam: AbstractGAM,
    target: str | None = None,
    confidence_interval_level: float = 0.95,
    axline_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    fill_between_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """A QQ-like-plot for random effect terms.

    This function plots the estimated random effects against Gaussian quantiles and
    includes a confidence envelope to assess whether the random effects follow a
    normal distribution, as assumed by the model.

    Args:
        term: The random effect term to plot. Must be a smooth term with a
            [`RandomEffect`][pymgcv.basis_functions.RandomEffect] basis function.
            If an integer is provided, it is assumed to be the index of the term
            in the predictors for ``target``.
        gam: The fitted GAM model containing the random effect.
        target: The target variable to plot when multiple predictors are present.
            If None and only one predictor exists, that predictor is used.
        confidence_interval_level: The confidence level for the confidence envelope.
        axline_kwargs: Keyword arguments passed to `matplotlib.axes.Axes.axline` for
            the reference line.
        scatter_kwargs: Keyword arguments passed to `matplotlib.axes.Axes.scatter` for
            the random effect points.
        fill_between_kwargs: Keyword arguments passed to
            `matplotlib.axes.Axes.fill_between` for the confidence envelope.
        ax: Matplotlib axes to use for the plot. If None, uses the current axes.

    Returns:
        The matplotlib axes object.

    !!! note
        The confidence interval calculation is based on the formula from:
        "Worm plot: a simple diagnostic device for modelling growth reference curves"
        (page 6). The random effects are constrained to be centered, so the reference
        line passes through (0, 0).


    """
    if gam.fit_state is None:
        raise RuntimeError("The model must be fitted before plotting.")

    if target is None:
        if len(gam.predictors) > 1:
            raise ValueError(
                "Target must be specified when multiple predictors are present.",
            )
        target = list(gam.predictors.keys())[0]

    if isinstance(term, int):
        term = gam.predictors[target][term]  # type: ignore - checked below

    if not isinstance(term, S):
        raise TypeError("Term is not a smooth term.")

    if not isinstance(term.bs, RandomEffect):
        raise TypeError("Term is not a random effect term.")

    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.05 * rcParams["lines.markersize"] ** 2)

    fill_between_kwargs = {} if fill_between_kwargs is None else fill_between_kwargs
    current_color = ax._get_lines.get_next_color()  # type: ignore Can't find reasonable alternative for now
    for kwargs in (scatter_kwargs, fill_between_kwargs):
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = current_color

    axline_kwargs = {} if axline_kwargs is None else axline_kwargs

    if "c" not in axline_kwargs and "color" not in axline_kwargs:
        axline_kwargs["color"] = "gray"

    axline_kwargs.setdefault("linestyle", "--")
    ax = plt.gca() if ax is None else ax

    levels = np.unique(gam.fit_state.data[term.varnames[0]])
    pred = gam.partial_effect(
        term=term,
        target=target,
        data=pd.DataFrame(pd.Series(levels, name=term.varnames[0], dtype="category")),
    )
    pred = np.sort(pred)
    n = len(levels)
    probs = (np.arange(n) + 0.5) / n
    x = norm.ppf(probs)

    _with_disable(ax.scatter)(x, pred, **scatter_kwargs)

    # Add CI lines. Confidence interval based on formula in
    # "Worm plot: a simple diagnostic device for modelling growth reference curves"
    # page 6. We need to multiply them by sd(.dat$y) because we are not normalizing the
    # random effects.
    alpha = (1 - confidence_interval_level) / 2
    pred_std = np.std(pred)
    interval = (
        pred_std * norm.ppf(alpha) * np.sqrt(probs * (1 - probs) / n) / norm.pdf(x)
    )

    ref_y = x * pred_std + np.median(pred)
    _with_disable(ax.fill_between)(
        x,
        ref_y - interval,
        ref_y + interval,
        alpha=0.2,
        **fill_between_kwargs,
    )

    # Use the median y value as the constraining point
    _with_disable(ax.axline)(
        (0, np.median(pred)),
        slope=pred_std.item(),
        **axline_kwargs,
    )
    ax.set_xlabel("Gaussian Quantiles")
    ax.set_ylabel(f"{term.varnames[0]} effect")
    return ax


def markov_random_field(
    *,
    term: S | int,
    gam: AbstractGAM,
    target: str | None = None,
    ax: Axes | None = None,
    **kwargs,
):
    """Plot a Markov Random Field spatial smooth.

    This function visualizes spatial data using polygons defined by the Markov Random Field
    basis function, colored by the partial effects of the fitted GAM model.

    Args:
        term: Either a smooth term object or an integer index of the term to plot.
        gam: The fitted GAM model containing the term.
        target: The target variable name when multiple predictors are present.
            If None and only one predictor exists, it will be used automatically.
        ax: Matplotlib axes to plot on. If None, uses current axes.
        **kwargs: Additional keyword arguments passed to PatchCollection for
            customizing the appearance of polygons (e.g., edgecolor, linewidth,
            facecolor, alpha, etc.).

    Returns:
        The matplotlib axes object with the plot.

    Raises:
        ValueError: If the model is not fitted, target is not specified with multiple
            predictors, or polygons are not defined in the basis.
        TypeError: If the term is not a smooth term with MarkovRandomField basis.
    """
    if gam.fit_state is None:
        raise ValueError("The model must be fitted before plotting.")

    data = gam.fit_state.data

    if target is None:
        if len(gam.predictors) > 1:
            raise ValueError(
                "Target must be specified when multiple predictors are present.",
            )
        target = list(gam.predictors.keys())[0]

    if isinstance(term, int):
        term = gam.predictors[target][term]  # type: ignore checked below

    if not isinstance(term, S) or not isinstance(term.bs, MarkovRandomField):
        raise TypeError("Term must be a smooth term with a MarkovRandomField basis.")

    if term.bs.polys is None:
        raise ValueError("Cannot plot MarkovRandomField with unknown polygons.")

    if ax is None:
        ax = plt.gca()

    polys = term.bs.polys
    var = term.varnames[0]
    categories = pd.Series(data[var]).cat.categories.astype(data[var].dtype)
    partial_effect = gam.partial_effect(term=0, data={var: categories})
    patches = []
    for level in categories:
        patches.append(Polygon(polys[level]))

    kwargs.setdefault("edgecolor", "black")
    p = PatchCollection(patches, **kwargs)
    p.set_array(partial_effect)
    ax.add_collection(p)
    p.set_clim(np.min(partial_effect), np.max(partial_effect))
    fig = ax.figure  # get the parent figure
    fig.colorbar(p, ax=ax, label=term.label(), pad=0)
    all_xy = np.vstack([poly for poly in polys.values()])
    ax.set_xlim(all_xy[:, 0].min(), all_xy[:, 0].max())
    ax.set_ylim(all_xy[:, 1].min(), all_xy[:, 1].max())
    return ax


def qq(
    gam: AbstractGAM,
    *,
    qq_fun: Callable[[AbstractGAM], QQResult] = qq_simulate,
    scatter_kwargs: dict | None = None,
    fill_between_kwargs: dict | None = None,
    axline_kwargs: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """A Q-Q plot of deviance residuals.

    Args:
        gam: The fitted GAM model.
        qq_fun: A function taking only the GAM model, and returning a `QQResult`
            object storing the theoretical residuals, residuals, and the confidence
            interval. Defaults to [`qq_simulate`][pymgcv.qq.qq_simulate], which is the
            most widely supported method only requiring the family to provide a
            sampling function. [`qq_transform`][pymgcv.qq.qq_transform] can be used for
            families providing a cdf method, which transforms the data to a known
            distribution for which an analytical confidence interval is available.
        scatter_kwargs: Key word arguments passed to `matplotlib.pyplot.scatter`.
        fill_between_kwargs: Key word arguments passed to
            `matplotlib.pyplot.fill_between`, for plotting the confidence interval.
        axline_kwargs: Key word arguments passed to `matplotlib.pyplot.axline` for
            plotting the reference line. Pass {"disable": True} to avoid plotting.
        ax: Matplotlib axes to use for the plot.


    !!! note

        To change settings of ``qq_fun``, use partial application, e.g.
        ```python
        from pymgcv.qq import qq_simulate
        from functools import partial
        import pymgcv.plot as gplt

        qq_fun = partial(qq_simulate, level=0.95, n_sim=10)
        # gplt.qq(..., qq_fun=qq_fun)
        ```

    Returns:
        The matplotlib axes object.

    !!! example

        As an example, we will create a heavy tailed response variable,
        and fit a [`Gaussian`][pymgcv.families.Gaussian] model, and a
        [`Scat`][pymgcv.families.Scat] model.

        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from pymgcv.families import Gaussian, Scat
        from pymgcv.gam import GAM
        import pymgcv.plot as gplt
        from pymgcv.terms import S

        rng = np.random.default_rng(1)
        n = 1000
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.standard_t(df=3, size=n)  # Heavy-tailed
        data = pd.DataFrame({"x": x, "y": y})

        models = [
            GAM({"y": S("x")}, family=Gaussian()),
            GAM({"y": S("x")}, family=Scat()),  # Better for heavy-tailed data
        ]

        fig, axes = plt.subplots(ncols=2)

        for model, ax in zip(models, axes, strict=False):
            model.fit(data)
            gplt.qq(model, ax=ax)
            ax.set_title(model.family.__class__.__name__)
            ax.set_box_aspect(1)

        # fig.show()  # Uncomment to display the figure
        ```
    """
    if gam.fit_state is None:
        raise RuntimeError("The model must be fitted before plotting.")

    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.2 * rcParams["lines.markersize"] ** 2)
    fill_between_kwargs = {} if fill_between_kwargs is None else fill_between_kwargs
    fill_between_kwargs.setdefault("alpha", 0.4)

    if "c" not in scatter_kwargs and "color" not in scatter_kwargs:
        scatter_kwargs["color"] = "black"

    axline_kwargs = {} if axline_kwargs is None else axline_kwargs

    if "c" not in axline_kwargs and "color" not in axline_kwargs:
        axline_kwargs["color"] = "gray"
    axline_kwargs.setdefault("linestyle", "--")
    ax = plt.gca() if ax is None else ax
    qq_data = qq_fun(gam)

    _with_disable(ax.fill_between)(
        qq_data.theoretical,
        *qq_data.interval,
        **fill_between_kwargs,
    )

    ax.scatter(qq_data.theoretical, qq_data.residuals, **scatter_kwargs)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Residuals")
    _with_disable(ax.axline)((0, 0), slope=1, **axline_kwargs)
    return ax


def residuals_vs_linear_predictor(
    gam: AbstractGAM,
    type: Literal[
        "deviance",
        "pearson",
        "scaled.pearson",
        "working",
        "response",
    ] = "deviance",
    target: str | None = None,
    ax: Axes | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
):
    """Plot the residuals against the linear predictor.

    Args:
        gam: The fitted GAM model.
        type: The type of residuals to plot.
        target: The target variable to plot residuals for.
        ax: The axes to plot on.
        scatter_kwargs: Keyword arguments to pass to the scatter plot.
    """
    if gam.family.n_observed_predictors > 1:
        raise NotImplementedError(
            "Multivariate response families are not supported for "
            "residuals_vs_linear_predictor.",
        )
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs.setdefault("s", 0.05 * rcParams["lines.markersize"] ** 2)

    ax = plt.gca() if ax is None else ax
    residuals = gam.residuals(type)
    if target is None:
        if len(gam.predictors) > 1:
            raise ValueError(
                "Target must be specified when multiple predictors are present.",
            )
        target = list(gam.predictors.keys())[0]
    predictions = gam.predict()[target]
    ax.scatter(predictions, residuals, **scatter_kwargs)
    ax.set_xlabel("Linear predictor")
    ax.set_ylabel("Residuals")
    return ax


def hexbin_residuals(
    residuals: np.ndarray,
    var1: str,
    var2: str,
    data: pd.DataFrame | Mapping[str, np.ndarray | pd.Series],
    *,
    gridsize: int = 25,
    max_val: int | float | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
):
    """Hexbin plot for visualising residuals as function of two variables.

    Useful e.g. for assessing if interactions are might be required. This
    is a thin wrapper around `matplotlib.pyplot.hexbin`, with better defaults
    for plotting residuals (e.g. uses a symmetric color scale).

    The default reduction function is `np.sum(res) / np.sqrt(len(res))`,
    which has constant variance w.r.t. the number of points.

    Args:
        residuals: Residuals to plot.
        var1: Name of the first variable.
        var2: Name of the second variable.
        data: The data (containing ``var1`` and ``var2``).
        gridsize: The number of hexagons in the x-direction. The y direction is chosen
            such that the hexagons are approximately regular.
        max_val: Maximum and minimum value for the symmetric color scale. Defaults to
            the maximum absolute value of the residuals.
        ax: Axes to plot on. If None, the current axes are used.
        **kwargs: Additional keyword arguments passed to `matplotlib.hexbin`.

    !!! example

        ```python
        import numpy as np
        import pymgcv.plot as gplt
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(1)

        fig, ax = plt.subplots()
        residuals = rng.normal(size=500)  # or gam.residuals()
        data = {
            "x0": rng.normal(size=residuals.shape),
            "x1": rng.normal(size=residuals.shape),
            }
        gplt.hexbin_residuals(residuals, "x0", "x1", data=data, ax=ax)
        ```
    """
    max_color = np.max(np.abs(residuals)) if max_val is None else max_val
    kwargs.setdefault("cmap", "coolwarm")
    kwargs.setdefault("reduce_C_function", lambda x: np.sum(x) / np.sqrt(len(x)))
    kwargs.setdefault("vmin", -max_color)
    kwargs.setdefault("vmax", max_color)
    ax = plt.gca() if ax is None else ax
    ax.hexbin(
        data[var1],
        data[var2],
        gridsize=gridsize,
        C=residuals,
        **kwargs,
    )
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    return ax


def _with_disable(plot_func):
    """Wraps a plot function to easily disable with disable=True."""

    def wrapper(*args: Any, disable: bool = False, **kwargs: Any):
        if disable:
            return None
        return plot_func(*args, **kwargs)

    return wrapper


def _is_random_wiggly(term: AbstractTerm) -> TypeGuard[T | S]:
    if isinstance(term, S | T):
        return isinstance(term.bs, FactorSmooth)
    return False


def _is_linear_functional(
    term: AbstractTerm,
    data: pd.DataFrame | dict[str, pd.Series | np.ndarray],
) -> bool:
    return any(np.asarray(data[k]).ndim > 1 for k in term.varnames) and isinstance(
        term,
        S | T,
    )
