from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar, Union

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sbs

from ..metrics.utils import group_label

T = TypeVar("T", bound="BasePlotArgs")


@dataclass
class BasePlotArgs:
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    xscale: str = None
    yscale: str = None

    figsize: tuple[float, float] | None = (16, 9)
    dpi: int | None = None

    grid: bool | str = False
    legend: bool = False
    legend_loc: str = "best"
    legend_kwargs: dict[str, Any] = field(default_factory=dict)

    fontsize: int | str | None = None
    title_fontsize: int | str | None = None
    tick_params: dict[str, Any] = field(default_factory=dict)

    xticks: Sequence[float] | None = None
    yticks: Sequence[float] | None = None

    reverse_xaxis: bool = False
    reverse_yaxis: bool = False

    tight_layout: bool = True

    def __post_init__(self) -> None:
        for name in ("xlim", "ylim", "xticks", "yticks"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                setattr(self, name, tuple(value))

    def as_dict(self) -> dict[str, Any]:
        """Convert the plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all plot configuration parameters.
        """
        return {
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "xlim": self.xlim,
            "ylim": self.ylim,
            "xscale": self.xscale,
            "yscale": self.yscale,
            "figsize": self.figsize,
            "dpi": self.dpi,
            "grid": self.grid,
            "legend": self.legend,
            "legend_loc": self.legend_loc,
            "legend_kwargs": dict(self.legend_kwargs),
            "fontsize": self.fontsize,
            "title_fontsize": self.title_fontsize,
            "tick_params": dict(self.tick_params),
            "xticks": self.xticks,
            "yticks": self.yticks,
            "tight_layout": self.tight_layout,
        }

    def apply(self, ax):
        """Apply stored plot properties to a matplotlib Axes object.

        Args:
            ax: matplotlib Axes instance to apply the properties to.

        Returns:
            ax: The modified matplotlib Axes object with properties applied.
        """
        self._apply_text(ax)
        self._apply_axes(ax)
        if isinstance(self.grid, bool):
            ax.grid(self.grid)
        elif isinstance(self.grid, str):
            ax.grid(True, which=self.grid)
        if self.legend:
            kwargs = {"loc": self.legend_loc, **(self.legend_kwargs or {})}
            try:
                ax.legend(**kwargs)
            except Exception:  # noqa: BLE001
                ax.legend()  # fallback: call without kwargs
        if self.tick_params:
            ax.tick_params(**self.tick_params)
        elif self.fontsize is not None:
            ax.tick_params(labelsize=self.fontsize)
        for axis in "xy":
            if getattr(self, f"reverse_{axis}axis"):
                getattr(ax, f"invert_{axis}axis")()
        return ax

    def _apply_text(self, ax) -> None:
        """The title and axis labels, in ``fontsize`` (the title in
        ``title_fontsize`` if given)."""
        size = {} if self.fontsize is None else {"fontsize": self.fontsize}
        if self.title is not None:
            title = size
            if self.title_fontsize is not None:
                title = {"fontsize": self.title_fontsize}
            ax.set_title(self.title, **title)
        for axis in "xy":
            label = getattr(self, f"{axis}label")
            if label is not None:
                getattr(ax, f"set_{axis}label")(label, **size)

    def _apply_axes(self, ax) -> None:
        """The ticks, limits and scales of x and y, in that order."""
        for axis in "xy":
            ticks = getattr(self, f"{axis}ticks")
            if ticks is not None:
                getattr(ax, f"set_{axis}ticks")(list(ticks))
        for axis in "xy":
            lim = getattr(self, f"{axis}lim")
            if lim is not None:
                getattr(ax, f"set_{axis}lim")(*lim)
        for axis in "xy":
            scale = getattr(self, f"{axis}scale")
            if scale:
                getattr(ax, f"set_{axis}scale")(scale)

    def override(self, other: Union["BasePlotArgs", dict[str, Any]] | None):
        """Update plot arguments in place with values from another source.

        Args:
            other (Optional[Union[BasePlotArgs, Dict[str, Any]]]): Plot arguments to override current values with.
                Can be either a BasePlotArgs instance or a dictionary. Values from `other` replace those
                of `self` when they are not None, dictionary fields (legend_kwargs, tick_params) too.

        Note:
            Works with inheritance - handles fields from both base and derived classes.
            For sequence-like fields (xlim, ylim, xticks, yticks) lists/tuples from `other` are converted to tuples.
        """
        if other is None:
            return

        is_dict = isinstance(other, dict)

        # Use self.__class__ to get fields from the actual class (including subclass fields)
        for f in fields(self.__class__):
            name = f.name
            v2 = other.get(name, None) if is_dict else getattr(other, name, None)

            if v2 is not None:
                setattr(self, name, v2)


@dataclass
class LinePlotArgs(BasePlotArgs):
    line_colors: Sequence[str] | None = None

    def as_dict(self):
        """Convert the line plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all line plot configuration parameters including line colors.
        """
        results = super().as_dict()
        results["line_colors"] = self.line_colors
        return results


@dataclass
class HeatmapPlotArgs(BasePlotArgs):
    heatmap_palette: str | None = "viridis"
    use_background_color: bool = True
    background_color: str = "white"

    def as_dict(self):
        """Convert the heatmap plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all heatmap plot configuration parameters including palette settings.
        """
        results = super().as_dict()
        results["heatmap_palette"] = self.heatmap_palette
        return results


@dataclass
class ScatterPlotArgs(BasePlotArgs):
    point_colors: Sequence[str] | None = None

    def as_dict(self):
        """Convert the scatter plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all scatter plot configuration parameters including point colors.
        """
        results = super().as_dict()
        results["point_colors"] = self.point_colors
        return results


def _hue(df: Any, by: Sequence[str]) -> Any:
    """The seaborn ``hue`` for the groups ``by`` of the pandas frame ``df``: the
    column, or a label per row (``group_label``) for several columns."""
    by = list(by)
    if len(by) == 1:
        return by[0]
    return pl.from_pandas(df[by]).select(group_label(by)).to_series().to_numpy()


def _measures(
    summary: pl.DataFrame, x: str, by: Sequence[str], measures: Iterable[str]
) -> pl.DataFrame:
    """The ``measures`` columns of ``summary`` in long form (``variable``,
    ``value``) per ``x`` and group of ``by``; an unknown measure fails."""
    unknown = sorted(set(measures) - set(summary.columns))
    if unknown:
        raise ValueError(
            f"unknown measures {unknown}; the summary has {summary.columns}"
        )
    return (
        summary.unpivot(index=[x, *by])
        .filter(pl.col("variable").is_in(list(measures)))
        .sort(by, maintain_order=True)
    )


def _measure_lines(pdf, x: str, by: Sequence[str], plot_args, ax) -> None:
    """One line per group of ``by`` and measure (line style) of ``_measures``."""
    sbs.lineplot(
        pdf,
        x=x,
        y="value",
        style="variable",
        hue=_hue(pdf, by),
        palette=plot_args.line_colors,
        ax=ax,
    )


def _save_fig(fig=None, file_name: str | None = None, plot_args: BasePlotArgs = None):
    """Save a matplotlib figure to file with optional plot arguments.

    Args:
        fig: matplotlib Figure object to save. Defaults to None.
        file_name (str, optional): Path where to save the figure. Defaults to None.
        plot_args (BasePlotArgs, optional): Plot arguments containing DPI and layout settings. Defaults to None.
    """
    if fig and file_name:
        if plot_args.tight_layout:
            fig.tight_layout()
        fig.savefig(file_name, dpi=plot_args.dpi)


@contextmanager
def _canvas(ax, plot_args: BasePlotArgs, file_name: str | None):
    """The axes to draw on: ``ax``, or those of a new figure of ``plot_args``'s
    size. On exit ``plot_args`` is applied to them, and a new figure is saved to
    ``file_name`` if given."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    yield ax
    plot_args.apply(ax)
    _save_fig(fig, file_name, plot_args)


def _create_plot_args(
    defaults: T,
    overrides: T | dict[str, Any] | None = None,
) -> T:
    """Create plot properties by merging defaults with overrides, preserving the exact type of the defaults object.

    Args:
        defaults (T): Default properties object (any BasePlotArgs subclass).
        overrides (Optional[Union[T, Dict[str, Any]]], optional): Properties to override (dict or same type as defaults). Defaults to None.

    Returns:
        T: New properties object of the same type as defaults with overrides applied.
    """
    if overrides is None:
        return defaults

    # Create a copy to avoid mutating the input
    import copy

    result = copy.deepcopy(defaults)
    result.override(overrides)
    return result
