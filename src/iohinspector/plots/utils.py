from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence, Union, Dict, Any
from dataclasses import fields
from typing import TypeVar, Generic

T = TypeVar('T', bound='BasePlotArgs')

@dataclass
class BasePlotArgs:
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None

    xscale: str = None
    yscale: str = None

    figsize: Optional[Tuple[float, float]] = (16,9)
    dpi: Optional[int] = None

    grid: Union[bool, str] = False
    legend: bool = False
    legend_loc: str = "best"
    legend_kwargs: Dict[str, Any] = field(default_factory=dict)

    fontsize: Optional[Union[int, str]] = None
    title_fontsize: Optional[Union[int, str]] = None
    tick_params: Dict[str, Any] = field(default_factory=dict)

    xticks: Optional[Sequence[float]] = None
    yticks: Optional[Sequence[float]] = None

    reverse_xaxis: bool = False
    reverse_yaxis: bool = False

    tight_layout: bool = True

    def __post_init__(self) -> None:
        if self.xlim is not None and not isinstance(self.xlim, tuple):
            self.xlim = tuple(self.xlim)  # type: ignore
        if self.ylim is not None and not isinstance(self.ylim, tuple):
            self.ylim = tuple(self.ylim)  # type: ignore
        if self.xticks is not None and not isinstance(self.xticks, tuple):
            self.xticks = tuple(self.xticks)  # type: ignore
        if self.yticks is not None and not isinstance(self.yticks, tuple):
            self.yticks = tuple(self.yticks)  # type: ignore

    def as_dict(self) -> Dict[str, Any]:
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

        Raises:
            RuntimeError: If matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required to apply plot properties") from exc


        # Title and labels with fontsize handling
        if self.title is not None:
            if self.title_fontsize is not None:
                ax.set_title(self.title, fontsize=self.title_fontsize)
            elif self.fontsize is not None:
                ax.set_title(self.title, fontsize=self.fontsize)
            else:
                ax.set_title(self.title)

        if self.xlabel is not None:
            if self.fontsize is not None:
                ax.set_xlabel(self.xlabel, fontsize=self.fontsize)
            else:
                ax.set_xlabel(self.xlabel)
        
        if self.ylabel is not None:
            if self.fontsize is not None:
                ax.set_ylabel(self.ylabel, fontsize=self.fontsize)
            else:
                ax.set_ylabel(self.ylabel)

        

        # Ticks
        if self.xticks is not None:
            ax.set_xticks(list(self.xticks))
        if self.yticks is not None:
            ax.set_yticks(list(self.yticks))

        # Limits
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)

        # Scales
        if self.xscale:
            ax.set_xscale(self.xscale)
        if self.yscale:
            ax.set_yscale(self.yscale)


        # Grid
        if isinstance(self.grid, bool):
            ax.grid(self.grid)
        elif isinstance(self.grid, str):
            ax.grid(True, which=self.grid)

        # Legend
        if self.legend:
            kwargs = dict(self.legend_kwargs or {})
            if "loc" not in kwargs:
                kwargs["loc"] = self.legend_loc
            # Only attempt to create legend if there are labeled artists
            try:
                ax.legend(**kwargs)
            except Exception:
                # fallback: call without kwargs
                ax.legend()

        # Tick params (includes labelsize if provided)
        if self.tick_params:
            ax.tick_params(**self.tick_params)
        elif self.fontsize is not None:
            ax.tick_params(labelsize=self.fontsize)

        # Reverse axes if requested
        if self.reverse_xaxis:
            ax.invert_xaxis()
        if self.reverse_yaxis:
            ax.invert_yaxis()

        return ax
    
   
    def override(self, other: Optional[Union["BasePlotArgs", Dict[str, Any]]]):
        """Update plot arguments in place with values from another source.

        Args:
            other (Optional[Union[BasePlotArgs, Dict[str, Any]]]): Plot arguments to override current values with. 
                Can be either a BasePlotArgs instance or a dictionary. Values from `other` override those 
                from `self` when they are not None. Dictionary fields (legend_kwargs, tick_params) are merged 
                with `other` taking precedence for overlapping keys.

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
    line_colors: Optional[Sequence[str]] = None
    
    def as_dict(self):
        """Convert the line plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all line plot configuration parameters including line colors.
        """
        results = super().as_dict()
        results["line_colors"] = self.line_colors
        return results
        

    def apply(self, ax):
        """Apply line plot properties to a matplotlib Axes object.

        Args:
            ax: matplotlib Axes instance to apply the line plot properties to.

        Returns:
            ax: The modified matplotlib Axes object with line plot properties applied.
        """
        return super().apply(ax)

    def override(self, other):
        """Update line plot arguments in place with values from another source.

        Args:
            other: Line plot arguments to override current values with.
        """
        return super().override(other)


@dataclass
class HeatmapPlotArgs(BasePlotArgs):
    heatmap_palette: Optional[str] = "viridis"
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
        

    def apply(self, ax):
        """Apply heatmap plot properties to a matplotlib Axes object.

        Args:
            ax: matplotlib Axes instance to apply the heatmap plot properties to.

        Returns:
            ax: The modified matplotlib Axes object with heatmap plot properties applied.
        """
        return super().apply(ax)

    def override(self, other):
        """Update heatmap plot arguments in place with values from another source.

        Args:
            other: Heatmap plot arguments to override current values with.
        """
        return super().override(other)


@dataclass
class ScatterPlotArgs(BasePlotArgs):
    point_colors: Optional[Sequence[str]] = None
    
    def as_dict(self):
        """Convert the scatter plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all scatter plot configuration parameters including point colors.
        """
        results = super().as_dict()
        results["point_colors"] = self.point_colors
        return results
        

    def apply(self, ax):
        """Apply scatter plot properties to a matplotlib Axes object.

        Args:
            ax: matplotlib Axes instance to apply the scatter plot properties to.

        Returns:
            ax: The modified matplotlib Axes object with scatter plot properties applied.
        """
        return super().apply(ax)

    def override(self, other):
        """Update scatter plot arguments in place with values from another source.

        Args:
            other: Scatter plot arguments to override current values with.
        """
        return super().override(other)

def _save_fig(fig = None, file_name: str=None, plot_args: BasePlotArgs=None):
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


def _create_plot_args(
    defaults: T,
    overrides: Optional[Union[T, Dict[str, Any]]] = None,
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