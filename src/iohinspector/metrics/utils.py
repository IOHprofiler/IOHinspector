import numpy as np
import polars as pl


def geometric_mean(series: pl.Series) -> float:
    """Helper function for polars: geometric mean"""
    return np.exp(np.log(series).mean())


def get_sequence(
    min: float,
    max: float,
    len: float,
    scale_log: bool = False,
    cast_to_int: bool = False,
) -> np.ndarray:
    """Create sequence of points, used for subselecting targets / budgets for allignment and data processing

    Args:
        min (float): Starting point of the range
        max (float): Final point of the range
        len (float): Number of steps
        scale_log (bool): Whether values should be scaled logarithmically. Defaults to False
        version (str, optional): Whether the value should be casted to integers (e.g. in case of budget) or not. Defaults to False.

    Returns:
        np.ndarray: Array of evenly spaced values
    """
    transform = lambda x: x
    if scale_log:
        assert min > 0
        min = np.log10(min)
        max = np.log10(max)
        transform = lambda x: 10**x
    if len == 1:
        values =np.array([min])
    else:
        if(max == min):
            values = np.ones(len) * min
        else:
            values = np.arange(
                min,
                max + (max - min) / (2 * (len - 1)),
                (max - min) / (len - 1),
                dtype=float,
            )
            
    values = transform(values)
    if cast_to_int:
        return np.unique(np.array(values, dtype=int))
    return np.unique(values)
