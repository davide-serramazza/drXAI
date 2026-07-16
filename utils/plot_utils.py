import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Sequence, Union

ArrayLike = Union[np.ndarray, Sequence[float]]


def _ensure_2d(series: np.ndarray) -> np.ndarray:
    """
    Ensure the series is 2D as (T, C):
    - If 1D (T,), convert to (T, 1)
    - If 3D (N, T, C), pick the first instance (N=0)
    - If 2D (T, C), keep as is
    """
    if series.ndim == 1:
        return series[:, None]
    if series.ndim == 3:
        # pick the first sample in batch to visualize
        return series[0]
    if series.ndim == 2:
        return series
    raise ValueError(f"Unsupported array shape {series.shape}; expected (T,), (T, C), or (N, T, C)")


def plot_time_series(
    series: ArrayLike,
    time: Optional[ArrayLike] = None,
    labels: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 4),
    colors: Optional[Sequence[str]] = None,
    linewidth: float = 1.5,
    alpha: float = 0.9,
    grid: bool = True,
    legend: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a univariate or multivariate time series.

    Parameters
    - series: np.ndarray or sequence with shapes (T,), (T, C), or (N, T, C)
    - time: Optional time axis of length T. If None, uses np.arange(T).
    - labels: Optional iterable of length C with channel/variable names.
    - title: Optional plot title.
    - figsize: Matplotlib figure size.
    - colors: Optional list of colors for each channel.
    - linewidth: Line width.
    - alpha: Line transparency.
    - grid: Whether to show grid.
    - legend: Whether to show legend when multivariate.
    - save_path: If provided, saves the figure to this path.
    - show: Whether to display the plot (plt.show()). If False, returns the fig, ax.

    Returns
    - (fig, ax) when show=False, otherwise None.
    """
    series = np.asarray(series)
    X = _ensure_2d(series)  # (T, C)
    T, C = X.shape

    if time is None:
        t = np.arange(T)
    else:
        t = np.asarray(time)
        if t.ndim != 1 or len(t) != T:
            raise ValueError(f"time must be 1D with length {T}, got shape {t.shape}")

    if labels is not None:
        labels = list(labels)
        if len(labels) != C:
            raise ValueError(f"labels length must equal number of channels C={C}")

    if colors is not None and len(colors) != C:
        raise ValueError(f"colors length must equal number of channels C={C}")

    fig, ax = plt.subplots(figsize=figsize)

    for c in range(C):
        lbl = labels[c] if labels is not None else (f"chan_{c}" if C > 1 else None)
        color = colors[c] if colors is not None else None
        ax.plot(t, X[:, c], label=lbl, color=color, linewidth=linewidth, alpha=alpha)

    ax.set_xlabel("time")
    ax.set_ylabel("value")

    if title:
        ax.set_title(title)

    if grid:
        ax.grid(True, linestyle='--', alpha=0.4)

    if legend and C > 1:
        ax.legend(loc='best')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig, ax


__all__ = ["plot_time_series"]
