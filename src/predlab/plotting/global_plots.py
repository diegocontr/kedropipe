from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .core import _PLOT_CONFIG, _get_colors


def plot_global_statistics(
    analyses: Dict[str, object],
    panel_configs: List[Dict],
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True,
):
    """Plot global analyses using analysis result objects.

    Parameters:
    - analyses: Dict mapping analysis type (e.g., 'lorenz_curve', 'calibration_curve', 'prediction_analysis')
      to an analysis result instance (subclass of GlobalAnalysis) that implements
      plot(ax, theme: dict, color_provider: Callable[[int], List[str]], panel_cfg: Optional[dict]).
    - panel_configs: List of panel config dicts, each with at least {'type': '<analysis_key>', 'title': '...'}
      and optional per-panel overrides like 'xscale', 'yscale', etc.
    - figsize: Optional tuple for overall figure size.
    - show: Whether to display the figure.
    """
    # Apply global style
    plt.style.use(_PLOT_CONFIG["style"])

    if not panel_configs:
        return None

    if figsize is None:
        figsize = (6 * len(panel_configs), 5)

    fig, axes = plt.subplots(1, len(panel_configs), figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    color_provider: Callable[[int], List[str]] = _get_colors
    theme = _PLOT_CONFIG

    for ax, cfg in zip(axes, panel_configs, strict=False):
        ptype = cfg.get("type")
        title = cfg.get("title", "")
        analysis_obj = analyses.get(ptype)
        if analysis_obj is None or not hasattr(analysis_obj, "plot"):
            ax.text(
                0.5,
                0.5,
                f"Analysis not available or does not implement plot(): {ptype}",
                ha="center",
                va="center",
            )
        else:
            # Delegate plotting to the analysis instance
            analysis_obj.plot(
                ax=ax, theme=theme, color_provider=color_provider, panel_cfg=cfg
            )
        if title and not title.startswith("#"):
            ax.set_title(title)

    fig.tight_layout()
    if show:
        plt.show()
        return None
    return fig, axes
