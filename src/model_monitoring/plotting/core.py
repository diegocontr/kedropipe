from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global plotting configuration
_PLOT_CONFIG = {
    "annotation_fontsize": 8,
    "annotation_delta_factor": 0.03,
    "style": "seaborn-v0_8-whitegrid",
    "cmap": "viridis",
    "target_color": "#1E1D25",  # Dark grey color for target lines
    "h_line_style": "--",
    "ci_alpha": 0.2,
}


def set_plot_theme(theme_dict: Optional[dict] = None, **kwargs):
    """Set global plotting theme for model_monitoring plots.

    Can be called with a dictionary or with keyword arguments.

    Example:
    set_plot_theme({'annotation_fontsize': 10, 'style': 'ggplot'})
    set_plot_theme(annotation_fontsize=10, style='ggplot')
    """
    if theme_dict:
        _PLOT_CONFIG.update(theme_dict)
    _PLOT_CONFIG.update(kwargs)


def _get_colors(
    num_colors: int, cmap: Optional[str] = None, custom_colors: Optional[list] = None
) -> list:
    """Helper to get a list of colors."""
    if custom_colors and len(custom_colors) >= num_colors:
        return custom_colors[:num_colors]

    cmap = cmap or _PLOT_CONFIG["cmap"]
    colormap = plt.cm.get_cmap(cmap)
    return [colormap(i) for i in np.linspace(0, 1, num_colors)]


def _annotate_plot(
    ax,
    x_pos: np.ndarray,
    y_values: pd.Series,
    color: str = "black",
    y_label: Optional[str] = None,
):
    """Adds annotations to a plot, formatting them based on the y-axis label.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        x_pos (np.ndarray): The x positions for the annotations.
        y_values (pd.Series): The y values to annotate.
        color (str, optional): The color of the annotation text. Defaults to "black".
        y_label (Optional[str], optional): The y-axis label. If it contains '%',
                                           the annotations will be formatted as percentages.
                                           Defaults to None.
    """
    fontsize = _PLOT_CONFIG["annotation_fontsize"]
    delta_factor = _PLOT_CONFIG["annotation_delta_factor"]
    ymin, ymax = ax.get_ylim()
    delta = (ymax - ymin) * delta_factor

    # Determine the format string based on the y-label
    format_str = "{:.2f}%" if y_label and "%" in y_label else "{:.2f}"

    for x, y in zip(x_pos, y_values, strict=False):
        ax.text(
            x,
            y + delta,
            format_str.format(y),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def _get_ci_error(stats_df: pd.DataFrame, col: str) -> Optional[list]:
    """Calculate confidence interval errors."""
    low_col = f"{col}_low"
    up_col = f"{col}_up"
    if low_col in stats_df.columns and up_col in stats_df.columns:
        lower_error = stats_df[col] - stats_df[low_col]
        upper_error = stats_df[up_col] - stats_df[col]
        return [lower_error.to_numpy(), upper_error.to_numpy()]
    return None


def _draw_bar_plot(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    all_cols: list,
    all_colors: list,
    show_ci: bool,
):
    """Draws a bar plot on the given axes."""
    num_series = len(all_cols)
    total_width = 0.8
    width = total_width / num_series
    offsets = np.arange(num_series) * width - (total_width - width) / 2

    for i, col in enumerate(all_cols):
        yerr = _get_ci_error(stats_df, col) if show_ci else None
        ax.bar(
            x_pos + offsets[i],
            stats_df[col],
            width,
            label=col,
            color=all_colors[i],
            alpha=0.8,
            yerr=yerr,
            capsize=4 if yerr else 0,
        )


def _draw_line_plot(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    all_cols: list,
    all_colors: list,
    show_ci: bool,
):
    """Draws a line plot on the given axes."""
    for i, col in enumerate(all_cols):
        ax.plot(x_pos, stats_df[col], marker="o", label=col, color=all_colors[i])
        if show_ci:
            low_col = f"{col}_low"
            up_col = f"{col}_up"
            if low_col in stats_df.columns and up_col in stats_df.columns:
                ax.fill_between(
                    x_pos,
                    stats_df[low_col],
                    stats_df[up_col],
                    color=all_colors[i],
                    alpha=_PLOT_CONFIG["ci_alpha"],
                )


def _handle_annotations(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    main_cols: list,
    main_colors: list,
    annotate: Optional[bool],
    plot_type: str,
    y_label: Optional[str] = None,
):
    """Handles plotting annotations."""
    if annotate is None and plot_type == "line":
        annotate = True

    if annotate and main_cols:
        y_values = stats_df[main_cols[0]]
        annotation_color = main_colors[0]
        _annotate_plot(ax, x_pos, y_values, color=annotation_color, y_label=y_label)


def _draw_mean_lines(
    ax,
    agg_stats: Optional[pd.Series],
    show_mean_line: str,
    main_cols: list,
    all_cols: list,
    all_colors: list,
):
    """Draws horizontal mean lines."""
    if agg_stats is None or not show_mean_line or not main_cols:
        return

    line_style = _PLOT_CONFIG["h_line_style"]
    cols_to_line = []
    if show_mean_line == "first":
        cols_to_line = [main_cols[0]]
    elif show_mean_line == "all":
        cols_to_line = main_cols

    for col in cols_to_line:
        if col in agg_stats.index:
            color_index = all_cols.index(col)
            ax.axhline(
                y=agg_stats[col],
                color=all_colors[color_index],
                linestyle=line_style,
                alpha=0.9,
            )


def _plot_panel(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    main_cols: list,
    target_col: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
    legend_pos: Optional[str] = None,
    h_line: Optional[float] = None,
    plot_type: str = "bar",
    colors: Optional[list] = None,
    target_color: Optional[str] = None,
    cmap: Optional[str] = None,
    annotate: Optional[bool] = None,
    show_mean_line: str = "first",
    agg_stats: Optional[pd.Series] = None,
    show_ci: bool = False,
):
    """Generic panel plotting function."""
    main_cols = [main_cols] if isinstance(main_cols, str) else main_cols
    target_color = target_color or _PLOT_CONFIG["target_color"]
    main_colors = _get_colors(len(main_cols), cmap=cmap, custom_colors=colors)

    all_cols = main_cols + ([target_col] if target_col else [])
    all_colors = main_colors + ([target_color] if target_col else [])

    if plot_type == "bar":
        _draw_bar_plot(ax, stats_df, x_pos, all_cols, all_colors, show_ci)
    elif plot_type == "line":
        _draw_line_plot(ax, stats_df, x_pos, all_cols, all_colors, show_ci)

    _handle_annotations(
        ax, stats_df, x_pos, main_cols, main_colors, annotate, plot_type, y_label
    )
    _draw_mean_lines(ax, agg_stats, show_mean_line, main_cols, all_cols, all_colors)

    if h_line is not None:
        ax.axhline(h_line, color="black", linestyle="--", linewidth=1)

    if y_label:
        ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)

    if (len(all_cols) > 1 or target_col) and ax.get_legend_handles_labels() != (
        [],
        [],
    ):
        legend = ax.legend(loc=legend_pos, frameon=False)
        if legend:
            legend.set_zorder(100)


def _plot_panel_prediction_vs_target(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    pred_col: list,
    target_col: str,
    plot_type: str = "bar",
    colors: Optional[list] = None,
    target_color: Optional[str] = None,
    cmap: Optional[str] = None,
    annotate: Optional[bool] = None,
    show_mean_line: str = "first",
    agg_stats: Optional[pd.Series] = None,
    show_ci: bool = False,
    y_label: Optional[str] = "Prediction & Target",
    x_label: Optional[str] = None,
    legend_pos: Optional[str] = None,
):
    """Plots prediction vs. target values. Can handle multiple prediction columns."""
    _plot_panel(
        ax,
        stats_df,
        x_pos,
        main_cols=pred_col,
        target_col=target_col,
        y_label=y_label,
        x_label=x_label,
        legend_pos=legend_pos,
        plot_type=plot_type,
        colors=colors,
        target_color=target_color,
        cmap=cmap,
        annotate=annotate,
        show_mean_line=show_mean_line,
        agg_stats=agg_stats,
        show_ci=show_ci,
    )
    if ax.get_legend_handles_labels() != ([], []):
        ax.legend(loc=legend_pos, frameon=False)


def _plot_panel_spp(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    spp_col: list,
    plot_type: str = "bar",
    colors: Optional[list] = None,
    cmap: Optional[str] = None,
    annotate: Optional[bool] = None,
    show_mean_line: bool = False,
    agg_stats: Optional[pd.Series] = None,
    show_ci: bool = False,
    y_label: Optional[str] = "Ratio",
    x_label: Optional[str] = None,
    legend_pos: Optional[str] = None,
):
    """Plots the S/PP ratio. Can handle multiple S/PP columns."""
    _plot_panel(
        ax,
        stats_df,
        x_pos,
        main_cols=spp_col,
        y_label=y_label,
        x_label=x_label,
        legend_pos=legend_pos,
        h_line=1,
        plot_type=plot_type,
        colors=colors,
        cmap=cmap,
        annotate=annotate,
        show_mean_line=show_mean_line,
        agg_stats=agg_stats,
        show_ci=show_ci,
    )


def _plot_panel_loss_ratios(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    elr_col: list,
    lr_col: str,
    plot_type: str = "bar",
    colors: Optional[list] = None,
    target_color: Optional[str] = None,
    cmap: Optional[str] = None,
    annotate: Optional[bool] = None,
    show_mean_line: str = "first",
    agg_stats: Optional[pd.Series] = None,
    show_ci: bool = False,
    y_label: Optional[str] = "Ratio",
    x_label: Optional[str] = None,
    legend_pos: Optional[str] = None,
):
    """Plots Expected Loss Ratio vs. Loss Ratio. Can handle multiple ELR columns."""
    _plot_panel(
        ax,
        stats_df,
        x_pos,
        main_cols=elr_col,
        target_col=lr_col,
        y_label=y_label,
        x_label=x_label,
        legend_pos=legend_pos,
        plot_type=plot_type,
        colors=colors,
        target_color=target_color,
        cmap=cmap,
        annotate=annotate,
        show_mean_line=show_mean_line,
        agg_stats=agg_stats,
        show_ci=show_ci,
    )
    if ax.get_legend_handles_labels() != ([], []):
        ax.legend(loc=legend_pos, frameon=False)


def _plot_panel_metric(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    metric_col: list,
    title: Optional[str] = None,
    plot_type: str = "bar",
    colors: Optional[list] = None,
    cmap: Optional[str] = None,
    show_ci: bool = False,
    y_label: Optional[str] = "Value",
    x_label: Optional[str] = None,
    legend_pos: Optional[str] = None,
    **kwargs,
):
    """Plots a single generic metric, e.g., Gini, Exposure. Can handle multiple metric columns."""
    _plot_panel(
        ax,
        stats_df,
        x_pos,
        main_cols=metric_col,
        y_label=y_label,
        x_label=x_label,
        legend_pos=legend_pos,
        plot_type=plot_type,
        colors=colors,
        cmap=cmap,
        show_ci=show_ci,
    )


# Dictionary mapping panel types to their plotting functions
PANEL_FUNCTIONS = {
    "pred_vs_target": _plot_panel_prediction_vs_target,
    "spp": _plot_panel_spp,
    "loss_ratios": _plot_panel_loss_ratios,
    "metric": _plot_panel_metric,
    "exposure": _plot_panel_metric,
}


def _group_panel_configs(panel_configs: list) -> dict:
    """Groups panel configurations by title."""
    grouped_configs = {}
    for config in panel_configs:
        title = config.get("title")
        if not title:
            raise ValueError("Each panel configuration must have a 'title'.")
        if title not in grouped_configs:
            grouped_configs[title] = []
        grouped_configs[title].append(config)

    for title, configs in grouped_configs.items():
        if len(configs) > 2:
            raise ValueError(
                f"Cannot plot more than two charts in the same panel. Found {len(configs)} for title '{title}'."
            )
    return grouped_configs


def _setup_figure(
    num_panels: int,
    figsize: Optional[tuple] = None,
    figure_layout: Optional[dict] = None,
):
    """Sets up the matplotlib figure and axes."""
    plt.style.use(_PLOT_CONFIG["style"])

    if figure_layout:
        # User provides a custom layout
        layout_params = figure_layout.copy()
        fig_size = layout_params.pop("figsize", figsize)
        fig, axes = plt.subplots(figsize=fig_size, **layout_params)
    else:
        # Default behavior: one row of plots
        fig_size = figsize if figsize else (6 * num_panels, 5)
        fig, axes = plt.subplots(1, num_panels, figsize=fig_size, sharex=True)

    # Flatten axes array for easy iteration, and handle single subplot case
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    elif not isinstance(axes, list):
        axes = [axes]

    return fig, axes


def _plot_single_panel(
    ax,
    stats_df: pd.DataFrame,
    x_pos: np.ndarray,
    config: dict,
    agg_stats: Optional[pd.Series],
):
    """Plots a single panel on a given axis."""
    panel_type = config.pop("type")
    config.pop("title", None)
    if panel_type in PANEL_FUNCTIONS:
        plot_func = PANEL_FUNCTIONS[panel_type]
        try:
            plot_func(ax, stats_df, x_pos, agg_stats=agg_stats, **config)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error plotting '{panel_type}':\n{e}",
                ha="center",
                va="center",
                color="red",
            )
    else:
        ax.text(
            0.5,
            0.5,
            f"Unknown panel type:\n'{panel_type}'",
            ha="center",
            va="center",
        )


def plot_segment_statistics(
    stats_df: pd.DataFrame,
    panel_configs: list,
    agg_stats: Optional[pd.Series] = None,
    figsize: Optional[tuple] = None,
    figure_layout: Optional[dict] = None,
    show: bool = True,
):
    """Generates a custom panel-based plot for segment statistics.

    Args:
        stats_df (pd.DataFrame): DataFrame with statistics for a segment.
        panel_configs (list): A list of dictionaries, where each dictionary
                              configures a panel. Each config must have a 'title'.
        agg_stats (pd.Series, optional): Series with aggregate statistics for mean lines.
        figsize (tuple, optional): The figure size for the plot.
                                   Defaults to (6 * num_unique_titles, 5).
        figure_layout (dict, optional): A dictionary to specify a custom subplot
                                        layout. Passed directly to `plt.subplots`.
                                        Example: `{'nrows': 2, 'ncols': 1, 'sharex': True}`.
        show (bool, optional): Whether to display the figure via plt.show().
                               Defaults to True. Set to False in pipelines to
                               capture and save the figure programmatically.

    Returns:
        Optional[Tuple[plt.Figure, List]]: Returns (fig, axes) when panels are
        configured. Returns None if no panels are configured.
    """
    if not isinstance(stats_df, pd.DataFrame):
        raise TypeError("stats_df must be a pandas DataFrame.")

    grouped_configs = _group_panel_configs(panel_configs)
    num_panels = len(grouped_configs)
    if num_panels == 0:
        print("No panels configured to plot.")
        return None

    fig, axes = _setup_figure(num_panels, figsize, figure_layout)

    if len(axes) < num_panels:
        raise ValueError(
            f"Figure layout provides {len(axes)} axes, but {num_panels} panels are configured."
        )

    ax_map = {title: axes[i] for i, title in enumerate(grouped_configs.keys())}

    segment_name = (
        stats_df["segment"].iloc[0] if "segment" in stats_df.columns else "Segment"
    )
    x_labels = stats_df.index.astype(str)
    x_pos = np.arange(len(x_labels))

    def plot_panel(ax, configs):
        if not configs:
            return
        if not configs[0].get("title", "").startswith("#"):
            ax.set_title(configs[0].get("title", ""))
        _plot_single_panel(ax, stats_df, x_pos, configs[0].copy(), agg_stats)
        if len(configs) > 1:
            ax2 = ax.twinx()
            ax2.grid(False)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            _plot_single_panel(ax2, stats_df, x_pos, configs[1].copy(), agg_stats)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2:
                legend_pos = configs[0].get("legend_pos", "best")
                legend = ax2.legend(
                    lines + lines2, labels + labels2, loc=legend_pos, frameon=False
                )
                if legend:
                    legend.set_zorder(100)
            if ax.get_legend():
                ax.get_legend().remove()

    for title, configs in grouped_configs.items():
        plot_panel(ax_map[title], configs)

    fig.suptitle(f"Segment: {segment_name}", fontsize=16, weight="bold")

    for ax in axes:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show()

    return fig, axes
