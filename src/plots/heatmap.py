import typing as T

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_2d_heatmap(
    df: pd.DataFrame,
    x_axis_feature_name: str,
    y_axis_feature_name: str,
    value_feature_name: str,
    label: str,
    title: str,
    save_path: T.Optional[str] = None,
    figsize: T.Tuple[int, int] = (10, 8),
    cmap: str = "YlOrRd_r",  # Yellow-Orange-Red reversed (lighter = better)
) -> None:
    pivot_df = df.pivot(
        index=y_axis_feature_name,
        columns=x_axis_feature_name,
        values=value_feature_name,
    ).iloc[::-1]

    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar_kws={"label": label},
    )

    plt.title(title)
    plt.xlabel(x_axis_feature_name)
    plt.ylabel(y_axis_feature_name)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_4d_heatmap_grid(
    df,
    outer_x: str,
    outer_y: str,
    heatmap_x: str,
    heatmap_y: str,
    value: str,
    cmap="viridis",
):
    outer_x_vals = sorted(df[outer_x].unique())
    outer_y_vals = sorted(df[outer_y].unique(), reverse=True)

    vmin = df[value].min()
    vmax = df[value].max()

    fig, axes = plt.subplots(
        nrows=len(outer_y_vals),
        ncols=len(outer_x_vals),
        figsize=(3 * len(outer_x_vals), 3 * len(outer_y_vals)),
        sharex=True,
        sharey=True,
    )

    if len(outer_y_vals) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(outer_x_vals) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, oy in enumerate(outer_y_vals):
        for j, ox in enumerate(outer_x_vals):
            ax = axes[i, j]
            subset = df[(df[outer_x] == ox) & (df[outer_y] == oy)]
            heatmap_data = subset.pivot(
                index=heatmap_y, columns=heatmap_x, values=value
            )
            heatmap_data = heatmap_data.sort_index(ascending=False)
            sns.heatmap(
                heatmap_data,
                ax=ax,
                annot=True,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                cbar=False,
            )
            title_text = f"{outer_x}={ox}\n{outer_y}={oy}"
            ax.set_title(title_text, fontsize=10)
            ax.set_xlabel(heatmap_x)
            ax.set_ylabel(heatmap_y)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
