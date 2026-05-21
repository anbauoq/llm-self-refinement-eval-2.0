import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import plotly.graph_objects as go

def plot_by_dataset(data, metric_col, ylabel, title, share_y=False, save_path=None):
    """
    Faceted plot:
    - Color = model (consistent across subplots)
    - Single marker style for all models
    - One global legend for models
    - Y-axis always formatted as X.x
    """

    # --- Convert max_tokens like "max256" -> 256 ---
    data = data.copy()
    data["max_tokens_int"] = data["max_tokens"].str.replace("max", "", regex=False).astype(int)

    # --- Figure setup ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=share_y)
    fig.suptitle(title, fontsize=24, fontweight='bold')

    datasets = sorted(data['dataset'].unique())
    models = data['model'].unique()

    # Color map for models (consistent across plots)
    cmap = plt.get_cmap('tab10')
    model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

    y_formatter = mticker.FormatStrFormatter('%.1f')

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        dataset_data = data[data['dataset'] == dataset]

        for model in dataset_data['model'].unique():
            model_data = (
                dataset_data[dataset_data['model'] == model]
                .sort_values('max_tokens_int')
            )

            ax.plot(
                model_data['max_tokens_int'],
                model_data[metric_col],
                marker='o',
                linewidth=1.8,
                markersize=6,
                color=model_colors[model],
                label=model
            )

        ax.set_title(dataset.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Max Tokens')
        ax.grid(True, alpha=0.3)

        # Force integer ticks on x-axis
        ax.set_xticks(sorted(dataset_data["max_tokens_int"].unique()))

        # Force y-axis float format X.x
        ax.yaxis.set_major_formatter(y_formatter)

        if not share_y:
            ax.set_ylabel(ylabel)

    if share_y:
        axes[0, 0].set_ylabel(ylabel)
        axes[1, 0].set_ylabel(ylabel)

    # --- Global legend placed in empty subplot (bottom-right) ---
    legend_ax = axes[1, 2]  # last subplot (since 5 datasets → 6th is empty)
    legend_ax.axis('off')

    handles = [
        mlines.Line2D([], [], color=model_colors[m], label=m, linewidth=2.5)
        for m in models
    ]

    legend_ax.legend(
        handles=handles,
        loc='center',
        fontsize=13,
        title='Models',
        title_fontsize=15,
        frameon=False
    )

    # Hide unused axes
    for idx in range(len(datasets), 6):
        axes[idx // 3, idx % 3].axis('off')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_aggregated_metric_by_dataset(
    aggregated_metrics,
    metric_col,
    ylabel,
    title,
    save_path=None,
):
    """
    Universal bar plot for aggregated metrics.
    Compares models across datasets for a given metric.
    Legend is INSIDE the plot, upper right.
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    datasets_sorted = sorted(aggregated_metrics['dataset'].unique())
    x = np.arange(len(datasets_sorted))
    width = 0.08

    models = aggregated_metrics['model'].unique()
    # One color per model (no reuse); cycle through tab10 + Set3 if many models
    n = len(models)
    cmap1 = plt.cm.tab10
    cmap2 = plt.cm.Set3
    colors = [
        cmap1(i % 10) if i < 10 else cmap2((i - 10) % 12)
        for i in range(n)
    ]

    for i, model in enumerate(models):
        model_data = aggregated_metrics[
            aggregated_metrics['model'] == model
        ]
        model_category = model_data['model_category'].iloc[0]

        values = [
            model_data[model_data['dataset'] == ds][metric_col].values[0]
            if len(model_data[model_data['dataset'] == ds]) > 0 else 0
            for ds in datasets_sorted
        ]

        color = colors[i]
        hatch = '//' if model_category == 'Reasoning' else None

        ax.bar(
            x + i * width,
            values,
            width,
            label=model,
            color=color,
            hatch=hatch,
            edgecolor='black',
            linewidth=0.5
        )

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold')

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets_sorted)

    ax.legend(
        fontsize=14,
        loc='upper right',
        frameon=True
    )

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_model_category_comparison(aggregated_metrics, title_suffix="Averaged Across All Datasets", save_path=None):
    
    # Aggregate by model category
    category_comparison = aggregated_metrics.groupby('model_category').agg({
        'initial_accuracy': 'mean',
        'correction_rate': 'mean',
        'delta_accuracy': 'mean',
        'n_corrected': 'sum',
        'n_incorrect_initial': 'sum'
    }).reset_index()

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f'Reasoning vs Non-Reasoning Models ({title_suffix})',
        fontsize=14,
        fontweight='bold'
    )

    metrics_names = [
        'initial_accuracy',
        'delta_accuracy',
        'correction_rate'
    ]

    titles = [
        'Initial Accuracy (%)',
        'Delta Accuracy (%)',
        'Correction Rate (%)'
    ]

    for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[idx]

        categories = category_comparison['model_category']
        values = category_comparison[metric]

        bars = ax.bar(
            categories,
            values,
            color=['#ff7f0e', '#1f77b4'],
            edgecolor='black',
            linewidth=1.5,
            width=0.6
        )

        # Value labels
        for bar in bars:
            height = bar.get_height()
            if metric == 'n_corrected':
                label = f'{int(height)}'
            else:
                label = f'{height:.4f}'

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Model Category', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_metric_heatmap(
    df,
    value_col,
    *,
    index_col="model",
    columns_col="dataset",
    title=None,
    xlabel="Dataset",
    ylabel="Model",
    figsize=(12, 8),
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_label=None,
    sort_index=True,
    sort_columns=True,
    save_path=None,
):
    """
    Universal heatmap for aggregated metrics (or any df) using pivot(index_col, columns_col, value_col).

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str
        Column to plot as heatmap values.
    index_col : str
        Rows (e.g., 'model').
    columns_col : str
        Columns (e.g., 'dataset').
    title : str | None
        Plot title.
    cbar_label : str | None
        Colorbar label (defaults to value_col if not provided).
    sort_index, sort_columns : bool
        Whether to sort row/column labels.
    """

    pivot = df.pivot(index=index_col, columns=columns_col, values=value_col)

    if sort_index:
        pivot = pivot.sort_index()
    if sort_columns:
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=linewidths,
        cbar_kws={"label": cbar_label or value_col},
        ax=ax
    )

    ax.set_title(title or f"{value_col} Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_accuracy_vs_correction_scatter(
    df,
    x_col="initial_accuracy",
    y_col="post_hint_accuracy",
    category_col="model_category",
    title=None,
    xlabel="Initial Accuracy",
    ylabel="Post-Hint Accuracy",
    figsize=(12, 8),
    alpha=0.6,
    size=100,
    markers=None,
    edgecolor="black",
    linewidth=0.5,
    grid_alpha=0.3
):
    """
    Universal scatter plot for accuracy vs correction-style analyses.
    """

    if markers is None:
        markers = {
            "Reasoning": "o",
            "Non-Reasoning": "s"
        }

    fig, ax = plt.subplots(figsize=figsize)

    for category in df[category_col].unique():
        sub = df[df[category_col] == category]

        marker = markers.get(category, "o")

        ax.scatter(
            sub[x_col],
            sub[y_col],
            label=category,
            alpha=alpha,
            s=size,
            marker=marker,
            edgecolors=edgecolor,
            linewidth=linewidth
        )

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    ax.set_title(
        title or f"{x_col} vs {y_col}",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(fontsize=11)
    ax.grid(True, alpha=grid_alpha)

    plt.tight_layout()
    plt.show()


def plot_four_category_boxplot(
    data,
    value_col,
    title,
    ylabel="Tokens",
    show_fliers=False
):
    """
    Creates boxplots for:
    - Reasoning + Corrected
    - Reasoning + Not corrected
    - Non-Reasoning + Corrected
    - Non-Reasoning + Not corrected

    Parameters:
        data : DataFrame (per_question_tokens)
        value_col : column to plot (e.g. 'hint_tokens')
        title : plot title
        ylabel : y-axis label
        show_fliers : whether to show outliers
    """

    reasoning_corrected = data[
        (data["model_category"] == "Reasoning") &
        (data["hint_outcome"] == "Corrected")
    ][value_col]

    reasoning_not = data[
        (data["model_category"] == "Reasoning") &
        (data["hint_outcome"] == "Not corrected")
    ][value_col]

    nonreasoning_corrected = data[
        (data["model_category"] == "Non-Reasoning") &
        (data["hint_outcome"] == "Corrected")
    ][value_col]

    nonreasoning_not = data[
        (data["model_category"] == "Non-Reasoning") &
        (data["hint_outcome"] == "Not corrected")
    ][value_col]

    data_to_plot = [
        reasoning_corrected,
        reasoning_not,
        nonreasoning_corrected,
        nonreasoning_not,
    ]

    labels = [
        "Reasoning\nCorrected",
        "Reasoning\nNot corrected",
        "Non-Reasoning\nCorrected",
        "Non-Reasoning\nNot corrected",
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=labels, showfliers=show_fliers)

    plt.title(title, fontweight="bold")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_tokens_boxplot(
    df,
    value_col,
    group_col,
    title=None,
    ylabel="Tokens",
    show_fliers=False,
    sort="alpha",
    rotation=45,
    figsize=(12, 6),
):
    """
    Universal boxplot:
      - df: DataFrame (e.g., per_question_tokens)
      - value_col: numeric column to plot (e.g., "hint_tokens")
      - group_col: column to group by (e.g., "model", "dataset", "model_category", "max_tokens")
      - sort: "alpha" (label order) or "median" (by median of value_col)
    """

    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in df columns.")
    if value_col not in df.columns:
        raise KeyError(f"'{value_col}' not found in df columns.")

    sub = df[[group_col, value_col]].dropna()

    # Determine group order
    groups = sub[group_col].astype(str)
    sub = sub.assign(_group=groups)

    if sort == "median":
        order = (
            sub.groupby("_group")[value_col]
            .median()
            .sort_values(ascending=True)
            .index.tolist()
        )
    else:  # "alpha"
        order = sorted(sub["_group"].unique())

    data = [sub.loc[sub["_group"] == g, value_col].values for g in order]

    if title is None:
        title = f"{value_col} by {group_col}"

    plt.figure(figsize=figsize)
    plt.boxplot(data, labels=order, showfliers=show_fliers)
    plt.xticks(rotation=rotation, ha="right")

    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

def plot_token_stage_boxplot(per_question_tokens, title="Token Usage Across Generation Stages", show_fliers=False):

    data = [
        per_question_tokens["initial_inference_tokens"],
        per_question_tokens["hint_tokens"],
        per_question_tokens["post_hint_inference_tokens"]
    ]

    labels = [
        "Initial inference",
        "Hints",
        "Post-hint inference"
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, showfliers=show_fliers)

    plt.ylabel("Generated tokens")
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_delta_accuracy_grouped_barchart(
    metrics,
    value_col="delta_accuracy",
    group_col="model",
    hue_col="dataset",
    title="Accuracy Gain After Hint-Based Correction by Model and Dataset",
    ylabel="Δ Accuracy (%)",
    figsize=(12, 5),
    rotation=45,
):
    """
    Grouped bar chart:
        - x-axis = group_col (e.g., model)
        - bar color = hue_col (e.g., dataset)
        - bar height = mean(value_col)

    Parameters
    ----------
    metrics : DataFrame
    value_col : str
    group_col : str
    hue_col : str
    """

    # --- aggregate ---
    grouped = (
        metrics
        .groupby([group_col, hue_col])[value_col]
        .mean()
        .unstack()
    )

    groups = grouped.index
    hues = grouped.columns

    x = np.arange(len(groups))
    width = 0.8 / len(hues)

    plt.figure(figsize=figsize)

    # --- bars ---
    for i, h in enumerate(hues):
        plt.bar(
            x + i * width,
            grouped[h],
            width,
            label=h
        )

    # --- formatting ---
    plt.ylabel(ylabel)
    plt.xlabel(group_col.capitalize())
    plt.title(title)
    plt.xticks(
        x + width * (len(hues) - 1) / 2,
        groups,
        rotation=rotation,
        ha="right"
    )

    plt.legend(title=hue_col.capitalize())
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_tokens_barchart(
    df,
    value_col,
    group_col,
    title=None,
    ylabel="Tokens",
    sort="alpha",
    agg="mean",          # "mean" or "median"
    rotation=45,
    figsize=(12, 6),
    save_path=None,
):
    """
    Universal bar chart (aggregation version of boxplot).

      - df: DataFrame (e.g., per_question_tokens)
      - value_col: numeric column (e.g., "hint_tokens")
      - group_col: grouping column (e.g., "model", "dataset", etc.)
      - sort: "alpha" or "median"
      - agg: "mean" or "median"
    """

    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in df columns.")
    if value_col not in df.columns:
        raise KeyError(f"'{value_col}' not found in df columns.")

    sub = df[[group_col, value_col]].dropna().copy()
    sub["_group"] = sub[group_col].astype(str)

    # --- aggregation function ---
    reducer = np.nanmedian if agg == "median" else np.nanmean

    # --- determine order ---
    if sort == "median":
        order = (
            sub.groupby("_group")[value_col]
            .median()
            .sort_values(ascending=True)
            .index.tolist()
        )
    else:
        order = sorted(sub["_group"].unique())

    # --- compute aggregated values ---
    values = [
        reducer(sub.loc[sub["_group"] == g, value_col].values)
        for g in order
    ]

    if title is None:
        title = f"{value_col} by {group_col}"

    # --- plot ---
    plt.figure(figsize=figsize)
    plt.bar(order, values)

    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_token_stage_barchart(
    per_question_tokens,
    title="Average Token Usage Across Generation Stages",
    agg="mean",   # "mean" or "median"
    save_path=None,
):
    """
    Bar chart version of token usage across stages.

    Parameters
    ----------
    per_question_tokens : DataFrame
    title : str
    agg : str
        Aggregation method: "mean" or "median"
    """

    # --- choose aggregation ---
    if agg == "median":
        reducer = np.nanmedian
    else:
        reducer = np.nanmean

    # --- compute aggregated values ---
    values = [
        reducer(per_question_tokens["initial_inference_tokens"]),
        reducer(per_question_tokens["hint_tokens"]),
        reducer(per_question_tokens["post_hint_inference_tokens"]),
    ]

    labels = [
        "Initial inference",
        "Hints",
        "Post-hint inference"
    ]

    # --- plot ---
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)

    plt.ylabel("Generated tokens")
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_outcome_by_model_category_barchart(
    data,
    value_col,
    title,
    ylabel="Tokens",
    agg="mean",          # "mean" or "median"
    figsize=(8.5, 5.2),
    rotation=0,
    save_path=None,
):
    """
    Appealing grouped bar chart:
      X-axis  = model_category (Reasoning / Non-Reasoning)
      Color   = hint_outcome (Corrected=green, Not corrected=red)
      Y-axis  = aggregated tokens (mean/median)
      No error bars, no n labels.
    """

    # --- choose aggregation ---
    reducer = np.nanmedian if agg == "median" else np.nanmean

    model_categories = ["Reasoning", "Non-Reasoning"]
    outcomes = ["Corrected", "Not corrected"]

    # --- compute values (robust to empty groups) ---
    vals = np.full((len(model_categories), len(outcomes)), np.nan, dtype=float)
    for i, mc in enumerate(model_categories):
        for j, oc in enumerate(outcomes):
            s = data.loc[
                (data["model_category"] == mc) &
                (data["hint_outcome"] == oc),
                value_col
            ].dropna()
            vals[i, j] = reducer(s.values) if len(s) else np.nan

    # --- plotting ---
    x = np.arange(len(model_categories))
    width = 0.34

    colors = {
        "Corrected": "forestgreen",
        "Not corrected": "firebrick",
    }

    plt.figure(figsize=figsize)

    # Bars (centered around each x tick)
    for j, oc in enumerate(outcomes):
        plt.bar(
            x + (j - 0.5) * width,
            vals[:, j],
            width=width,
            label=oc,
            color=colors[oc],
            edgecolor="white",
            linewidth=1.2
        )

    # Labels & title
    plt.xticks(x, model_categories, rotation=rotation)
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")

    # Subtle grid + clean spines
    ax = plt.gca()
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend (compact)
    plt.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def _fmt_sankey(x):
    return f"{int(x):,}"


def _rgba_sankey(hex_color, a=0.55):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def plot_hint_correction_sankey(metrics):
    """
    Plot Sankey diagram of hint correction flow.

    metrics: DataFrame with columns n_questions, n_incorrect_initial, n_corrected.
    Returns the Plotly figure.
    """
    n_init_correct = (metrics["n_questions"] - metrics["n_incorrect_initial"]).sum()
    n_init_incorrect = metrics["n_incorrect_initial"].sum()
    n_corrected = metrics["n_corrected"].sum()
    n_still_wrong = n_init_incorrect - n_corrected
    total = n_init_correct + n_init_incorrect

    def _pct(x):
        return f"{100 * x / total:.1f}%" if total else "0%"

    GREY = "#6B7280"
    GREEN = "#37AD69"
    RED = "#c63f5f"
    BG = "#FAFBFD"

    SCALE = 1.35
    NODE_PAD = 26
    NODE_THICKNESS = 18
    LINK_OUTLINE_ALPHA = 0.85
    LINK_OUTLINE_WIDTH = 1
    ALPHA_NODE = 0.95
    ALPHA_LINK_MAIN = 0.55
    ALPHA_LINK_LIGHT = 0.42

    x = [0.02, 0.36, 0.36, 0.94, 0.94]
    y = [0.50, 0.20, 0.96, 0.14, 0.86]

    node_colors = [
        _rgba_sankey(GREY, ALPHA_NODE),
        _rgba_sankey(GREEN, ALPHA_NODE),
        _rgba_sankey(RED, ALPHA_NODE),
        _rgba_sankey(GREEN, ALPHA_NODE),
        _rgba_sankey(RED, ALPHA_NODE),
    ]

    sources = [0, 0, 1, 2, 2]
    targets = [1, 2, 3, 3, 4]

    true_vals = [
        int(n_init_correct),
        int(n_init_incorrect),
        int(n_init_correct),
        int(n_corrected),
        int(n_still_wrong),
    ]

    vis_vals = [v * SCALE for v in true_vals]

    link_colors = [
        _rgba_sankey(GREEN, ALPHA_LINK_LIGHT),
        _rgba_sankey(RED, ALPHA_LINK_LIGHT),
        _rgba_sankey(GREEN, 0.40),
        _rgba_sankey(GREEN, 0.70),
        _rgba_sankey(RED, ALPHA_LINK_MAIN),
    ]

    hover_txt = [
        f"Total → Initially correct: {true_vals[0]:,} ({_pct(true_vals[0])})",
        f"Total → Initially incorrect: {true_vals[1]:,} ({_pct(true_vals[1])})",
        f"Initially correct → Final correct: {true_vals[2]:,} ({_pct(true_vals[2])})",
        f"Initially incorrect → Final correct (Corrected): {true_vals[3]:,} ({_pct(true_vals[3])})",
        f"Initially incorrect → Final incorrect: {true_vals[4]:,} ({_pct(true_vals[4])})",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                domain=dict(x=[0, 1], y=[0.02, 0.82]),
                node=dict(
                    pad=NODE_PAD,
                    thickness=NODE_THICKNESS,
                    label=[""] * 5,
                    color=node_colors,
                    x=x,
                    y=y,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[
                        "Total",
                        f"Initially correct: {_pct(n_init_correct)}",
                        f"Initially incorrect: {_pct(n_init_incorrect)}",
                        f"Final correct: {_pct(n_init_correct + n_corrected)}",
                        f"Final incorrect: {_pct(n_still_wrong)}",
                    ],
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=vis_vals,
                    color=link_colors,
                    customdata=hover_txt,
                    hovertemplate="%{customdata}<extra></extra>",
                    line=dict(
                        color=f"rgba(255,255,255,{LINK_OUTLINE_ALPHA})",
                        width=LINK_OUTLINE_WIDTH,
                    ),
                ),
            )
        ]
    )

    label_bg = "rgba(245,245,245,0.78)"
    font = dict(size=16, family="Inter, Arial", color="#111")

    def add_box(px, py, text, anchor="center"):
        fig.add_annotation(
            x=px,
            y=py,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            font=font,
            bgcolor=label_bg,
            borderwidth=0,
            borderpad=10,
            xanchor=anchor,
            yanchor="middle",
            align="left",
        )

    add_box(0.23, 0.70, f"Initially correct<br><b>{_pct(n_init_correct)}</b>")
    add_box(0.23, 0.06, f"Initially incorrect<br><b>{_pct(n_init_incorrect)}</b>")
    add_box(0.60, 0.25, f"Corrected<br><b>{_pct(n_corrected)}</b>")
    add_box(0.82, 0.80, f"Final correct<br><b>{_pct(n_init_correct + n_corrected)}</b>", "left")
    add_box(0.82, 0.15, f"Final incorrect<br><b>{_pct(n_still_wrong)}</b>", "left")

    fig.update_layout(
        title=dict(
            text="Accuracy before and after hint injection",
            x=0.5,
            font=dict(size=28, family="Inter, Arial", color="#111"),
        ),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=20, r=20, t=90, b=20),
        width=640,
        height=520,
        font=dict(size=13, family="Inter, Arial", color="#111"),
    )

    return fig

def plot_hintlen_vs_accuracy_gain(
    merged_df,
    x_col="avg_hint_tokens",
    y_col="delta_accuracy",
    label_col="model",
    category_col="model_category",
    title="Hint Length vs Accuracy Gain",
    figsize=(10, 7),
    s=150,
    alpha=0.88,
    fontsize=10,
    save_path=None,
):
    df = merged_df.copy()

    MODEL_LABELS = {
        "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-Qwen-1.5B",
        "DeepSeek-R1-Distill-Llama-8B":  "DS-R1-Llama-8B",
    }

    # Marker by category: Reasoning = square, non-Reasoning = circle
    MARKER_BY_CATEGORY = {"Reasoning": "s", "Non-Reasoning": "o"}
    if category_col in df.columns:
        # normalize common variants
        def _norm_cat(val):
            v = str(val).strip().lower()
            return "Reasoning" if v == "reasoning" else "Non-Reasoning"
    else:
        def _norm_cat(val):
            return "Non-Reasoning"

    # Pretty, distinct colors per model
    PALETTE = [
        "#2E86AB",  # steel blue
        "#A23B72",  # magenta
        "#F18F01",  # amber
        "#C73E1D",  # terracotta
        "#3B1F2B",  # dark plum
        "#95C623",  # lime
        "#6A4C93",  # purple
        "#E07A5F",  # coral
        "#81B29A",  # sage
        "#5C4D7D",  # slate violet
    ]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    models = df[label_col].unique()
    for i, model in enumerate(models):
        subset = df[df[label_col] == model]
        color = PALETTE[i % len(PALETTE)]
        display_name = MODEL_LABELS.get(str(model), str(model))
        marker = "o"
        if category_col in df.columns and len(subset) > 0:
            cat = _norm_cat(subset[category_col].iloc[0])
            marker = MARKER_BY_CATEGORY.get(cat, "o")
        ax.scatter(
            subset[x_col],
            subset[y_col],
            color=color,
            s=s,
            alpha=alpha,
            marker=marker,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
            label=display_name,
        )

    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_margin = (x_max - x_min) * 0.08 or 40
    ax.set_xlim(max(0, x_min - x_margin), x_max + x_margin)
    ax.set_xlabel("Average Hint Length (tokens)", fontsize=12, color="#333")
    ax.set_ylabel("Delta Accuracy (%)", fontsize=12, color="#333")
    ax.set_title(title, fontsize=18, fontweight="600", color="#222", pad=12)
    ax.legend(
        loc="best",
        fontsize=fontsize,
        frameon=True,
        framealpha=0.95,
        edgecolor="#ddd",
        fancybox=True,
    )
    ax.grid(True, alpha=0.35, linestyle="-", color="#ccc")
    ax.tick_params(colors="#444", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
