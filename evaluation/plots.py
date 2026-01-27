import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def plot_by_dataset(data, metric_col, ylabel, title, share_y=False):
    """
    Faceted plot:
    - Color = model (consistent across subplots)
    - Marker = Reasoning vs Non-Reasoning
    - One global legend for models + one for markers
    - share_y controls shared vs per-dataset y scale
    """

    # --- Convert max_tokens like "max256" -> 256 ---
    data = data.copy()
    data["max_tokens_int"] = data["max_tokens"].str.replace("max", "", regex=False).astype(int)

    # --- Figure setup ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=share_y)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    datasets = sorted(data['dataset'].unique())
    models = data['model'].unique()

    # Color map for models (consistent across plots)
    cmap = plt.get_cmap('tab10')
    model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        dataset_data = data[data['dataset'] == dataset]

        for model in dataset_data['model'].unique():
            model_data = dataset_data[
                dataset_data['model'] == model
            ].sort_values('max_tokens_int')

            category = model_data['model_category'].iloc[0]
            marker = 'o' if category == 'Reasoning' else 's'

            ax.plot(
                model_data['max_tokens_int'],
                model_data[metric_col],
                marker=marker,
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

        if not share_y:
            ax.set_ylabel(ylabel)

    if share_y:
        axes[0, 0].set_ylabel(ylabel)
        axes[1, 0].set_ylabel(ylabel)

    # --- Global legend (models) ---
    handles = [
        mlines.Line2D([], [], color=model_colors[m], label=m, linewidth=2)
        for m in models
    ]
    fig.legend(handles=handles, loc='center right', fontsize=9, title='Models')

    # --- Marker legend (categories) ---
    reasoning = mlines.Line2D([], [], color='black', marker='o',
                              linestyle='None', label='Reasoning')
    non_reasoning = mlines.Line2D([], [], color='black', marker='s',
                                  linestyle='None', label='Non-Reasoning')
    fig.legend(handles=[reasoning, non_reasoning],
               loc='lower center', ncol=2, fontsize=9, title='Model Type')

    # Hide unused axes
    for idx in range(len(datasets), 6):
        axes[idx // 3, idx % 3].axis('off')

    plt.tight_layout(rect=[0, 0.08, 0.85, 0.95])
    plt.show()


def plot_aggregated_metric_by_dataset(
    aggregated_metrics,
    metric_col,
    ylabel,
    title
):
    """
    Universal bar plot for aggregated metrics.
    Compares models across datasets for a given metric.
    Legend is INSIDE the plot, on the LEFT.
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    datasets_sorted = sorted(aggregated_metrics['dataset'].unique())
    x = np.arange(len(datasets_sorted))
    width = 0.08

    models = aggregated_metrics['model'].unique()

    colors_reasoning = plt.cm.Set1(np.linspace(0, 0.5, len(models)))
    colors_non_reasoning = plt.cm.Set2(np.linspace(0, 0.5, len(models)))

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

        color = (
            colors_reasoning[i]
            if model_category == 'Reasoning'
            else colors_non_reasoning[i]
        )
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
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets_sorted)

    # -------- LEGEND INSIDE, LEFT --------
    ax.legend(
        fontsize=9,
        loc='upper left',
        frameon=True
    )

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_model_category_comparison(aggregated_metrics, title_suffix="Averaged Across All Datasets"):
    
    # Aggregate by model category
    category_comparison = aggregated_metrics.groupby('model_category').agg({
        'initial_accuracy': 'mean',
        'correction_rate': 'mean',
        'delta_accuracy': 'mean',
        'n_corrected': 'sum',
        'n_incorrect_initial': 'sum'
    }).reset_index()

    # Plot setup
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f'Reasoning vs Non-Reasoning Models ({title_suffix})',
        fontsize=14,
        fontweight='bold'
    )

    metrics_names = [
        'initial_accuracy',
        'correction_rate',
        'n_corrected',
        'delta_accuracy'
    ]

    titles = [
        'Initial Accuracy',
        'Correction Rate',
        'Total Corrections',
        'Delta Accuracy'
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
    sort_columns=True
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
    plt.show()


def plot_accuracy_vs_correction_scatter(
    df,
    x_col="initial_accuracy",
    y_col="post_hint_accuracy",
    category_col="model_category",
    title=None,
    xlabel="Initial Accuracy",
    ylabel="Correction Rate (% Incorrect Corrected)",
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
