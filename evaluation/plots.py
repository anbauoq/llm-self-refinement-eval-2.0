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

    # --- Figure setup ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=share_y)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    datasets = sorted(data['dataset'].unique())
    models = data['model_short'].unique()

    # Color map for models (consistent across plots)
    cmap = plt.get_cmap('tab10')
    model_colors = {model: cmap(i % 10) for i, model in enumerate(models)}

    for idx, dataset in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        dataset_data = data[data['dataset'] == dataset]

        for model in dataset_data['model_short'].unique():
            model_data = dataset_data[dataset_data['model_short'] == model].sort_values('max_tokes')

            category = model_data['model_category'].iloc[0]
            marker = 'o' if category == 'Reasoning' else 's'

            ax.plot(
                model_data['max_tokes'],
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

        # If y is NOT shared, each subplot should label its own y-axis
        if not share_y:
            ax.set_ylabel(ylabel)

    # If y IS shared, label only the left column (cleaner)
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

    models = aggregated_metrics['model_short'].unique()

    colors_reasoning = plt.cm.Set1(np.linspace(0, 0.5, len(models)))
    colors_non_reasoning = plt.cm.Set2(np.linspace(0, 0.5, len(models)))

    for i, model in enumerate(models):
        model_data = aggregated_metrics[
            aggregated_metrics['model_short'] == model
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
