import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def visualize_data_splits(*data_splits, label_mapping=None, dataset_names=None, title="Comparison of Data Distributions"):
    """
    Visualizes the class distributions of multiple data splits dynamically.

    Args:
        *data_splits (array-like): Variable number of datasets (e.g., y, y_train, y_test).
        label_mapping (dict, optional): A dictionary mapping numeric values to class labels.
        dataset_names (list, optional): A list of dataset names to use in the legend.
        title (str, optional): The title of the plot. Default is "Comparison of Data Distributions".
    """
    if len(data_splits) < 1:
        raise ValueError("At least one dataset must be provided.")

    if dataset_names and len(dataset_names) != len(data_splits):
        raise ValueError("The number of dataset names must match the number of data splits.")

    # If label mapping is provided, remap the numeric labels to class labels
    if label_mapping:
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        data_splits = [
            [inverse_mapping[label] if label in inverse_mapping else label for label in split]
            for split in data_splits
        ]

    # Calculate class distributions for each dataset
    counters = [Counter(split) for split in data_splits]

    # Identify and sort all unique classes
    def safe_sort_key(x):
        """Ensures safe sorting for mixed types of keys."""
        return (isinstance(x, str), x)

    all_classes = sorted(set().union(*[set(counter.keys()) for counter in counters]), key=safe_sort_key)

    # Prepare values for bar plot
    values = [[counter.get(cls, 0) for cls in all_classes] for counter in counters]

    # Plot the data
    x = np.arange(len(all_classes))  # Class indices
    width = 0.8 / len(data_splits)  # Adjust bar width dynamically

    plt.figure(figsize=(12, 6))
    for i, split_values in enumerate(values):
        label = dataset_names[i] if dataset_names else f"Dataset {i+1}"
        plt.bar(x + (i - len(data_splits) / 2) * width, split_values, width, label=label)

    # Adjust axis labels and titles
    plt.xlabel("Classes")
    plt.ylabel("Counts")
    plt.title(title)
    plt.xticks(x, all_classes, rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()
