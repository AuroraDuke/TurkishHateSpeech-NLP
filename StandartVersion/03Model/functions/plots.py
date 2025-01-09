import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix ,ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plots a confusion matrix for the given predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Title for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()
    
def plot_val_train_loss(num_epochs,train_losses,val_losses,plot_name):
      # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_epochs), train_losses, label="Train Loss", marker='o')
        plt.plot(range(num_epochs), val_losses, label="Validation Loss", marker='x')
        plt.title(plot_name)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

def plot_metrics_histogram(gram_name, results_df):
    """
    Plots metrics from a wide-format DataFrame as a grouped histogram.

    Args:
        gram_name (str): Name of the dataset or feature set.
        results_df (DataFrame): DataFrame with columns 'model', 'sampling', and metrics.
    """
    metrics = [col for col in results_df.columns if col not in ['model', 'sampling']]

    if not metrics:
        print("No metrics to plot. Please ensure the DataFrame contains metric columns.")
        return

    num_metrics = len(metrics)
    x = range(len(results_df['Vectorization Method'].unique()))
    width = 0.2  # Width of each bar

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    for i, metric in enumerate(metrics):
        pivoted_data = results_df.pivot(index='Vectorization Method', columns='Model', values=metric)
        if pivoted_data.isnull().values.all():
            print(f"Warning: Metric '{metric}' has no data and will be skipped.")
            continue

        for j, column in enumerate(pivoted_data.columns):
            values = pivoted_data[column].values
            positions = [pos + (i * len(pivoted_data.columns) + j) * width for pos in x]

            ax.bar(
                positions,
                values,
                width=width,
                label=f'{metric.replace("_", " ").capitalize()} - {column}',
                color=colors[(i * len(pivoted_data.columns) + j) % len(colors)],
                edgecolor='black'
            )

    ax.set_title(f'{gram_name} Metrics by Model and {vectorization_method}', fontsize=16)
    ax.set_xlabel('Vectorization Method', fontsize=14)
    ax.set_ylabel('Metrics (e.g., Accuracy, Precision, Recall, F1)', fontsize=14)
    ax.set_xticks([pos + width * (num_metrics * len(pivoted_data.columns) - 1) / 2 for pos in x])
    ax.set_xticklabels(results_df['Vectorization Method'].unique(), fontsize=12, rotation=45)
    ax.legend(title='Metric - Model', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_conclusion(df, model_col='Model', vectorization_col='Vectorization Method', measurements=None, rotate_xticks=0):
    """
    Plots a conclusion histogram for given measurements based on model and vectorization method.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        model_col (str): Column name for the model.
        vectorization_col (str): Column name for the vectorization method.
        measurements (list): List of measurement columns to plot.
        rotate_xticks (int): Degree of rotation for x-axis tick labels (default: 0).
    
    Returns:
        None
    """
    if measurements is None:
        measurements = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Combine category without modifying the original DataFrame
    categories = df[model_col] + " - " + df[vectorization_col]

    # Set up the plot
    plt.figure(figsize=(14, 12)) 
    bar_width = 0.2  # Width of each bar group
    x = range(len(categories))

    # Plot each measurement
    for i, measurement in enumerate(measurements):
        plt.bar(
            [p + i * bar_width for p in x],  # Bar positions
            df[measurement],  # Measurement values
            bar_width,  # Bar width
            label=measurement  # Measurement label
        )

    # Configure x-axis labels
    plt.xticks(
        [p + (bar_width * (len(measurements) - 1)) / 2 for p in x],
        categories,
        rotation=rotate_xticks  # Rotate labels by the given degree
    )

    # Add titles and labels
    plt.title("Conclusion Histogram by Model and Vectorization Method")
    plt.xlabel("Model - Vectorization Method")
    plt.ylabel("Measurement Value")
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

