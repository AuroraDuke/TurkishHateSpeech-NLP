import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from functions.ann_model import ANN  # ANN class from ann_model.py

def plot_metrics_from_wide_df(gram_name, results_df):
    """
    Plots metrics from a wide-format DataFrame.

    Args:
        gram_name (str): Name of the dataset or feature set.
        results_df (DataFrame): DataFrame with columns 'model', 'sampling', and metrics.
    """
    metrics = [col for col in results_df.columns if col not in ['model', 'sampling']]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        pivoted_data = results_df.pivot(index='sampling', columns='model', values=metric)
        pivoted_data.plot(kind='bar', ax=ax, alpha=0.8, edgecolor='black')
        plt.title(f'{gram_name} {metric.replace("_", " ").capitalize()} by Model and Sampling')
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.xlabel('Sampling Method')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()

def plot_class_distribution(y, title):
    """
    Plots the class distribution of the dataset.

    Args:
        y (Series): Target labels.
        title (str): Title for the plot.
    """
    unique_counts = y.value_counts()
    label_mapping = {0: 'None', 1: 'Hate', 2: 'Offensive'}
    unique_counts.index = unique_counts.index.map(label_mapping)

    plt.bar(unique_counts.index.tolist(), unique_counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title(f"{title} Dataset")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

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

def evaluate_model_with_metrics(gram_name, model_name, sampling, model, X_train, y_train, X_test, y_test):
    """
    Evaluates the model using various metrics and returns the results.

    Args:
        gram_name (str): Dataset name.
        model_name (str): Model name.
        sampling (str): Sampling method.
        model: The model to evaluate.
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.

    Returns:
        tuple: Metrics (accuracy, precision, recall, f1, training_accuracy, validation_accuracy).
    """
    model.fit(X_train, y_train)

    train_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    training_accuracy = np.mean(train_scores)

    y_pred = model.predict(X_test).ravel()
    validation_accuracy = accuracy_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{model_name}, {sampling}, {gram_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Training Accuracy: {training_accuracy}")
    print(f"Validation Accuracy: {validation_accuracy}")

    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix: {model_name} - {sampling}")

    return accuracy, precision, recall, f1, training_accuracy, validation_accuracy

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
    x = range(len(results_df['sampling'].unique()))
    width = 0.2  # Width of each bar

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    for i, metric in enumerate(metrics):
        pivoted_data = results_df.pivot(index='sampling', columns='model', values=metric)
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

    ax.set_title(f'{gram_name} Metrics by Model and Sampling', fontsize=16)
    ax.set_xlabel('Sampling Method', fontsize=14)
    ax.set_ylabel('Metrics (e.g., Accuracy, Precision, Recall, F1)', fontsize=14)
    ax.set_xticks([pos + width * (num_metrics * len(pivoted_data.columns) - 1) / 2 for pos in x])
    ax.set_xticklabels(results_df['sampling'].unique(), fontsize=12, rotation=45)
    ax.legend(title='Metric - Model', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def model_training(gram_name, X_train, X_test, y_train, y_test,samplings,X_train_features,X_test_features, len_labels):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long, device=device)

    results = {'model': [], 'sampling': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for sampling_method, (X_train_res, y_train_res) in samplings.items():
        X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train_res, dtype=torch.long, device=device)
        
        print(f"\n{sampling_method}:")
        input_size = X_train_tensor.shape[1]
        hidden_layer_sizes = [64, 32, 16]
        output_size = len_labels

        model = ANN(input_size, hidden_layer_sizes, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
        batch_size = 128
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            correct_train = 0
            total_train = 0

            for i in range(0, len(X_train_tensor), batch_size):
                X_batch = X_train_tensor[i:i+batch_size]
                y_batch = y_train_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == y_batch).sum().item()
                total_train += y_batch.size(0)

            average_train_loss = epoch_loss / num_batches
            train_losses.append(average_train_loss)

            train_accuracy = correct_train / total_train

            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                val_loss = criterion(outputs, y_test_tensor).item()
                val_losses.append(val_loss)

                _, predicted_test = torch.max(outputs, 1)
                val_accuracy = (predicted_test == y_test_tensor).sum().item() / y_test_tensor.size(0)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)

            accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
            precision = precision_score(y_test_tensor.cpu(), predicted.cpu(), average='weighted')
            recall = recall_score(y_test_tensor.cpu(), predicted.cpu(), average='weighted')
            f1 = f1_score(y_test_tensor.cpu(), predicted.cpu(), average='weighted')

            results['model'].append('ANN')
            results['sampling'].append(sampling_method)
            results['accuracy'].append(accuracy)
            print(accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)

            model_path = f'models/{gram_name}_{sampling_method}_ANN.pth'
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')

            plot_confusion_matrix(y_test_tensor.cpu(), predicted.cpu(), f"Confusion Matrix: {gram_name} - {sampling_method}")

        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_epochs), train_losses, label="Train Loss", marker='o')
        plt.plot(range(num_epochs), val_losses, label="Validation Loss", marker='x')
        plt.title(f"{gram_name} - {sampling_method} Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    results_df = pd.DataFrame(results)
   # plot_metrics_from_wide_df(gram_name, results_df)
    plot_metrics_histogram(gram_name, results_df)
    results_df.to_csv(f'models/{gram_name}_results.csv', index=False)
    print(f"Results saved to models/{gram_name}_results.csv")

    return results_df