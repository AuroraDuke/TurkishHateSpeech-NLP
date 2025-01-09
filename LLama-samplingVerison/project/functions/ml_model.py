import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import joblib
from .plots import plot_confusion_matrix


def ml_model_fit(X_train, X_test, y_train, y_test, model, model_name="Model", vectorization_method="Default"):
    """
    Trains the model, evaluates it on the test set, saves the model, and returns a summary DataFrame with key metrics.

    Parameters:
    - X_train, X_test: Training and test feature sets
    - y_train, y_test: Training and test labels
    - model: A scikit-learn compatible model
    - model_name: Name of the model for logging and reporting
    - vectorization_method: Description of vectorization or preprocessing applied

    Returns:
    - A pandas DataFrame summarizing the model's performance metrics.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Cross-validation scores for training accuracy
    train_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    training_accuracy = np.mean(train_scores)

    # Predictions and evaluation on test set
    y_pred = model.predict(X_test).ravel()
    validation_accuracy = accuracy_score(y_test, y_pred)

    # Compute metrics
    metrics = {
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "Training Accuracy": training_accuracy,
        "Validation Accuracy": validation_accuracy
    }

    # Print metrics
    print(f"Model Evaluation: {model_name}")
    print(" | ".join([f"{key}: {value:.5f}" for key, value in metrics.items()]))

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix: {model_name}")

    # Save the model
    model_dir = "ml_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"ml_{model_name}_{vectorization_method}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    # Results DataFrame
    results = pd.DataFrame({
        'Model': [model_name],
        'Vectorization Method': [vectorization_method],
        'Accuracy': [metrics["Accuracy"]],
        'Precision': [metrics["Precision"]],
        'Recall': [metrics["Recall"]],
        'F1 Score': [metrics["F1 Score"]]
    })

    return results


# Loop through models and evaluate
def evaluate_multiple_models(X_train, X_test, y_train, y_test, modelsAndNames, vectorization_method="Default"):
    all_results = []
    for model_name, model in modelsAndNames:
        print(f"\nEvaluating {model_name}...")
        result = ml_model_fit(X_train, X_test, y_train, y_test, model, model_name=model_name, vectorization_method=vectorization_method)
        all_results.append(result)
    return pd.concat(all_results, ignore_index=True)
