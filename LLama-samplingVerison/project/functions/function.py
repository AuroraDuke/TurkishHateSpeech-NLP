import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functions.ann_model import ANN  # ANN class from ann_model.py
from .plots import plot_confusion_matrix, plot_val_train_loss


def model_training(vectorization_method, X_train, X_test,  y_train, y_test, len_labels):
    results = {
        'Model': [], 'Vectorization Method': [], 
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []
     }
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long, device=device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    
    print(f"\n{vectorization_method}:")
    input_size = X_train_tensor.shape[1]
    hidden_layer_sizes = [64, 32, 16]
    output_size = len_labels

    model = ANN(input_size, hidden_layer_sizes, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    batch_size = 1
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

        results['Model'].append('ANN')
        results['Vectorization Method'].append(vectorization_method)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1 Score'].append(f1)
        print(vectorization_method)
        print(f"{vectorization_method} :\nF1 Score {f1:.4f} | \nAcc:{accuracy:.4f} | Prec:{precision:.4f} | Recall:{recall:.4f}  ")
        
       
        
        plot_name= f"{vectorization_method}"
        plot_confusion_matrix(y_test_tensor.cpu(), predicted.cpu(),plot_name)
        plot_val_train_loss(num_epochs,train_losses,val_losses,plot_name)
      

    results_df = pd.DataFrame(results)
    
    #save model
    model_name= f"{vectorization_method}"
    model_path = f'models/{model_name}_ANN.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    #save measurement of model 
    results_df.to_csv(f'models/{model_name}_results.csv', index=False)
    print(f"Results saved to models/{model_name}_results.csv")

    return results_df
