# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:10:09 2024

@author: ilker
"""
import torch.nn as nn
import torch


class ANN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation='relu'):
        super(ANN, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Giriş katmanı
        prev_size = input_size
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            if activation == 'relu':
                self.activations.append(nn.ReLU())
            elif activation == 'tanh':
                self.activations.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.activations.append(nn.Sigmoid())
            prev_size = size

        # Çıkış katmanı
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)  



"""



class ANN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes, num_layers=1):
        super(ANN, self).__init__()

        # İlk RNN katmanı
        self.rnn1 = nn.RNN(input_size, hidden_size1, num_layers, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()

        # İkinci RNN katmanı
        self.rnn2 = nn.RNN(hidden_size1, hidden_size2, num_layers, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()

        # Üçüncü RNN katmanı
        self.rnn3 = nn.RNN(hidden_size2, hidden_size3, num_layers, batch_first=True)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.relu3 = nn.ReLU()

        # Çıkış katmanı
        self.fc4 = nn.Linear(hidden_size3, num_classes)

        # Dropout katmanı
        self.dropout = nn.Dropout(0.3)  # %30 dropout

    def forward(self, x):
        # İlk RNN katmanı
        x, _ = self.rnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        # İkinci RNN katmanı
        x, _ = self.rnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        # Üçüncü RNN katmanı
        x, _ = self.rnn3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        # Eğer sadece 2D veriniz varsa (batch_size, hidden_size) şeklinde, son çıkışı direkt alabilirsiniz.
        # x = self.fc4(x) şeklinde bir değişiklik yapılabilir.
        
        # Son zaman dilimini almayı denemek için:
        if len(x.shape) == 3:
            x = self.fc4(x[:, -1, :])  # Son zaman dilimindeki çıkışı al
        else:
            x = self.fc4(x)  # 2D veri ise doğrudan çıkışı al
        
        return x




"""

"""
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)  # İkinci ReLU aktivasyonu ekledik
        x = self.fc3(x)
        return x
"""

"""
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    """