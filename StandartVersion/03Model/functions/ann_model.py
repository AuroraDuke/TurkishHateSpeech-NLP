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

