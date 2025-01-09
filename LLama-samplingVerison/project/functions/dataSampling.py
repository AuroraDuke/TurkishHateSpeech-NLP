# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:03:52 2024

@author: ilker
"""
#note: will can make batch,gpu,disk
from collections import Counter
import os
from imblearn.over_sampling import SMOTE
import numpy as np
from .functions_word2vec import transform_word2vec

def turkishWord2Vec(X_train, X_test, y_train, y_test,model_type="basic"):
   
    current_dir = os.getcwd()
    if model_type == "basic":
        model_path = os.path.join(current_dir, 'functions/turkishword2vec/trmodel')
    elif model_type == "finetune":
        model_path = os.path.join(current_dir, 'functions/turkishword2vec_Llama_finetune/trLlamaspeechmodel_finetuned.kv')
    else:
        raise ValueError("Model not found. Please use 'basic' or 'finetune'.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
     #convert to Word2Vec
    print("Word2Vec conversion is performed for X_train...")
    X_train_wv = transform_word2vec(X_train, model_path)

    print("Word2Vec conversion is performed for X_test...")
    X_test_wv = transform_word2vec(X_test, model_path)

    #check the output
    print(f"X_train_wv shape: {X_train_wv.shape}")
    print(f"X_test_wv shape: {X_test_wv.shape}")

    # Apply SMOTE to balance the dataset
    X_train_smote, y_train_smote = sampling(X_train_wv, y_train)
    print(f"x_train_smote shape: {X_train_smote.shape}, y_train_smote shape: {y_train_smote.shape}")

    # Return transformed and balanced datasets
    return X_train_smote, X_test_wv, y_train_smote, y_test



def makeVector(feature_union,X_train,X_test,y_train,y_test):
    X_train = feature_union.fit_transform(X_train).toarray()
    X_test = feature_union.transform(X_test).toarray()
    
    y_train = y_train.to_numpy()
    print(f"y_train shape: {y_train.shape}")
    
    feature_names = feature_union.get_feature_names_out()
    print("First 10 feature names:", feature_names[:10])
    
    print(f"X_train_features shape: {X_train.shape}")
    print(f"X_test_features shape: {X_test.shape}")
    
    X_train,y_train=sampling(X_train,y_train)
    print(f"X_train shape: {X_train.shape},y_train shape: {y_train.shape},")
    
    return X_train,X_test,y_train,y_test

    
def sampling(X_train,y_train):
    print(f"Before sampling {Counter(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After sampling {Counter(y_train_smote)}")
    return X_train_smote,y_train_smote
