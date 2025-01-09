# Twitter Hate Speech Detection

## Overview
This project focuses on detecting hate speech in Turkish tweets using machine learning (ML) and deep learning (DL) models. By leveraging synthetic and non-synthetic datasets, the project compares different vectorization methods and model performances to identify the most effective techniques for hate speech detection.

---

## Objectives
- Detect hate speech in Turkish tweets.
- Compare the performance of synthetic and non-synthetic datasets.
- Evaluate vectorization methods and modeling approaches for optimal performance.

---

## Methodology

### Introduction
Social media platforms provide an environment where individuals can freely express their views, but they have also become a medium where hate speech can spread rapidly. Especially on platforms with large user bases such as Twitter, the detection of hate speech has become an important issue in terms of digital security and public health. In this study, machine learning and deep learning methods were used with synthetic and non-synthetic datasets to detect hate speech in texts shared on the Twitter platform. The main purpose of the study is to compare the results obtained with the two types of datasets and to reveal the most effective modeling approaches in this field.

### Datasets
The datasets used in the study were divided into two main groups: synthetic and non-synthetic. The same preprocessing procedures were applied to both datasets:

1. **Non-Synthetic Dataset**: A dataset consisting of existing hate speech texts.
2. **Synthetic Dataset**: Synthetic data generation, using the Llama 3.1 7D language model, was structured to add new texts to the existing dataset and enrich the dataset. This process was structured to help balance the unbalanced classes.
   - To overcome the problems caused by imbalanced data classes, the SMOTE (Synthetic Minority Oversampling Technique) method was applied. The same data processing and modeling procedures were applied to both datasets.

### Data Preprocessing
- **Tokenization and Cleansing**: Data is split into words (tokenized) and cleaned of non-significant characters, stop words, and special characters.
- **Zemberek Spell Correction**: Zemberek library was used to correct grammatical errors in Turkish texts.
- **Zemberek Normalize**: Normalization operations are performed with the Zemberek library for consistent modeling of texts.

### Feature Extraction
Texts were converted into numerical data using different vectorization methods:

- **TurkishWord2Vec**: A pre-trained word embedding model specific to the Turkish language.
- **Fine-Tune Word2Vec**: Study-specific retrained Word2Vec model.
- **N-gram Based Models**:
  - **Word-gram**: Word-based n-gram representations.
  - **Char-gram**: Character-based n-gram features.
  - **TF-IDF and CountVectorizer**: N-gram methods are used with two vectorization methods. These provide frequency-based representations of texts to increase model performance.
- **Combined Methods**: Combination of word and character-based features (e.g., Word Unigram + Word Bigram + Char Trigrams).

### Model Training
The following models were trained to detect hate speech:

- **CatBoost and XGBoost**: Gradient boosting algorithms.
- **MLPC-SGD**: The multilayer perceptron (MLP) model optimized by stochastic gradient descent method.
- **ExtraTreesClassifier**: Tree-based ensemble learning model.
- **ANN**: A customized artificial neural network architecture.

### Evaluation Metrics
The following metrics were used to measure model performances:

- Accuracy
- Precision
- Recall
- F1 Score

---

## Results

### Results and Analysis

#### Non-Synthetic Dataset Results:
| Model         | Vectorization Method | Accuracy  | Precision | Recall   | F1 Score |
|---------------|----------------------|-----------|-----------|----------|----------|
| CatBoostC     | TurkishWord2Vec      | 0.629794  | 0.716249  | 0.629794 | 0.66494  |
| XGBoost       | TurkishWord2Vec      | 0.766962  | 0.749744  | 0.766962 | 0.757297 |
| MLPC-sgd      | TurkishWord2Vec      | 0.725664  | 0.767823  | 0.725664 | 0.74114  |
| ExtraTreesC   | TurkishWord2Vec      | 0.770403  | 0.730594  | 0.770403 | 0.735169 |
| ANN           | TurkishWord2Vec      | 0.161259  | 0.771834  | 0.161259 | 0.249338 |
| CatBoostC     | FineTuneWord2Vec     | 0.538348  | 0.672612  | 0.538348 | 0.590817 |
| XGBoost       | FineTuneWord2Vec     | 0.721239  | 0.700948  | 0.721239 | 0.7093   |
| MLPC-sgd      | FineTuneWord2Vec     | 0.301868  | 0.650208  | 0.301868 | 0.379265 |
| ExtraTreesC   | FineTuneWord2Vec     | 0.752212  | 0.698773  | 0.752212 | 0.702753 |
| ANN           | FineTuneWord2Vec     | 0.23648   | 0.700203  | 0.23648  | 0.098889 |

#### Synthetic Dataset Results:
| Model         | Vectorization Method | Accuracy  | Precision | Recall   | F1 Score |
|---------------|----------------------|-----------|-----------|----------|----------|
| CatBoostC     | TurkishWord2Vec      | 0.523996  | 0.556403  | 0.523996 | 0.531438 |
| XGBoost       | TurkishWord2Vec      | 0.782372  | 0.782442  | 0.782372 | 0.782356 |
| MLPC-sgd      | TurkishWord2Vec      | 0.591307  | 0.607418  | 0.591307 | 0.59582  |
| ExtraTreesC   | TurkishWord2Vec      | 0.805312  | 0.817553  | 0.805312 | 0.800627 |
| ANN           | TurkishWord2Vec      | 0.262904  | 0.581094  | 0.262904 | 0.215534 |
| CatBoostC     | FineTuneWord2Vec     | 0.509206  | 0.521106  | 0.509206 | 0.512013 |
| XGBoost       | FineTuneWord2Vec     | 0.740417  | 0.740764  | 0.740417 | 0.740551 |
| MLPC-sgd      | FineTuneWord2Vec     | 0.367039  | 0.44585   | 0.367039 | 0.370599 |
| ExtraTreesC   | FineTuneWord2Vec     | 0.792937  | 0.801651  | 0.792937 | 0.789837 |
| ANN           | FineTuneWord2Vec     | 0.17205   | 0.658466  | 0.17205  | 0.061152 |

### Overall Performance Comparison

| Vectorization Method           | F1 Score (Synthetic) | F1 Score (Non-Synthetic) |
|--------------------------------|----------------------|--------------------------|
| Char Unigram + Char Trigram    | 82.95%              | 82.42%                   |
| WUniG + WBiG + ChBiG + ChTriG  | 74.00%              | 70.93%                   |
| TurkishWord2Vec                | 80.00%              | 75.72%                   |
| Fine-Tune Word2Vec             | 78.98%              | 70.27%                   |

The results show that character-based methods (e.g., CharTrigram) generally provide the highest performance, while methods such as Fine-Tune Word2Vec exhibit lower performance.

### Model Performances

| Model         | Vectorization Method      | Accuracy | F1 Score |
|---------------|---------------------------|----------|----------|
| XGBoost       | CharTrigram               | 83.87%   | 81.67%   |
| ExtraTrees    | ChUnigram + ChTrigram     | 84.66%   | 84.32%   |
| MLPC-SGD      | ChBigram + ChTrigram      | 82.97%   | 82.82%   |
| CatBoostC     | CharTrigram               | 74.14%   | 72.91%   |
| ANN           | CharTrigram               | 78.36%   | 79.59%   |

### The Impact of Synthetic Data

Using synthetic data improved performance in models trained on smaller datasets. However, in general, models trained on non-synthetic data achieved higher accuracy and F1 scores. For example:

- The success rate difference between non-synthetic and synthetic data ranged between 3% and 5%, favoring non-synthetic data.
- The most successful vectorization group was Char Unigram + Char Trigram, with an average performance difference of 4% - 5%.

#### Non-Synthetic
![7adca585-60e8-4ebf-acad-d35ff7d77658](https://github.com/user-attachments/assets/26371cbe-4400-42f4-a337-b1cfcc2ca127)

#### Synthetic
![50673389-2980-4e6c-a759-91d9cb80264b](https://github.com/user-attachments/assets/d4951c27-43c1-4cc8-8d87-2c0074085e2a)

---

## Future Work
- Explore advanced language models (e.g., GPT series) for synthetic data generation.
- Test generalizability across datasets from other social media platforms (e.g., Facebook, Instagram).

---

## Installation

[GitHub: Turkish Word2Vec](https://github.com/akoksal/Turkish-Word2Vec)

[Article: Article]((https://dergipark.org.tr/tr/download/article-file/1664504))
[GitHub: DataSet](https://github.com/imayda/turkish-hate-speech-dataset-2)

[Download: TurkishWord2Vec](https://drive.google.com/drive/folders/1IBMTAGtZ4DakSCyAoA4j7Ch0Ft1aFoww)


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or contributions, please reach out to [your-email@example.com](mailto:your-email@example.com).
