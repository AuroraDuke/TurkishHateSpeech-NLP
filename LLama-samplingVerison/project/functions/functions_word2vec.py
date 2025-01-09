import os
import numpy as np
from gensim.models import KeyedVectors

def calculate_word2vec(doc, word_vectors):
    """Belirtilen dokümanın ortalama Word2Vec vektörünü hesaplar."""
    vectors = [word_vectors[word] for word in doc.split() if word in word_vectors]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)

def process_sentence(sentence, word_vectors):
    """Bir cümleyi alır ve Word2Vec vektörüne dönüştürür."""
    # Cümledeki kelimeleri tokenize et
    words = sentence.split()
    # Word2Vec modeli içinde olmayan kelimeleri filtrele
    filtered_words = [word for word in words if word in word_vectors]
    # Ortalama vektörü hesapla
    return calculate_word2vec(" ".join(filtered_words), word_vectors)

def transform_word2vec(X, model_path):
    """
    Metin verisini Word2Vec vektörlerine dönüştürür.

    Args:
        X (list or pd.Series): Metin verisi.
        model_path (str): Türkçe Word2Vec model dosyasının yolu.

    Returns:
        np.array: Word2Vec vektörleri.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

    return np.array([process_sentence(text, word_vectors) for text in X])

