import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch

# CONFIG
DATASET_PATH = 'En-Ba-Dataset(20k_4)/dataset.csv'
EMBEDDING_DIM = 100
OUTPUT_PATH = 'embedded_data/dataset_numeric_final.csv'
VOCAB_PATH = 'vocabulary.txt'
MODEL_NAME = 'bert-base-multilingual-cased'  # multilingual BERT

def preprocess(text):
    """Preprocess text by lowercasing and removing special characters"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def build_vocab(sentences):
    """Build vocabulary from all words"""
    vocab = Counter()
    for sent in sentences:
        tokens = preprocess(sent).split()
        vocab.update(tokens)
    return {word: {"index": idx, "count": count} 
            for idx, (word, count) in enumerate(vocab.most_common())}

def save_vocab(vocab, filepath):
    """Save vocabulary to file as a dictionary format"""
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]['count'], reverse=True))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)
    print(f'\nTop 20 most common words:')
    for i, (word, info) in enumerate(sorted_vocab.items()):
        if i >= 20: break
        print(f'{word}: {info["count"]}')

def normalize_vector(vec):
    """Convert vector to positive integers between 1 and 1000"""
    # Take absolute values
    vec = np.abs(vec)
    # Normalize to range [1, 1000]
    vec_min, vec_max = vec.min(), vec.max()
    if vec_max == vec_min:
        return np.ones_like(vec, dtype=int)
    normalized = 1 + ((vec - vec_min) * 999 / (vec_max - vec_min))
    return np.round(normalized).astype(int)

def get_bert_embedding(text, tokenizer, model):
    """Get BERT embedding for a text"""
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get BERT embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        
    # Convert to numpy and reduce dimensions if needed
    embedding = embeddings.numpy()[0]
    if len(embedding) > EMBEDDING_DIM:
        # Use PCA or simply take first EMBEDDING_DIM components
        embedding = embedding[:EMBEDDING_DIM]
    
    return embedding

def embed_sentence(text, tokenizer, model):
    """Convert a sentence to integer vector using BERT"""
    # Get BERT embedding
    vec = get_bert_embedding(preprocess(text), tokenizer, model)
    # Convert to positive integers
    return normalize_vector(vec)

def main():
    print('Loading BERT model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print('Loading dataset...')
    df = pd.read_csv(DATASET_PATH)
    sentences = df['Sentence'].astype(str).tolist()
    labels = df['Label'].tolist()

    print('Building vocabulary...')
    vocab = build_vocab(sentences)
    print(f'Total vocabulary size: {len(vocab)}')
    save_vocab(vocab, VOCAB_PATH)
    
    print('\nEmbedding sentences...')
    embedded = []
    for sent in tqdm(sentences, desc='Embedding'):
        vec = embed_sentence(sent, tokenizer, model)
        embedded.append(vec)

    print('\nSaving embeddings...')
    embedded_arr = np.vstack(embedded)
    print(f'Embedding shape: {embedded_arr.shape}')
    print(f'Value range: [{embedded_arr.min()}, {embedded_arr.max()}]')
    
    out_df = pd.DataFrame(embedded_arr)
    out_df['Label'] = labels
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved embedded data to {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
