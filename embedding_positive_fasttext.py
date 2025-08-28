import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from tqdm import tqdm
import json

# CONFIG
DATASET_PATH = 'En-Ba-Dataset(20k_4)/dataset.csv'
EMBEDDING_DIM = 100
OUTPUT_PATH = 'embedded_data/dataset_numeric_final.csv'
VOCAB_PATH = 'vocabulary.txt'

class FastTextEmbedding:
    def __init__(self, dim=100, window=5, min_count=1, epochs=5):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.word_vectors = {}
        
    def _create_ngrams(self, word, n_min=3, n_max=6):
        """Create character n-grams from a word"""
        chars = ' ' + word + ' '
        ngrams = []
        for n in range(n_min, min(n_max + 1, len(chars))):
            for i in range(len(chars) - n + 1):
                ngrams.append(chars[i:i+n])
        return ngrams

    def _initialize_vectors(self, vocab):
        """Initialize vectors for words and n-grams"""
        for word in vocab:
            # Initialize word vector (1-10 range)
            self.word_vectors[word] = np.random.randint(1, 11, self.dim)
            # Initialize n-gram vectors (1-10 range)
            for ngram in self._create_ngrams(word):
                if ngram not in self.word_vectors:
                    self.word_vectors[ngram] = np.random.randint(1, 11, self.dim)

    def _normalize_vector(self, vec):
        """Normalize vector to positive integers in range 1-1000"""
        # Ensure no negative values
        vec = np.abs(vec)
        # Avoid division by zero
        if vec.max() == vec.min():
            return np.ones_like(vec, dtype=int)
        # Normalize to 1-1000 range
        normalized = 1 + ((vec - vec.min()) / (vec.max() - vec.min()) * 999)
        return np.round(normalized).astype(int)

    def train(self, sentences):
        """Train FastText model on sentences"""
        # Build vocabulary
        vocab = set()
        for sent in sentences:
            vocab.update(sent)

        # Initialize vectors
        print("Initializing vectors...")
        self._initialize_vectors(vocab)

        # Training
        print("Training FastText model...")
        learning_rate = 0.01
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for sent in tqdm(sentences):
                for i, target in enumerate(sent):
                    # Get context words
                    start = max(0, i - self.window)
                    end = min(len(sent), i + self.window + 1)
                    context = sent[start:i] + sent[i+1:end]
                    
                    if target in self.word_vectors:
                        target_ngrams = self._create_ngrams(target)
                        target_vec = self.word_vectors[target].copy()
                        
                        # Add n-gram vectors (with normalization)
                        for ngram in target_ngrams:
                            if ngram in self.word_vectors:
                                target_vec = np.maximum(1, target_vec + self.word_vectors[ngram])
                        
                        # Update based on context
                        for ctx_word in context:
                            if ctx_word in self.word_vectors:
                                ctx_vec = self.word_vectors[ctx_word].copy()
                                # Simple update rule with normalization
                                gradient = np.clip(target_vec - ctx_vec, -1, 1) * learning_rate
                                
                                # Update vectors ensuring they stay positive
                                self.word_vectors[target] = self._normalize_vector(
                                    self.word_vectors[target] - gradient)
                                self.word_vectors[ctx_word] = self._normalize_vector(
                                    self.word_vectors[ctx_word] + gradient)

        # Final normalization of all vectors
        for word in self.word_vectors:
            self.word_vectors[word] = self._normalize_vector(self.word_vectors[word])

    def get_vector(self, word):
        """Get vector for a word, including n-gram information"""
        if word in self.word_vectors:
            vec = self.word_vectors[word].copy()
            n_contributors = 1
            
            # Add n-gram vectors
            for ngram in self._create_ngrams(word):
                if ngram in self.word_vectors:
                    vec += self.word_vectors[ngram]
                    n_contributors += 1
            
            # Average and normalize
            vec = vec / n_contributors
            return self._normalize_vector(vec)
        
        return np.ones(self.dim, dtype=int)  # Default vector for unknown words

def preprocess(text):
    """Preprocess text by lowercasing and removing special characters"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

def build_vocab(sentences):
    """Build vocabulary from all words"""
    vocab = Counter()
    for sent in sentences:
        tokens = preprocess(sent)
        vocab.update(tokens)
    return {word: {"index": idx, "count": count} 
            for idx, (word, count) in enumerate(vocab.most_common())}

def save_vocab(vocab, filepath):
    """Save vocabulary to file as a dictionary format"""
    # Sort by count in descending order
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]['count'], reverse=True))
    
    # Save as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)
    
    # Print top 20 words
    print(f'\nTop 20 most common words:')
    for i, (word, info) in enumerate(sorted_vocab.items()):
        if i >= 20:
            break
        print(f'{word}: {info["count"]}')

def embed_sentence(tokens, model):
    """Convert a sentence to a vector using word embeddings"""
    vectors = []
    for token in tokens:
        vec = model.get_vector(token)
        vectors.append(vec)
    
    # Average the vectors (all positive at this point)
    sent_vec = np.mean(vectors, axis=0) if vectors else np.ones(model.dim, dtype=int)
    return np.round(sent_vec).astype(int)

def verify_no_negatives(array):
    """Verify there are no negative numbers in the array"""
    min_val = array.min()
    max_val = array.max()
    print(f"\nVerifying values:")
    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")
    assert min_val >= 0, "Found negative values!"
    return min_val >= 0

def main():
    print('Loading dataset...')
    df = pd.read_csv(DATASET_PATH)
    sentences = df['Sentence'].astype(str).tolist()
    labels = df['Label'].tolist()

    print('Building vocabulary...')
    vocab = build_vocab(sentences)
    print(f'Total vocabulary size: {len(vocab)}')
    
    # Save vocabulary
    save_vocab(vocab, VOCAB_PATH)
    print(f'Vocabulary saved to {VOCAB_PATH}')

    print('\nPreprocessing sentences...')
    processed_sentences = [preprocess(sent) for sent in sentences]

    print('\nTraining FastText model...')
    model = FastTextEmbedding(dim=EMBEDDING_DIM)
    model.train(processed_sentences)

    print('\nEmbedding sentences...')
    embedded = []
    for sent in tqdm(sentences, desc='Embedding'):
        tokens = preprocess(sent)
        vec = embed_sentence(tokens, model)
        embedded.append(vec)

    print('\nSaving embeddings...')
    embedded_arr = np.vstack(embedded)
    
    # Verify no negative numbers
    if verify_no_negatives(embedded_arr):
        print("All values are non-negative ✓")
    
    out_df = pd.DataFrame(embedded_arr)
    out_df['Label'] = labels
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved embedded data to {OUTPUT_PATH}')
    print(f'Embedding shape: {embedded_arr.shape}')

if __name__ == '__main__':
    main()
