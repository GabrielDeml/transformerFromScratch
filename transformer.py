# The only import we need is numpy everything else will be written from scratch
import numpy as np
import os
import requests


# TODO: Dataset class
# TODO: Implement Input Embedding
# TODO: Implement Positional Encoding
# TODO: Implement Scaled Dot-Product Attention
# TODO: Implement Multi-Head Attention
# TODO: Implement Layer Normalization
# TODO: Implement Position-wise Feed-Forward Networks
# TODO: Implement the Encoder block
# TODO: Implement the Decoder block (including masked attention and encoder-decoder attention)
# TODO: Implement the final Linear layer and Softmax function
# TODO: Combine everything into the Transformer model class


class Dataset:
    def __init__(self, data_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_path="input.txt"):
        """
        Initializes the Dataset class. Downloads the Tiny Shakespeare dataset if not present,
        reads the text, creates the vocabulary, and encodes the text.
        """
        self.data_path = data_path
        self._download_data(data_url)
        self._load_data()
        self._build_vocab()
        self._encode_data()

    def _download_data(self, url):
        """Downloads the dataset if it doesn't exist locally."""
        if not os.path.exists(self.data_path):
            print(f"Downloading dataset from {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(self.data_path, 'w') as f:
                f.write(response.text)
            print(f"Dataset saved to {self.data_path}")

    def _load_data(self):
        """Loads the text data from the file."""
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        print(f"Data loaded. Length of dataset in characters: {len(self.text)}")

    def _build_vocab(self):
        """Builds the character vocabulary and mappings."""
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print(f"Vocabulary built. Size: {self.vocab_size}")
        print(f"Vocabulary: {''.join(self.chars)}")

        # Create mappings from characters to integers and vice versa
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def _encode_data(self):
        """Encodes the entire text dataset into integers."""
        self.data = np.array([self.stoi[ch] for ch in self.text], dtype=np.int64)
        print("Data encoded.")

    def get_batch(self, batch_size, block_size):
        """
        Generates a batch of data (inputs x and targets y).
        Args:
            batch_size (int): Number of independent sequences in a batch.
            block_size (int): Maximum context length for predictions.
        Returns:
            tuple: A tuple containing input batch (x) and target batch (y), both numpy arrays.
        """
        # Generate random starting points for sequences in the batch
        ix = np.random.randint(0, len(self.data) - block_size, batch_size)

        # Extract input sequences (x) and target sequences (y)
        # y is the sequence shifted by one position to the right
        x = np.stack([self.data[i:i + block_size] for i in ix])
        y = np.stack([self.data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    def encode(self, s):
        """Encodes a string into a list of integers."""
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """Decodes a list of integers back into a string."""
        return ''.join([self.itos[i] for i in l])
    

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len

    def __call__(self, x):
        pe = np.zeros((self.max_len, self.d_model))
        for pos in range(self.max_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pe[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                else:
                    pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) / self.d_model)))
        return pe

class ScaledDotProductAttention:
    def __init__(self, d_model):
        self.d_model = d_model

    def __call__(self, q, k, v):
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        scores = np.softmax(scores, axis=-1)
        out = np.matmul(scores, v)
        return out

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = np.random.randn(d_model, d_ff)
        self.b1 = np.random.randn(d_ff)
        self.w2 = np.random.randn(d_ff, d_model)
        self.b2 = np.random.randn(d_model)

    def __call__(self, x):
        x = x @ self.w1 + self.b1
        x = x @ self.w2 + self.b2
        return x

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = Transformer.FeedForward(d_model, d_model)
        self.k_proj = Transformer.FeedForward(d_model, d_model)
        self.v_proj = Transformer.FeedForward(d_model, d_model)
        self.out_proj = Transformer.FeedForward(d_model, d_model)

    def __call__(self, q, k, v):
        batch_size = q.shape[0]
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = q.reshape(batch_size, -1, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        attention = Transformer.ScaledDotProductAttention(self.d_model)(q, k, v)
        attention = attention.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        return self.out_proj(attention)

    def __init__(self, vocab_size, block_size, n_heads, n_blocks, dropout):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        # ... initialize layers as needed ...

    def forward(self, x):
        """
        Forward pass of the Transformer model.
        """
        x = Transformer.PositionalEncoding(self.d_model)(x)
        for _ in range(self.n_blocks):
            x = Transformer.EncoderBlock(self.d_model, self.n_heads)(x)
        return x

class LayerNorm:
    def __init__(self, d_model):
        self.d_model = d_model
        self.scale = np.ones(d_model)
        self.shift = np.zeros(d_model)

    def __call__(self, x):
        return self.scale * (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5) + self.shift

class EncoderBlock:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model)

    def __call__(self, x):
        x = self.ln1(x)
        x = self.mha(x, x, x)
        x = self.ln2(x)
        x = self.ff(x)
        return x
    
class DecoderBlock:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model)

    def __call__(self, x):
        x = self.ln1(x)
        x = self.mha(x, x, x)
        x = self.ln2(x)
        x = self.ff(x)
        return x
    
class Transformer:
    def __init__(self, vocab_size, block_size, n_heads, n_blocks, dropout):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout

    def forward(self, x):
        x = Transformer.PositionalEncoding(self.d_model)(x)
        for _ in range(self.n_blocks):
            x = Transformer.EncoderBlock(self.d_model, self.n_heads)(x)
        return x
    
    def backward(self, x, y):
        logits = self.forward(x)
        loss = np.mean(np.log(logits[np.arange(len(y)), y]))
        return loss
    
    def update(self):
        """
        Updates the model parameters using gradient descent.
        """
        self.w1 -= self.lr * self.w1_grad
        self.b1 -= self.lr * self.b1_grad
        self.w2 -= self.lr * self.w2_grad
        self.b2 -= self.lr * self.b2_grad
    
    def loss(self, x, y):
        logits = self.forward(x)
        loss = np.mean(np.log(logits[np.arange(len(y)), y]))
        return loss
    
    def generate(self, context, max_new_tokens):
        x = np.array(context)
        for _ in range(max_new_tokens):
            logits = self.forward(x)
            next_char = np.argmax(logits[-1])
            x = np.concatenate([x, [next_char]], axis=0)
        return x
    



if __name__ == '__main__':
    dataset = Dataset()
    transformer = Transformer(dataset.vocab_size, dataset.block_size, 4, 3, 0.1)
    xb, yb = dataset.get_batch(4, 8)
    print(transformer.forward(xb))
    print(transformer.loss(xb, yb))
    print(transformer.generate("hello", 10))
    print(transformer.forward(xb))
    print(transformer.loss(xb, yb))
    print(transformer.generate("hello", 10))
    print(transformer.forward(xb))
    print(transformer.loss(xb, yb))
    print(transformer.generate("hello", 10))
    
    # Training loop
    for i in range(1000):
        xb, yb = dataset.get_batch(4, 8)
        loss = transformer.loss(xb, yb)
        print(loss)
        transformer.backward(xb, yb)
        transformer.update()

        if i % 100 == 0:
            print(transformer.generate("hello", 10))
            print("-"*100)

    print(transformer.generate("hello", 100))

