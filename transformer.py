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

# Example usage (optional, can be removed later)
if __name__ == '__main__':
    dataset = Dataset()
    print(f"First 100 characters encoded: {dataset.data[:100]}")
    print(f"Decoded: {dataset.decode(dataset.data[:100].tolist())}")

    batch_size = 4
    block_size = 8
    xb, yb = dataset.get_batch(batch_size, block_size)
    print("\n--- Example Batch ---")
    print("Inputs (xb):")
    print(xb)
    print("Targets (yb):")
    print(yb)

    for b in range(batch_size): # iterate through batch dimension
        print(f"\nBatch item {b+1}:")
        context = xb[b]
        target = yb[b]
        print(f"Input context: {dataset.decode(context.tolist())}")
        print(f"Target chars:  {dataset.decode(target.tolist())}")


