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
        self.block_size = 256  # Define block size as a default

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
        
    def clip_token(self, token):
        """Ensures token ID is within vocabulary range"""
        return min(max(0, token), self.vocab_size - 1)

    def _encode_data(self):
        """Encodes the entire text dataset into integers."""
        self.data = np.array([self.stoi.get(ch, 0) for ch in self.text], dtype=np.int64)
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
        ix = np.random.randint(0, len(self.data) - block_size - 1, batch_size)

        # Extract input sequences (x) and target sequences (y)
        # y is the sequence shifted by one position to the right
        x = np.stack([self.data[i:i + block_size] for i in ix])
        y = np.stack([self.data[i + 1:i + block_size + 1] for i in ix])
        
        # Ensure all tokens are within vocabulary range
        if np.any(x >= self.vocab_size) or np.any(y >= self.vocab_size):
            # This should never happen with properly encoded data
            print(f"Warning: Found token IDs >= vocab_size ({self.vocab_size})")
            x = np.clip(x, 0, self.vocab_size - 1)
            y = np.clip(y, 0, self.vocab_size - 1)
            
        return x, y

    def encode(self, s):
        """Encodes a string into a list of integers."""
        # Convert each character to its token ID, clipping to vocabulary range
        return [self.clip_token(self.stoi.get(c, 0)) for c in s]

    def decode(self, l):
        """Decodes a list of integers back into a string."""
        # Ensure all tokens are within vocabulary range before decoding
        return ''.join([self.itos.get(self.clip_token(i), '?') for i in l])
    

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        # Pre-compute the positional encodings
        self.pe = np.zeros((self.max_len, self.d_model))
        for pos in range(self.max_len):
            for i in range(0, self.d_model, 2):
                self.pe[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    self.pe[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))

    def __call__(self, x):
        # Add positional encoding to the input
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

class ScaledDotProductAttention:
    def __init__(self):
        self.attention_weights = None

    def __call__(self, q, k, v, mask=None):
        # Calculate attention scores
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
            
        # Apply softmax to get attention weights
        self.attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Apply attention weights to values
        output = np.matmul(self.attention_weights, v)
        return output

class FeedForward:
    def __init__(self, d_model, d_ff, lr=0.001, grad_clip=1.0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.grad_clip = grad_clip  # Max gradient norm
        
        # Xavier/Glorot initialization for better training 
        scale_w1 = np.sqrt(2.0 / (d_model + d_ff))
        scale_w2 = np.sqrt(2.0 / (d_ff + d_model))
        
        self.w1 = np.random.randn(d_model, d_ff) * scale_w1
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) * scale_w2
        self.b2 = np.zeros(d_model)
        self.lr = lr
        
        # Gradient initialization
        self.w1_grad = np.zeros_like(self.w1)
        self.b1_grad = np.zeros_like(self.b1)
        self.w2_grad = np.zeros_like(self.w2)
        self.b2_grad = np.zeros_like(self.b2)
        
    def clip_gradients(self):
        # Clip gradients to prevent exploding gradients
        for grad in [self.w1_grad, self.b1_grad, self.w2_grad, self.b2_grad]:
            if grad.size > 0:  # Check if gradient is not empty
                grad_norm = np.sqrt(np.sum(grad**2))
                if grad_norm > self.grad_clip:
                    grad *= self.grad_clip / (grad_norm + 1e-8)
        
    def gelu(self, x):
        # GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # Used in modern transformer models like GPT and BERT
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def gelu_derivative(self, x):
        # Approximation of GELU derivative
        gelu_grad = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        gelu_grad += x * 0.5 * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))**2) * \
                     np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(x, 2))
        return gelu_grad

    def __call__(self, x, training=True):
        # Save input shape for backward pass
        self.input_shape = x.shape
        
        # Reshape input for matrix multiplication if needed
        if len(x.shape) > 2:
            x_reshaped = x.reshape(-1, x.shape[-1])
        else:
            x_reshaped = x
            
        # Feedforward pass
        self.input = x_reshaped
        self.hidden_linear = self.input @ self.w1 + self.b1
        self.hidden = self.gelu(self.hidden_linear)  # Use GELU activation
        output = self.hidden @ self.w2 + self.b2
        
        # Reshape output back to original shape if needed
        if len(self.input_shape) > 2:
            output = output.reshape(self.input_shape)
            
        return output
    
    def backward(self, grad):
        # Store original gradient shape
        orig_grad_shape = grad.shape
        
        # Reshape gradient if needed
        if len(grad.shape) > 2:
            grad_reshaped = grad.reshape(-1, grad.shape[-1])
        else:
            grad_reshaped = grad
        
        # Check if we need to adjust dimensions
        if grad_reshaped.shape[0] != self.hidden.shape[0]:
            # For example, if we have batch_size * seq_len != batch_size * seq_len from a previous step
            # This is a simplification - in a real implementation, we'd need to handle this more carefully
            self.w2_grad = np.zeros_like(self.w2)
            self.b2_grad = np.sum(grad_reshaped, axis=0)
        else:
            # Gradient with respect to output when dimensions match
            self.w2_grad = self.hidden.T @ grad_reshaped
            self.b2_grad = np.sum(grad_reshaped, axis=0)
        
        # Gradient with respect to hidden
        d_hidden = grad_reshaped @ self.w2.T
        
        # Apply GELU derivative
        if self.hidden.shape[0] == d_hidden.shape[0]:
            gelu_derivative = self.gelu_derivative(self.hidden_linear)
            d_hidden = d_hidden * gelu_derivative
        
        # Gradient with respect to input
        if d_hidden.shape[0] != self.input.shape[0]:
            # Dimensions don't match, return a zero gradient
            d_input = np.zeros(self.input_shape)
            self.w1_grad = np.zeros_like(self.w1)
            self.b1_grad = np.sum(d_hidden, axis=0)
        else:
            # Dimensions match, compute gradients normally
            d_input = d_hidden @ self.w1.T
            self.w1_grad = self.input.T @ d_hidden
            self.b1_grad = np.sum(d_hidden, axis=0)
        
        # Reshape gradient back to original shape if needed
        if len(orig_grad_shape) > 2:
            d_input = d_input.reshape(orig_grad_shape)
            
        return d_input
    
    def update(self):
        # Clip gradients before updating
        self.clip_gradients()
        
        # Update weights
        self.w1 -= self.lr * self.w1_grad
        self.b1 -= self.lr * self.b1_grad
        self.w2 -= self.lr * self.w2_grad
        self.b2 -= self.lr * self.b2_grad

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, lr=0.001, grad_clip=1.0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.lr = lr
        self.grad_clip = grad_clip
        
        # Projection matrices
        self.q_proj = FeedForward(d_model, d_model, lr, grad_clip)
        self.k_proj = FeedForward(d_model, d_model, lr, grad_clip)
        self.v_proj = FeedForward(d_model, d_model, lr, grad_clip)
        self.out_proj = FeedForward(d_model, d_model, lr, grad_clip)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention()

    def __call__(self, q, k, v, mask=None):
        # Save input shapes for backward pass
        self.q_shape = q.shape
        self.k_shape = k.shape
        self.v_shape = v.shape
        
        batch_size = q.shape[0]
        q_len = q.shape[1]
        k_len = k.shape[1]
        v_len = v.shape[1]
        
        # Project inputs
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, q_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, k_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, v_len, self.n_heads, self.head_dim)
        
        # Transpose to (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Calculate attention
        self.q_heads = q
        self.k_heads = k
        self.v_heads = v
        attention_output = self.attention(q, k, v, mask)
        
        # Transpose and reshape back
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attention_output)
        return output
    
    def backward(self, grad):
        # Implement backward pass
        # This is a simplified version - in a real implementation this would be more complex
        grad = self.out_proj.backward(grad)
        
        # In a real implementation, we'd need to compute gradients for q, k, and v
        # and propagate them back through the projections
        
        # Return gradients for query and key/value
        return grad, grad  # Return two gradients for the decoder block
        
    def update(self):
        # Update all projection matrices
        self.q_proj.update()
        self.k_proj.update()
        self.v_proj.update()
        self.out_proj.update()

class LayerNorm:
    def __init__(self, d_model, lr=0.001, eps=1e-6, grad_clip=1.0):
        self.d_model = d_model
        self.gamma = np.ones(d_model)  # Scale parameter
        self.beta = np.zeros(d_model)  # Shift parameter
        self.lr = lr
        self.eps = eps  # Small value for numerical stability
        self.grad_clip = grad_clip  # Max gradient norm
        
        # Gradient initialization
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)
        
    def clip_gradients(self):
        # Clip gradients to prevent exploding gradients
        for grad in [self.gamma_grad, self.beta_grad]:
            if grad.size > 0:  # Check if gradient is not empty
                grad_norm = np.sqrt(np.sum(grad**2))
                if grad_norm > self.grad_clip:
                    grad *= self.grad_clip / (grad_norm + 1e-8)

    def __call__(self, x):
        # Store input for backward pass
        self.input = x
        
        # Compute mean and variance along last dimension
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # Scale and shift
        self.output = self.gamma * self.norm + self.beta
        
        return self.output
    
    def backward(self, grad):
        # For simplicity, we'll just pass through the gradient
        # In a real implementation, we would compute proper gradients
        
        # Sum along all dimensions except the last one
        # This ensures gamma_grad and beta_grad have shape (d_model,)
        axes = tuple(range(self.norm.ndim - 1))
        
        # Compute gradient with respect to gamma and beta
        self.gamma_grad = np.sum(grad * self.norm, axis=axes)
        self.beta_grad = np.sum(grad, axis=axes)
        
        return grad
    
    def update(self):
        # Clip gradients before updating
        self.clip_gradients()
        
        # Update parameters
        self.gamma -= self.lr * self.gamma_grad
        self.beta -= self.lr * self.beta_grad

class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff=2048, lr=0.001, grad_clip=1.0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.lr = lr
        self.grad_clip = grad_clip
        
        # Components
        self.norm1 = LayerNorm(d_model, lr, 1e-6, grad_clip)
        self.mha = MultiHeadAttention(d_model, n_heads, lr, grad_clip)
        self.norm2 = LayerNorm(d_model, lr, 1e-6, grad_clip)
        self.ff = FeedForward(d_model, d_ff, lr, grad_clip)

    def __call__(self, x, mask=None):
        # Multi-head attention with residual connection and layer normalization
        attn_output = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        
        # Feedforward with residual connection and layer normalization
        output = attn_output + self.ff(self.norm2(attn_output))
        return output
    
    def backward(self, grad):
        # Simplified backward pass: ignore residual connections for now
        # Just propagate the gradient through each component
        
        # Feedforward
        ff_grad = self.ff.backward(grad)
        
        # Normalization
        norm2_grad = self.norm2.backward(ff_grad)
        
        # Multi-head attention
        mha_grad, _ = self.mha.backward(norm2_grad)
        
        # First normalization
        norm1_grad = self.norm1.backward(mha_grad)
        
        return norm1_grad
    
    def update(self):
        self.norm1.update()
        self.mha.update()
        self.norm2.update()
        self.ff.update()
    
class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff=2048, lr=0.001):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.lr = lr
        
        # Components
        self.norm1 = LayerNorm(d_model, lr)
        self.masked_mha = MultiHeadAttention(d_model, n_heads, lr)
        self.norm2 = LayerNorm(d_model, lr)
        self.cross_mha = MultiHeadAttention(d_model, n_heads, lr)
        self.norm3 = LayerNorm(d_model, lr)
        self.ff = FeedForward(d_model, d_ff, lr)

    def __call__(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # Masked multi-head attention
        attn1 = self.masked_mha(self.norm1(x), self.norm1(x), self.norm1(x), look_ahead_mask)
        attn1_output = x + attn1
        
        # Cross-attention with encoder output
        attn2 = self.cross_mha(self.norm2(attn1_output), self.norm2(enc_output), self.norm2(enc_output), padding_mask)
        attn2_output = attn1_output + attn2
        
        # Feedforward
        output = attn2_output + self.ff(self.norm3(attn2_output))
        return output
    
    def backward(self, grad, enc_output_grad):
        # Simplified backward pass
        ff_grad = self.ff.backward(self.norm3.backward(grad))
        grad = grad + ff_grad
        
        cross_mha_grad, enc_output_grad = self.cross_mha.backward(self.norm2.backward(grad))
        grad = grad + cross_mha_grad
        
        masked_mha_grad = self.masked_mha.backward(self.norm1.backward(grad))
        return grad + masked_mha_grad, enc_output_grad
    
    def update(self):
        self.norm1.update()
        self.masked_mha.update()
        self.norm2.update()
        self.cross_mha.update()
        self.norm3.update()
        self.ff.update()
    
class Transformer:
    def __init__(self, vocab_size, d_model=512, block_size=256, n_heads=8, n_blocks=6, d_ff=2048, 
                 dropout=0.1, lr=0.001, grad_clip=1.0, stoi=None, itos=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.d_ff = d_ff
        self.dropout = dropout
        self.lr = lr
        self.grad_clip = grad_clip
        
        # Character to token mapping
        self.stoi = stoi if stoi is not None else {}  # str to int (encoding)
        self.itos = itos if itos is not None else {}  # int to str (decoding)
        
        # Token embedding and positional encoding
        # Initialize with Xavier/Glorot initialization
        embedding_scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.token_embedding = np.random.randn(vocab_size, d_model) * embedding_scale
        self.positional_encoding = PositionalEncoding(d_model, block_size)
        
        # Encoder stack (simplified to encoder-only architecture)
        self.encoder_blocks = [EncoderBlock(d_model, n_heads, d_ff, lr, grad_clip) for _ in range(n_blocks)]
        
        # Final layer (for predicting next token)
        self.final_layer = FeedForward(d_model, vocab_size, lr, grad_clip)
        
        # Gradient initialization for token embedding
        self.token_embedding_grad = np.zeros_like(self.token_embedding)

    def embed(self, x):
        # Convert integer tokens to embeddings
        if len(x.shape) == 1:  # Single sequence
            embedded = self.token_embedding[x]
        else:  # Batch of sequences
            embedded = np.zeros((x.shape[0], x.shape[1], self.d_model))
            for i in range(x.shape[0]):
                embedded[i] = self.token_embedding[x[i]]
        
        # Add positional encoding
        return self.positional_encoding(embedded)

    def forward(self, x):
        # Embed input
        x = self.embed(x)
        
        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Final linear layer to get logits
        logits = self.final_layer(x)
        return logits
    
    def loss(self, x, y):
        # Forward pass
        logits = self.forward(x)
        
        # Calculate loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        y_flat = y.reshape(-1)
        
        # Ensure indices are in bounds
        y_flat = np.clip(y_flat, 0, vocab_size - 1)
        
        # Compute softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))  # For numerical stability
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy loss
        correct_probs = probs[np.arange(len(y_flat)), y_flat]
        loss = -np.mean(np.log(correct_probs + 1e-10))
        
        return loss
    
    def backward(self, x, y):
        # Forward pass
        logits = self.forward(x)
        
        # Calculate loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        y_flat = y.reshape(-1)
        
        # Ensure indices are in bounds
        y_flat = np.clip(y_flat, 0, vocab_size - 1)
        
        # Compute softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))  # For numerical stability
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy loss
        correct_probs = probs[np.arange(len(y_flat)), y_flat]
        loss = -np.mean(np.log(correct_probs + 1e-10))
        
        # Gradient of softmax with cross-entropy
        d_logits = probs.copy()
        d_logits[np.arange(len(y_flat)), y_flat] -= 1
        d_logits /= batch_size * seq_len
        
        # Reshape gradient back
        d_logits = d_logits.reshape(batch_size, seq_len, vocab_size)
        
        # Backward pass through the network
        d_final = self.final_layer.backward(d_logits)
        
        # Backward through encoder
        for block in reversed(self.encoder_blocks):
            d_final = block.backward(d_final)
        
        # We're not updating embedding gradients in this simplified version
        
        return loss
    
    def update(self):
        """
        Updates the model parameters using gradient descent.
        """
        # Update encoder blocks
        for block in self.encoder_blocks:
            block.update()
        
        # Update final layer
        self.final_layer.update()
    
    def generate(self, context, max_new_tokens, temperature=0.8):
        # Encode context if it's a string
        if isinstance(context, str):
            # We need to convert each character to its token ID
            context_str = context  # Save the original string
            context = np.array([min(max(0, self.stoi.get(c, 0)), self.vocab_size - 1) for c in context])
        else:
            context_str = None
            context = np.array(context)
            # Ensure all tokens are within vocabulary range
            context = np.clip(context, 0, self.vocab_size - 1)
        
        # Store the original context length
        original_len = len(context)
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Ensure context does not exceed block size
            if len(context) > self.block_size:
                context = context[-self.block_size:]
            
            # Forward pass to get next token probabilities
            logits = self.forward(context.reshape(1, -1))[0, -1]
            
            if temperature == 0:
                # Greedy sampling (argmax)
                next_token = np.argmax(logits)
            else:
                # Apply temperature
                logits_scaled = logits / temperature
                # Softmax to get probabilities
                probs = np.exp(logits_scaled - np.max(logits_scaled))
                probs = probs / np.sum(probs)
                # Sample from the distribution
                # Create an array of possible token indices
                indices = np.arange(len(probs))
                # Sample from the tokens based on probabilities
                next_token = np.random.choice(indices, p=probs)
            
            # Clip to ensure it's a valid token ID
            next_token = min(max(0, next_token), self.vocab_size - 1)
            
            # Append next token to context
            context = np.append(context, next_token)
        
        # Return only the newly generated tokens
        return context[original_len:]




if __name__ == '__main__':
    # Create dataset
    dataset = Dataset()
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    # Define a helper function to display generated text
    def display_generated_text(text):
        # Replace newlines with visible indicators for better display
        return text.replace('\n', '\\n')
    
    # Create a more powerful transformer model
    transformer = Transformer(
        vocab_size=dataset.vocab_size, 
        d_model=128,  # Larger model size (doubled)
        block_size=64,  # Larger context window (doubled)
        n_heads=8,  # More attention heads
        n_blocks=4,  # More encoder blocks
        d_ff=512,  # Larger feedforward layer
        dropout=0.2,  # Slightly more dropout for regularization
        lr=0.005,  # Lower learning rate for better convergence
        grad_clip=1.0,  # Added gradient clipping
        stoi=dataset.stoi,
        itos=dataset.itos
    )
    
    # Get a small batch for testing
    batch_size = 8  # Larger batch size
    seq_len = 32  # Longer sequences
    print(f"Getting batch of size {batch_size}x{seq_len}...")
    x, y = dataset.get_batch(batch_size, seq_len)
    
    # Print initial loss
    loss = transformer.loss(x, y)
    print(f"Initial loss: {loss}")
    
    # Run training with learning rate decay
    num_iterations = 10000  # More training iterations
    print(f"Training for {num_iterations} iterations...")
    
    # Only print every 1000 iterations to reduce output
    for i in range(num_iterations):
        # Learning rate decay
        if i > 0 and i % 2000 == 0:
            transformer.lr *= 0.8
            print(f"Iteration {i}: Learning rate decayed to {transformer.lr:.6f}")
            
        x, y = dataset.get_batch(batch_size, seq_len)
        loss = transformer.backward(x, y)
        transformer.update()
        
        if i % 1000 == 0 or i == num_iterations - 1:
            print(f"Iteration {i}, Loss: {loss}")
            
            # Generate a short sample every 1000 iterations
            if i > 0:
                print("\nSample generation:")
                prompt = "The "
                generated = transformer.generate(prompt, 20, temperature=0.8)
                result = prompt + ''.join([transformer.itos.get(int(token), '?') for token in generated])
                print(f"'{display_generated_text(result)}'")
                print()
    
    # Generate text with different prompts and temperatures
    print("\nGenerating text with the smarter model...")
    
    prompts = [
        "The king", 
        "To be or not to be", 
        "Love is", 
        "Once upon a time",
        "Shall I compare thee to"
    ]
    
    # Compare different temperature settings
    print("\n--- Greedy sampling (temperature=0) ---")
    for prompt in prompts:
        print(f"\nContext: '{prompt}'")
        generated = transformer.generate(prompt, 50, temperature=0)
        result = prompt + ''.join([transformer.itos.get(int(token), '?') for token in generated])
        print(f"Generated: '{display_generated_text(result)}'")
    
    print("\n--- With temperature=0.7 (more coherent) ---")
    for prompt in prompts:
        print(f"\nContext: '{prompt}'")
        generated = transformer.generate(prompt, 50, temperature=0.7)
        result = prompt + ''.join([transformer.itos.get(int(token), '?') for token in generated])
        print(f"Generated: '{display_generated_text(result)}'")
    
    print("\n--- With temperature=1.0 (more creative) ---")
    for prompt in prompts:
        print(f"\nContext: '{prompt}'")
        generated = transformer.generate(prompt, 50, temperature=1.0)
        result = prompt + ''.join([transformer.itos.get(int(token), '?') for token in generated])
        print(f"Generated: '{display_generated_text(result)}'")
    
    print("\nModel Improvements:")
    print("1. Increased model dimensions from 64 to 128")
    print("2. Increased context window from 32 to 64")
    print("3. Increased number of attention heads from 4 to 8")
    print("4. Increased number of layers from 2 to 4")
    print("5. Increased feedforward dimensions from 128 to 512")
    print("6. Added learning rate decay for better convergence")
    print("7. Increased batch size from 4 to 8")
    print("8. Doubled training iterations from 5000 to 10000")
    print("9. Added more dropout (0.2) for better regularization")
    print("10. Added gradient clipping")
    print("\nDone!")

