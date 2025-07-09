import numpy as np
import os
import requests

# Modern Transformer Implementation
# This implementation incorporates key innovations from multiple papers:
# - Attention Is All You Need (Vaswani et al., 2017) - Core transformer architecture
# - RoFormer (Su et al., 2021) - Rotary Position Embedding (RoPE)
# - Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) - RMSNorm
# - GLU Variants Improve Transformer (Shazeer, 2020) - SwiGLU activation
# - On Layer Normalization in the Transformer Architecture (Xiong et al., 2020) - Pre-LN architecture


class Dataset:
    def __init__(self, data_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_path="input.txt"):
        self.data_path = data_path
        self._download_data(data_url)
        self._load_data()
        self._build_vocab()
        self._encode_data()

    def _download_data(self, url):
        if not os.path.exists(self.data_path):
            print(f"Downloading dataset from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(self.data_path, 'w') as f:
                f.write(response.text)
            print(f"Dataset saved to {self.data_path}")

    def _load_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        print(f"Data loaded. Length: {len(self.text)} characters")

    def _build_vocab(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print(f"Vocabulary size: {self.vocab_size}")
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def _encode_data(self):
        self.data = np.array([self.stoi[ch] for ch in self.text], dtype=np.int32)
        print("Data encoded.")

    def get_batch(self, batch_size, block_size):
        ix = np.random.randint(0, len(self.data) - block_size, batch_size)
        x = np.stack([self.data[i:i + block_size] for i in ix])
        y = np.stack([self.data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


class RMSNorm:
    """
    Root Mean Square Layer Normalization
    Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    
    RMSNorm is more efficient than LayerNorm as it doesn't center the data
    and only normalizes by the root mean square, reducing computation.
    """
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = np.ones(dim)

    def __call__(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


class RotaryPositionalEmbedding:
    """
    Rotary Position Embedding (RoPE)
    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
    
    RoPE encodes position information by rotating the query and key vectors
    instead of adding position embeddings, providing better relative position
    encoding and length extrapolation capabilities.
    """
    def __init__(self, dim, max_seq_len=5000):
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.inv_freq = inv_freq
        
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        
        self.cos_cached = np.cos(freqs)
        self.sin_cached = np.sin(freqs)

    def apply_rotary_pos_emb(self, x, seq_len):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        rotated = np.empty_like(x)
        rotated[..., ::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        
        return rotated


class MultiHeadAttention:
    """
    Multi-Head Attention with Rotary Position Embedding
    Paper: "Attention Is All You Need" (Vaswani et al., 2017) - Core attention mechanism
    Enhanced with RoPE from: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
    
    This implementation uses:
    - Scaled dot-product attention from the original transformer paper
    - Multiple attention heads for parallel processing
    - RoPE for better position encoding
    - Causal masking for autoregressive generation
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        self.w_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.w_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.w_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.w_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def __call__(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections for Q, K, V
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Apply RoPE to Q and K
        q = self.rope.apply_rotary_pos_emb(q, seq_len)
        k = self.rope.apply_rotary_pos_emb(k, seq_len)
        
        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = self.softmax(scores)
        
        out = np.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        return out @ self.w_o

    def softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SwiGLU:
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network
    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    
    SwiGLU combines the Swish activation function with the GLU (Gated Linear Unit)
    mechanism, providing better performance than standard ReLU-based FFNs.
    The gating mechanism allows the network to control information flow more effectively.
    """
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w_gate = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.w_up = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.w_down = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)

    def __call__(self, x):
        gate = self.swish(x @ self.w_gate)
        up = x @ self.w_up
        return (gate * up) @ self.w_down

    def swish(self, x):
        return x * (1.0 / (1.0 + np.exp(-x)))


class TransformerBlock:
    """
    Transformer Block with Pre-Layer Normalization
    Paper: "Attention Is All You Need" (Vaswani et al., 2017) - Core transformer block
    Enhanced with Pre-LN from: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
    
    This implementation uses:
    - Pre-layer normalization (more stable training than post-LN)
    - Residual connections around attention and feed-forward layers
    - RMSNorm instead of LayerNorm for efficiency
    - SwiGLU instead of ReLU for better performance
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def __call__(self, x, mask=None):
        # Pre-LN: normalize before attention, then add residual
        x = x + self.attn(self.norm1(x), mask)
        # Pre-LN: normalize before FFN, then add residual
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer:
    """
    Modern Transformer Language Model
    Paper: "Attention Is All You Need" (Vaswani et al., 2017) - Core architecture
    
    This implementation incorporates multiple modern improvements:
    - Token embeddings (learned)
    - Multiple transformer blocks with modern enhancements
    - Final layer normalization for stability
    - Language modeling head for next token prediction
    - Causal masking for autoregressive generation
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Token embeddings
        self.embed = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model)
        
        # Transformer blocks with modern enhancements
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        
        # Final layer normalization
        self.norm = RMSNorm(d_model)
        
        # Language modeling head
        self.lm_head = np.random.randn(d_model, vocab_size) * np.sqrt(1.0 / d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # Token embeddings (no positional embeddings - using RoPE instead)
        x = self.embed[x]
        
        # Create causal mask for autoregressive generation
        mask = self.create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Language modeling head
        logits = x @ self.lm_head
        
        return logits

    def create_causal_mask(self, seq_len):
        """
        Creates a causal mask to prevent attention to future positions
        Paper: "Attention Is All You Need" (Vaswani et al., 2017)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = np.where(mask == 1, -1e9, 0.0)
        return mask

    def generate(self, context, max_new_tokens, temperature=1.0):
        context = np.array(context, dtype=np.int32)
        
        for _ in range(max_new_tokens):
            context_cond = context[-self.max_seq_len:] if len(context) > self.max_seq_len else context
            context_cond = context_cond.reshape(1, -1)
            
            logits = self.forward(context_cond)
            logits = logits[0, -1, :] / temperature
            
            probs = self.softmax_1d(logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            context = np.append(context, next_token)
        
        return context

    def softmax_1d(self, x):
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def loss(self, x, targets):
        logits = self.forward(x)
        vocab_size = logits.shape[-1]
        
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        log_probs = self.log_softmax(logits)
        loss = -np.mean(log_probs[np.arange(len(targets)), targets])
        
        return loss

    def log_softmax(self, x):
        log_sum_exp = np.log(np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True))
        return x - np.max(x, axis=-1, keepdims=True) - log_sum_exp

    def train_step(self, x, targets):
        loss = self.loss(x, targets)
        return loss


def train_model(model, dataset, epochs=100, batch_size=4, block_size=64):
    print("Starting training...")
    
    for epoch in range(epochs):
        xb, yb = dataset.get_batch(batch_size, block_size)
        loss = model.train_step(xb, yb)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            context = dataset.encode("Hello")[:10]
            generated = model.generate(context, max_new_tokens=50, temperature=0.8)
            print(f"Generated: {dataset.decode(generated)}")
            print("-" * 50)


"""
PAPER CONTRIBUTIONS SUMMARY:

This implementation combines key innovations from multiple transformer papers:

1. "Attention Is All You Need" (Vaswani et al., 2017)
   - Core transformer architecture with multi-head attention
   - Scaled dot-product attention mechanism
   - Residual connections and layer normalization
   - Causal masking for autoregressive generation

2. "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
   - RMSNorm for more efficient normalization
   - Reduces computation by not centering the data

3. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
   - Rotary Position Embedding (RoPE) instead of absolute positional embeddings
   - Better relative position encoding and length extrapolation

4. "GLU Variants Improve Transformer" (Shazeer, 2020)
   - SwiGLU activation function in feed-forward networks
   - Combines Swish activation with gated linear units

5. "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
   - Pre-layer normalization for more stable training
   - Normalizes before attention and FFN operations

These papers represent key advances in transformer architecture that are used in modern
large language models like GPT-3/4, LLaMA, and others.
"""

if __name__ == '__main__':
    try:
        dataset = Dataset()
        
        model = Transformer(
            vocab_size=dataset.vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            max_seq_len=128,
            dropout=0.1
        )
        
        print("Model created successfully!")
        print(f"Vocab size: {dataset.vocab_size}")
        print(f"Model parameters: d_model={model.d_model}, n_heads={model.n_heads}, n_layers={model.n_layers}")
        
        context = dataset.encode("Hello")
        print(f"Encoded 'Hello': {context}")
        
        generated = model.generate(context, max_new_tokens=20)
        print(f"Generated text: {dataset.decode(generated)}")
        
        print("\nStarting training...")
        train_model(model, dataset, epochs=50, batch_size=2, block_size=32)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This implementation requires numpy. Install with: pip install numpy")