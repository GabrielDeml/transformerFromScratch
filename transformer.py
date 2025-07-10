import numpy as np
import os
import requests

# Modern Transformer Implementation with Training
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


class AdamOptimizer:
    """Adam optimizer for gradient updates"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize moment estimates
        self.m = {id(p): np.zeros_like(p) for p in params}
        self.v = {id(p): np.zeros_like(p) for p in params}
    
    def step(self, grads):
        self.t += 1
        
        for p, g in zip(self.params, grads):
            if g is None:
                continue
                
            # Update biased moments
            self.m[id(p)] = self.beta1 * self.m[id(p)] + (1 - self.beta1) * g
            self.v[id(p)] = self.beta2 * self.v[id(p)] + (1 - self.beta2) * g**2
            
            # Bias correction
            m_hat = self.m[id(p)] / (1 - self.beta1**self.t)
            v_hat = self.v[id(p)] / (1 - self.beta2**self.t)
            
            # Update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class LionOptimizer:
    """
    Lion Optimizer - Evolved Sign Momentum
    Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
    
    Lion is more memory-efficient than Adam (uses only momentum, not variance)
    and often achieves better performance with appropriate hyperparameters.
    Recommended lr for Lion is typically 3-10x smaller than Adam.
    """
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        
        # Only need momentum (no variance like Adam)
        self.m = {id(p): np.zeros_like(p) for p in params}
    
    def step(self, grads):
        for p, g in zip(self.params, grads):
            if g is None:
                continue
            
            # Get momentum for this parameter
            m = self.m[id(p)]
            
            # Weight decay (decoupled, applied directly to parameters)
            if self.weight_decay > 0:
                p *= (1 - self.lr * self.weight_decay)
            
            # Lion update: use sign of interpolation between gradient and momentum
            update = self.beta1 * m + (1 - self.beta1) * g
            p -= self.lr * np.sign(update)
            
            # Update momentum with EMA of gradient
            self.m[id(p)] = self.beta2 * m + (1 - self.beta2) * g


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
        self.weight_grad = None

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / self.rms
    
    def backward(self, grad_output):
        # Gradient w.r.t weight
        self.weight_grad = np.sum(grad_output * self.x / self.rms, axis=(0, 1))
        
        # Gradient w.r.t input
        grad_normed = grad_output * self.weight / self.rms
        grad_rms = -np.sum(grad_output * self.weight * self.x / (self.rms**3), axis=-1, keepdims=True) * self.x / self.rms
        
        return grad_normed + grad_rms


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
        
        # Create position frequency matrix
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.inv_freq = inv_freq
        
        # Precompute sin and cos for all positions
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        
        self.cos_cached = np.cos(freqs)
        self.sin_cached = np.sin(freqs)

    def apply_rotary_pos_emb(self, x, seq_len):
        # x shape: [batch, heads, seq_len, head_dim]
        batch, heads, _, head_dim = x.shape
        
        # Get cached sin/cos values for current sequence length
        cos = self.cos_cached[:seq_len].reshape(1, 1, seq_len, -1)
        sin = self.sin_cached[:seq_len].reshape(1, 1, seq_len, -1)
        
        # Split x into even and odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        rotated = np.empty_like(x)
        rotated[..., ::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        
        return rotated


class ALiBi:
    """
    Attention with Linear Biases (ALiBi)
    Paper: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al., 2021)
    
    ALiBi adds linear biases to attention scores based on relative positions,
    eliminating the need for position embeddings and enabling better length extrapolation.
    """
    def __init__(self, n_heads, max_seq_len=5000):
        self.n_heads = n_heads
        
        # Calculate slopes for each head using geometric sequence
        # For n_heads=8: slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-2 ** -(np.log2(n) - 3))
                ratio = start
                return [start * (ratio ** i) for i in range(n)]
            
            if np.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                # If not power of 2, interpolate
                closest_power_of_2 = 2 ** int(np.floor(np.log2(n)))
                return get_slopes_power_of_2(closest_power_of_2) + \
                       get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
        
        slopes = np.array(get_slopes(n_heads)).reshape(1, n_heads, 1, 1)
        self.slopes = -slopes  # Negative slopes for penalty
        
        # Precompute relative position matrix
        positions = np.arange(max_seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = np.triu(relative_positions)  # Only use upper triangle (causal)
        self.relative_positions = relative_positions
    
    def get_alibi_biases(self, seq_len):
        # Get the relative positions for current sequence length
        rel_pos = self.relative_positions[:seq_len, :seq_len]
        
        # Apply slopes to get biases
        # Shape: [1, n_heads, seq_len, seq_len]
        alibi_biases = rel_pos[None, None, :, :] * self.slopes
        
        return alibi_biases


class MultiHeadAttention:
    """
    Multi-Head Attention with configurable position encoding
    Paper: "Attention Is All You Need" (Vaswani et al., 2017) - Core attention mechanism
    Enhanced with:
    - RoPE from: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
    - ALiBi from: "Train Short, Test Long: Attention with Linear Biases" (Press et al., 2021)
    """
    def __init__(self, d_model, n_heads, dropout=0.1, pos_encoding='rope', max_seq_len=5000, qk_norm=False):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.pos_encoding = pos_encoding
        self.qk_norm = qk_norm
        
        # Xavier initialization for better training
        self.w_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (d_model + d_model))
        self.w_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (d_model + d_model))
        self.w_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (d_model + d_model))
        self.w_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / (d_model + d_model))
        
        # Gradients
        self.w_q_grad = None
        self.w_k_grad = None
        self.w_v_grad = None
        self.w_o_grad = None
        
        # QK-Normalization parameters
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        
        # Initialize position encoding
        if pos_encoding == 'rope':
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
            self.alibi = None
        elif pos_encoding == 'alibi':
            self.rope = None
            self.alibi = ALiBi(n_heads, max_seq_len)
        else:  # no position encoding
            self.rope = None
            self.alibi = None
            
        self.training = True

    def forward(self, x, mask=None):
        self.x = x
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections for Q, K, V
        self.q = x @ self.w_q
        self.k = x @ self.w_k
        self.v = x @ self.w_v
        
        # Reshape for multi-head attention
        q = self.q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose to [batch, heads, seq_len, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Apply position encoding
        if self.rope is not None:
            # Apply RoPE to Q and K
            q = self.rope.apply_rotary_pos_emb(q, seq_len)
            k = self.rope.apply_rotary_pos_emb(k, seq_len)
        
        # Apply QK-Normalization if enabled
        if self.qk_norm:
            # Normalize Q and K across head_dim dimension
            # Shape: [batch, heads, seq_len, head_dim]
            q_shape = q.shape
            k_shape = k.shape
            
            # Reshape for normalization
            q_for_norm = q.transpose(0, 2, 1, 3).reshape(-1, self.head_dim)
            k_for_norm = k.transpose(0, 2, 1, 3).reshape(-1, self.head_dim)
            
            # Apply normalization
            q_normed = self.q_norm.forward(q_for_norm)
            k_normed = self.k_norm.forward(k_for_norm)
            
            # Reshape back
            q = q_normed.reshape(q_shape[0], q_shape[2], q_shape[1], q_shape[3]).transpose(0, 2, 1, 3)
            k = k_normed.reshape(k_shape[0], k_shape[2], k_shape[1], k_shape[3]).transpose(0, 2, 1, 3)
        
        # Store for backward
        self.q_rot = q
        self.k_rot = k
        self.v_perm = v
        
        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply ALiBi biases if using ALiBi
        if self.alibi is not None:
            alibi_biases = self.alibi.get_alibi_biases(seq_len)
            scores = scores + alibi_biases
        
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        self.attn_weights = self.softmax(scores)
        
        # Apply dropout during training
        if self.training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, self.attn_weights.shape) / (1 - self.dropout)
            self.attn_weights_dropout = self.attn_weights * dropout_mask
        else:
            self.attn_weights_dropout = self.attn_weights
        
        # Attention output
        out = np.matmul(self.attn_weights_dropout, v)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        self.out_before_proj = out
        
        # Output projection
        return out @ self.w_o

    def backward(self, grad_output):
        batch_size, seq_len, d_model = grad_output.shape
        
        # Gradient w.r.t output projection
        # out_before_proj shape: (batch_size, seq_len, d_model)
        out_flat = self.out_before_proj.reshape(-1, d_model)  # (batch_size * seq_len, d_model)
        grad_output_flat = grad_output.reshape(-1, d_model)  # (batch_size * seq_len, d_model)
        self.w_o_grad = out_flat.T @ grad_output_flat  # (d_model, d_model)
        
        # Gradient through output projection
        grad_out = grad_output @ self.w_o.T
        grad_out = grad_out.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        grad_out = grad_out.transpose(0, 2, 1, 3)
        
        # Gradient through attention
        grad_v = self.attn_weights_dropout.transpose(0, 1, 3, 2) @ grad_out
        grad_attn = grad_out @ self.v_perm.transpose(0, 1, 3, 2)
        
        # Gradient through softmax
        grad_scores = grad_attn * self.attn_weights
        grad_scores = grad_scores - self.attn_weights * np.sum(grad_scores, axis=-1, keepdims=True)
        grad_scores = grad_scores * self.scale
        
        # Gradient through Q @ K^T
        grad_q = grad_scores @ self.k_rot
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ self.q_rot
        
        # Reshape gradients
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Gradient w.r.t weight matrices
        x_flat = self.x.reshape(-1, d_model)
        self.w_q_grad = x_flat.T @ grad_q.reshape(-1, d_model)
        self.w_k_grad = x_flat.T @ grad_k.reshape(-1, d_model)
        self.w_v_grad = x_flat.T @ grad_v.reshape(-1, d_model)
        
        # Gradient w.r.t input
        grad_x = grad_q @ self.w_q.T + grad_k @ self.w_k.T + grad_v @ self.w_v.T
        
        return grad_x

    def softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SwiGLU:
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network
    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.hidden_dim = d_ff
        
        # He initialization for ReLU-like activations
        self.w_gate = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.w_up = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.w_down = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        
        # Gradients
        self.w_gate_grad = None
        self.w_up_grad = None
        self.w_down_grad = None

    def forward(self, x):
        self.x = x
        
        # Gate and up projections
        self.gate_pre_act = x @ self.w_gate
        self.gate = self.swish(self.gate_pre_act)
        self.up = x @ self.w_up
        
        # Gated activation
        self.gated = self.gate * self.up
        
        # Down projection
        return self.gated @ self.w_down

    def backward(self, grad_output):
        batch_size, seq_len, _ = grad_output.shape
        
        # Reshape for matrix operations
        grad_output_flat = grad_output.reshape(-1, self.d_model)
        gated_flat = self.gated.reshape(-1, self.hidden_dim)
        
        # Gradient w.r.t down projection
        self.w_down_grad = gated_flat.T @ grad_output_flat
        
        # Gradient through down projection
        grad_gated_flat = grad_output_flat @ self.w_down.T
        grad_gated = grad_gated_flat.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Gradient through gating
        grad_gate = grad_gated * self.up
        grad_up = grad_gated * self.gate
        
        # Gradient through swish
        grad_gate_pre = grad_gate * self.swish_derivative(self.gate_pre_act)
        
        # Gradient w.r.t weight matrices
        x_flat = self.x.reshape(-1, self.d_model)
        self.w_gate_grad = x_flat.T @ grad_gate_pre.reshape(-1, self.hidden_dim)
        self.w_up_grad = x_flat.T @ grad_up.reshape(-1, self.hidden_dim)
        
        # Gradient w.r.t input
        grad_x = grad_gate_pre @ self.w_gate.T + grad_up @ self.w_up.T
        
        return grad_x

    def swish(self, x):
        return x * (1.0 / (1.0 + np.exp(-x)))
    
    def swish_derivative(self, x):
        sig = 1.0 / (1.0 + np.exp(-x))
        return sig + x * sig * (1 - sig)


class TransformerBlock:
    """
    Transformer Block with Pre-Layer Normalization
    Paper: "Attention Is All You Need" (Vaswani et al., 2017) - Core transformer block
    Enhanced with Pre-LN from: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pos_encoding='rope', max_seq_len=5000, qk_norm=False):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding, max_seq_len, qk_norm)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None):
        # Store for residual connections
        self.x1 = x
        
        # Pre-LN: normalize before attention, then add residual
        self.norm1_out = self.norm1.forward(x)
        self.attn_out = self.attn.forward(self.norm1_out, mask)
        x = x + self.attn_out
        
        # Store for second residual
        self.x2 = x
        
        # Pre-LN: normalize before FFN, then add residual
        self.norm2_out = self.norm2.forward(x)
        self.ffn_out = self.ffn.forward(self.norm2_out)
        x = x + self.ffn_out
        
        return x
    
    def backward(self, grad_output):
        # Gradient through second residual
        grad_ffn = grad_output
        grad_x2 = grad_output
        
        # Gradient through FFN
        grad_norm2_out = self.ffn.backward(grad_ffn)
        
        # Gradient through norm2
        grad_x2 += self.norm2.backward(grad_norm2_out)
        
        # Gradient through first residual
        grad_attn = grad_x2
        grad_x1 = grad_x2
        
        # Gradient through attention
        grad_norm1_out = self.attn.backward(grad_attn)
        
        # Gradient through norm1
        grad_x1 += self.norm1.backward(grad_norm1_out)
        
        return grad_x1


class Transformer:
    """
    Modern Transformer Language Model with Training
    Paper: "Attention Is All You Need" (Vaswani et al., 2017) - Core architecture
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1, 
                 optimizer='lion', lr=None, weight_decay=0.0, pos_encoding='rope', qk_norm=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.qk_norm = qk_norm
        
        # Token embeddings with proper initialization
        self.embed = np.random.randn(vocab_size, d_model) * np.sqrt(2.0 / vocab_size)
        self.embed_grad = None
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout, pos_encoding, max_seq_len, qk_norm) 
                      for _ in range(n_layers)]
        
        # Final layer normalization
        self.norm = RMSNorm(d_model)
        
        # Language modeling head (tied with embeddings for efficiency)
        self.lm_head = self.embed.T.copy()
        self.lm_head_grad = None
        
        # Collect all parameters for optimizer
        self.params = [self.embed, self.lm_head]
        for block in self.blocks:
            self.params.extend([
                block.attn.w_q, block.attn.w_k, block.attn.w_v, block.attn.w_o,
                block.ffn.w_gate, block.ffn.w_up, block.ffn.w_down,
                block.norm1.weight, block.norm2.weight
            ])
        self.params.append(self.norm.weight)
        
        # Initialize optimizer
        if optimizer == 'lion':
            # Lion typically needs 3-10x smaller lr than Adam
            default_lr = 3e-4 if lr is None else lr
            self.optimizer = LionOptimizer(self.params, lr=default_lr, weight_decay=weight_decay)
        else:  # adam
            default_lr = 1e-3 if lr is None else lr
            self.optimizer = AdamOptimizer(self.params, lr=default_lr)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # Token embeddings
        self.embed_input = x
        x = self.embed[x]
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        self.block_outputs = []
        for block in self.blocks:
            x = block.forward(x, mask)
            self.block_outputs.append(x)
        
        # Final layer normalization
        x = self.norm.forward(x)
        self.norm_out = x
        
        # Language modeling head
        logits = x @ self.lm_head
        
        return logits
    
    def backward(self, grad_logits):
        # Gradient w.r.t LM head
        # norm_out: (batch_size, seq_len, d_model)
        # grad_logits: (batch_size, seq_len, vocab_size)
        norm_out_reshaped = self.norm_out.reshape(-1, self.d_model)  # (batch_size * seq_len, d_model)
        grad_logits_reshaped = grad_logits.reshape(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
        self.lm_head_grad = norm_out_reshaped.T @ grad_logits_reshaped  # (d_model, vocab_size)
        
        # Gradient through LM head
        grad_norm_out = grad_logits @ self.lm_head.T
        
        # Gradient through final norm
        grad_x = self.norm.backward(grad_norm_out)
        
        # Gradient through transformer blocks (reverse order)
        for i in reversed(range(self.n_layers)):
            grad_x = self.blocks[i].backward(grad_x)
        
        # Gradient w.r.t embeddings
        batch_size, seq_len = self.embed_input.shape
        self.embed_grad = np.zeros_like(self.embed)
        grad_x_flat = grad_x.reshape(-1, self.d_model)
        embed_input_flat = self.embed_input.reshape(-1)
        
        # Accumulate gradients for each token
        for i in range(len(embed_input_flat)):
            self.embed_grad[embed_input_flat[i]] += grad_x_flat[i]

    def create_causal_mask(self, seq_len):
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = np.where(mask == 1, -1e9, 0.0)
        return mask

    def generate(self, context, max_new_tokens, temperature=1.0):
        self.set_training(False)
        context = np.array(context, dtype=np.int32)
        
        for _ in range(max_new_tokens):
            context_cond = context[-self.max_seq_len:] if len(context) > self.max_seq_len else context
            context_cond = context_cond.reshape(1, -1)
            
            logits = self.forward(context_cond)
            logits = logits[0, -1, :] / temperature
            
            probs = self.softmax_1d(logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            context = np.append(context, next_token)
        
        self.set_training(True)
        return context

    def softmax_1d(self, x):
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def loss(self, x, targets):
        logits = self.forward(x)
        vocab_size = logits.shape[-1]
        
        # Store for backward
        self.logits = logits
        self.targets = targets
        
        # Reshape for loss computation
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        # Cross entropy loss
        log_probs = self.log_softmax(logits)
        loss = -np.mean(log_probs[np.arange(len(targets)), targets])
        
        return loss

    def log_softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
        return x - x_max - log_sum_exp

    def train_step(self, x, targets):
        # Forward pass
        loss = self.loss(x, targets)
        
        # Compute gradients for cross entropy loss
        batch_size, seq_len, vocab_size = self.logits.shape
        logits_flat = self.logits.reshape(-1, vocab_size)
        targets_flat = self.targets.reshape(-1)
        
        # Gradient of cross entropy loss
        probs = np.exp(self.log_softmax(logits_flat))
        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        grad_logits = grad_logits.reshape(batch_size, seq_len, vocab_size) / len(targets_flat)
        
        # Backward pass
        self.backward(grad_logits)
        
        # Collect gradients
        grads = [self.embed_grad, self.lm_head_grad]
        for block in self.blocks:
            grads.extend([
                block.attn.w_q_grad, block.attn.w_k_grad, 
                block.attn.w_v_grad, block.attn.w_o_grad,
                block.ffn.w_gate_grad, block.ffn.w_up_grad, block.ffn.w_down_grad,
                block.norm1.weight_grad, block.norm2.weight_grad
            ])
        grads.append(self.norm.weight_grad)
        
        # Update parameters
        self.optimizer.step(grads)
        
        return loss
    
    def set_training(self, mode):
        for block in self.blocks:
            block.attn.training = mode


class CosineScheduler:
    """
    Cosine Learning Rate Schedule with Linear Warmup
    Combines linear warmup for initial stability with cosine decay for smooth convergence
    """
    def __init__(self, optimizer, warmup_steps=100, max_lr=None, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.max_lr = max_lr if max_lr is not None else self.base_lr
        self.current_step = 0
        
    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
            
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (self.current_step - self.warmup_steps) / max(1, self.current_step)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            
        self.optimizer.lr = lr
        return lr


def train_model(model, dataset, epochs=1000, batch_size=4, block_size=64, use_scheduler=True):
    print("Starting training...")
    
    # Initialize scheduler if requested
    scheduler = None
    if use_scheduler:
        scheduler = CosineScheduler(model.optimizer, warmup_steps=100, max_lr=model.optimizer.lr)
    
    for epoch in range(epochs):
        xb, yb = dataset.get_batch(batch_size, block_size)
        loss = model.train_step(xb, yb)
        
        # Update learning rate
        if scheduler:
            lr = scheduler.step(epoch)
        
        if epoch % 100 == 0:
            lr_str = f", LR: {model.optimizer.lr:.6f}" if scheduler else ""
            print(f"Epoch {epoch}, Loss: {loss:.4f}{lr_str}")
            
            context = dataset.encode("Hello")
            generated = model.generate(context, max_new_tokens=50, temperature=0.8)
            print(f"Generated: {dataset.decode(generated)}")
            print("-" * 50)


if __name__ == '__main__':
    try:
        dataset = Dataset()
        
        # Create model with new features
        model = Transformer(
            vocab_size=dataset.vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            max_seq_len=128,
            dropout=0.1,
            optimizer='lion',      # Use Lion optimizer
            lr=3e-4,              # Lower learning rate for Lion
            weight_decay=0.01,    # Add weight decay regularization
            pos_encoding='alibi', # Try ALiBi instead of RoPE
            qk_norm=True         # Enable QK-normalization
        )
        
        print("Model created successfully with advanced features!")
        print(f"Vocab size: {dataset.vocab_size}")
        print(f"Model config:")
        print(f"  - Architecture: d_model={model.d_model}, n_heads={model.n_heads}, n_layers={model.n_layers}")
        print(f"  - Optimizer: Lion (lr={model.optimizer.lr}, weight_decay={model.optimizer.weight_decay})")
        print(f"  - Position Encoding: {model.pos_encoding.upper()}")
        print(f"  - QK-Normalization: {'Enabled' if model.qk_norm else 'Disabled'}")
        print(f"  - Learning Rate Schedule: Cosine with warmup")
        
        context = dataset.encode("Hello")
        print(f"\nEncoded 'Hello': {context}")
        
        generated = model.generate(context, max_new_tokens=20)
        print(f"Generated text: {dataset.decode(generated)}")
        
        print("\nStarting training...")
        train_model(model, dataset, epochs=1000, batch_size=8, block_size=64, use_scheduler=True)
        
        print("\nTraining complete! Final generation:")
        context = dataset.encode("To be or not to be")
        generated = model.generate(context, max_new_tokens=100, temperature=0.8)
        print(f"Prompt: 'To be or not to be'")
        print(f"Generated: {dataset.decode(generated)}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        print("Note: This implementation requires numpy. Install with: pip install numpy")