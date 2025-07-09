# What is the goal of this project?

The goal is to understand the parts to a modern transformer model. 

What I have been realizing is that there is more than just attention. There is many more parts and components to make a modern LLM. What made me realize this is I saw the big bird paper and realized that there was more to it than just attention. This is my attempt to put all of the diffent key parts to a modern LLM into one spot. Hopefully I will even be able to implement it into a mini modern LLM.

Key papers:

## Foundational Papers:
1. **[Attention Is All You Need](papers/attention_is_all_you_need.md)** (Vaswani et al., 2017) - The original transformer paper
2. **[BERT: Pre-training of Deep Bidirectional Transformers](papers/bert.md)** (Devlin et al., 2018) - Bidirectional transformers
3. **[Language Models are Few-Shot Learners](papers/gpt3.md)** (GPT-3, Brown et al., 2020) - Scaling transformers

## Attention Mechanisms:
4. **[Big Bird: Transformers for Longer Sequences](papers/big_bird.md)** (Zaheer et al., 2020) - Efficient attention for long sequences
5. **[Longformer: The Long-Document Transformer](papers/longformer.md)** (Beltagy et al., 2020) - Sparse attention patterns
6. **[FlashAttention: Fast and Memory-Efficient Exact Attention](papers/flash_attention.md)** (Dao et al., 2022) - Hardware-efficient attention implementation

## Positional Encoding:
7. **[RoFormer: Enhanced Transformer with Rotary Position Embedding](papers/roformer.md)** (Su et al., 2021) - RoPE
8. **[ALiBi: Attention with Linear Biases](papers/alibi.md)** (Press et al., 2021) - Alternative to positional embeddings

## Architecture Improvements:
9. **[GLU Variants Improve Transformer](papers/glu_variants.md)** (Shazeer, 2020) - SwiGLU activation functions
10. **[Root Mean Square Layer Normalization](papers/rmsnorm.md)** (Zhang & Sennrich, 2019) - RMSNorm
11. **[On Layer Normalization in the Transformer Architecture](papers/layer_normalization.md)** (Xiong et al., 2020) - Pre-LN vs Post-LN

## Tokenization:
12. **[SentencePiece: A simple and language independent subword tokenizer](papers/sentencepiece.md)** (Kudo & Richardson, 2018)
13. **[Neural Machine Translation of Rare Words with Subword Units](papers/bpe.md)** (Sennrich et al., 2016) - BPE

## Training & Optimization:
14. **[Mixed Precision Training](papers/mixed_precision_training.md)** (Micikevicius et al., 2017) - Efficient training techniques
15. **[Scaling Laws for Neural Language Models](papers/scaling_laws.md)** (Kaplan et al., 2020) - Understanding model scaling 

# What are the parts of a transformer model?

1. Tokenization
2. Positional Encoding
3. Embedding
4. Self-Attention
5. Feed-Forward Network

