# 🧠 GPT-Style Text Generation with PyTorch

---

## 📌 Overview

This project is a **custom GPT-like character-level language model** implemented in **PyTorch**.  
It:

- Loads a vocabulary from `vocab.txt`
- Encodes and decodes text into token IDs
- Loads a pre-trained model (`model-1.pkl`)
- Generates text based on a user prompt using causal self-attention

The model architecture includes:

- Token & position embeddings
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Cross-entropy loss for training
- Sampling-based text generation

---

## ✨ Features

✅ Custom GPT-like Transformer model  
✅ Character-level encoding/decoding from vocabulary file  
✅ Text generation with temperature-free multinomial sampling  
✅ Uses **CUDA** if available for acceleration  
✅ Can load from pre-trained `.pkl` checkpoint  

---

Main Components
Head → Single self-attention head

MultiHeadAttention → Multiple heads concatenated

FeedForward → Fully connected layers for token processing

Block → Transformer block with attention + feed-forward

GPTLanguageModel → Stacks multiple blocks + embeddings + output head
