# A Mini Shakespearean Language Model from Scratch

This repo implements a small Transformer trained on Tiny Shakespeare.  

## Repository Structure

- `main.py` — Training entry point and configuration
- `model.py` — Transformer components (Absolute positional encoding, RoPE attention , Masked attention, Feed forward network and Mixture of experts (MoE) Model )
- `preprocessing.py` — Tokenization (char, tiktoken) & batch creation
- `data/` — Tiny Shakespeare dataset
- `loss_plot/` — Loss curves for experiments

## How to Run

1. Install dependencies:
   ```python
   pip install torch tiktoken matplotlib
   ```

2. Train:
   ```python
   python main.py
   ```

> Edit the config in \`main.py\` to switch tokenizers, positional encodings, attention, or MoE.

## Configuration Example

```python
Config = {
    "tokenization_method": "char",          # or "tiktoken"
    "positional_encoding_type": "Absolute", # or "None"
    "attention_mechanism": "MaskedAttentionHead",  # or "RoPE_attention"
    "feed_forward": "standard",             # or "MoE"
    "context_size": 32,
    "batch_size": 16,
    "d_model": 64,
    "n_heads": 4,
    "top_k": 2,
    "n_experts": 4,
    "dropout": 0.1,
    "epochs": 3000,
    "log_interval": 20
}

```
## References

- Karpathy — *Let's build GPT from scratch* (YouTube >>>> https://www.youtube.com/watch?v=zduSFxRajkE )  
- OpenAI \`tiktoken\`: https://github.com/openai/tiktoken
