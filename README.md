# A Mini Shakespearean Language Model from Scratch

This repo implements a small Transformer trained on Tiny Shakespeare.  

## Repository Structure

- `main.py` — training entry point and configuration
- `model.py` — Transformer components (Absolute PE, RoPE, masked attention, FFN/MoE)
- `preprocessing.py` — tokenization (char, tiktoken) & batch creation
- `data/` — Tiny Shakespeare dataset
- `loss_plot/` — loss curves for experiments

## How to Run

1. Install dependencies:
   \`\`\`bash
   pip install torch tiktoken matplotlib
   \`\`\`

2. Train:
   \`\`\`bash
   python main.py
   \`\`\`

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

- Karpathy — *Let's build GPT from scratch* (YouTube)  
- OpenAI \`tiktoken\`: https://github.com/openai/tiktoken
