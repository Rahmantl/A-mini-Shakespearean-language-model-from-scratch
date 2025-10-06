import seaborn
import torch
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from preprocessing import get_batches, tokenize_train_val_split
from model import Transformer, evaluate_loss, MaskedAttentionHead, RoPE_attention



# def train(model, optimizer, print_logs=False):
#     losses_train = np.zeros(0)
#     losses_val = np.zeros(0)
#     start_time = time.time()
#     for epoch in range(Config['epochs']):
#         optimizer.zero_grad()
#         xs, ys = get_batches(train_data, val_data, "train", Config['batch_size'], Config['context_size'])
#         logits, loss = model(xs, targets=ys)

#         loss.backward()
#         optimizer.step()

#         if epoch % Config['log_interval'] == 0:
#             batch_time = time.time() - start_time
#             l_train, l_val = evaluate_loss(model, Config, train_data, val_data)
#             losses_train = np.append(losses_train, l_train)
#             losses_val = np.append(losses_val, l_val)
#             if print_logs:
#                 print(
#                     f"Epoch {epoch} | train loss {l_train:.3f} | val loss {l_val:.3f} | Time {batch_time:.3f} |")

#             start_time = time.time()
#     epochs = np.arange(0, losses_train.shape[0])
#     sns.set(style="whitegrid")

#     plt.figure(figsize=(8, 6))
#     sns.lineplot(x=epochs, y=losses_train, label='Training Loss', marker='o')
#     sns.lineplot(x=epochs, y=losses_val, label='Validation Loss', marker='o')

#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.show()

#     return losses_train, losses_val
#######################################
def train(model, optimizer, print_logs=False):
    losses_train = []
    losses_val = []
    start_time = time.time()

    model.train()
    for epoch in range(Config['epochs']):
        optimizer.zero_grad()
        xs, ys = get_batches(train_data, val_data, "train",
                             Config['batch_size'], Config['context_size'])
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()

        if epoch % Config['log_interval'] == 0:
            batch_time = time.time() - start_time

            
            l_train, l_val = evaluate_loss(model, Config, train_data, val_data)
            if torch.is_tensor(l_train):
                l_train = float(l_train.detach().cpu().item())
            if torch.is_tensor(l_val):
                l_val = float(l_val.detach().cpu().item())

            losses_train.append(l_train)
            losses_val.append(l_val)

            if print_logs:
                print(f"Epoch {epoch} | train {l_train:.3f} | val {l_val:.3f} | {batch_time:.3f}s")

            start_time = time.time()

    #check
    print(f"Collected {len(losses_train)} train points and {len(losses_val)} val points.")

    #Plot
    import seaborn as sns
    sns.set(style="whitegrid")
    epochs = np.arange(len(losses_train))

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=epochs, y=losses_train, label='Training Loss', marker='o')
    sns.lineplot(x=epochs, y=losses_val,   label='Validation Loss', marker='o')
    plt.xlabel('Log step (every %d epochs)' % Config['log_interval'])
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()

    #SAVE BEFORE SHOW
    plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
    print("Plot saved as loss_curve.png")

    #Show inline (if running in a notebook cell)
    try:
        plt.show()
    except Exception:
        pass

    return np.array(losses_train), np.array(losses_val)



#######################################
if __name__ == "__main__":

    Config = {
        'tokenization_method': "char",  # char or tiktoken
        'positional_encoding_type': "Absolute",  # Absolute, None
        'attention_mechanism': RoPE_attention,  # MaskedAttentionHead, RoPE_attention
        'feed_forward': "MoE",  # standard # MoE
        'context_size': 32,
        'batch_size': 16,
        'd_model': 64,
        'epochs': 3000,
        'log_interval': 100,
        'n_heads': 4,
        'top_k': 2,
        'n_experts': 4,
        'dropout': 0.1
    }

    train_data, val_data, vocab_size, decoder = tokenize_train_val_split(Config["tokenization_method"])
    
    Config.update({
        'vocab_size': vocab_size,
    })

    model = Transformer(Config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=1e-7, lr=1e-3)
    train(model, optimizer, print_logs=True)

    context, ys = get_batches(train_data, val_data, "train", 1, Config["context_size"])

    generated_tokens = model.generate(context, 2000).tolist()

    if Config["tokenization_method"] == "char":
        generated_text = [decoder(x) for x in generated_tokens]
    elif Config["tokenization_method"] == "tiktoken":
        generated_text = [decoder.decode(x) for x in generated_tokens]

    for text in generated_text:
        print(text)
