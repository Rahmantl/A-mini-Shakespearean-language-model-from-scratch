import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset

from preprocessing import get_batches

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class Transformer(nn.Module):
    def __init__(self, config):
        """
        Creates the Transformer model in accordance with the config file.
        Args:
        :param config:
        """
        super().__init__()
        self.config = config
        # Embedding layer
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # Positional encoding layer
        self.positional_encoding = (
            AbsolutePE(config) if config['positional_encoding_type'] == "Absolute"
            else None
        )

        # Attention layer
        self.attention = MaskedMultiheadAttention(config)

        # Feed-forward network
        if config["feed_forward"] == "standard":
            self.linear = Standard_FF(config)
        elif config["feed_forward"] == "MoE":
            self.linear = MoEModel(config)
        else:
            raise ValueError(f"Unsupported feed-forward type: {config['feed_forward']}")

        # Normalization layers
        self.norm_1 = nn.RMSNorm(config['d_model'])
        self.norm_2 = nn.RMSNorm(config['d_model'])

        # Final linear layer
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, x, targets=None):
        # x and targets are (B, T) array of indices in the range 0 to vocab_size
        # if targets is provided, we compute the loss
        # if targets is None, we are in generation mode and only return the logits
        # since here we are comparing against the cross-entropy loss, we can directly use the logits (comparing with ground truth indices)
        x = self.embedding(x)
        
        if self.positional_encoding is not None:
            x = x + self.positional_encoding()

        x_att = self.attention(x)

        x = self.norm_1(x + x_att)
        x_lin = self.linear(x)
        x = self.norm_2(x + x_lin)
        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits

    def generate(self, context, max_new_tokens):
        # contex is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #idx_cond is the curent index
            idx_cond = context[:, -self.config["context_size"]:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # focus on the last time step (B, vocab_size), last element in time dimension
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities (B, vocab_size)
            # sample from the distribution
            #we asked pytorch to give us one sample ((B, 1)), because in each batch dimension we are going to single prediction
            idx_next = torch.multinomial(probs, num_samples=1) 
            #idx_ next comes from the sampling process, according to the sampling probability  distribution (prob), we get the next index
            
            # append sampled index to the running sequence
            context = torch.cat((context, idx_next), dim=1) # (B, T+1)
        return context
##
# Since

class AbsolutePE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Todo Implement the absolute positional encoding mechanism.
        #  The positional information can be computed once during initialization of the class and then returned
        #  during training and inference.
        
        d=config['d_model']
        T=config['context_size']#maximum length of the sequence
        alpha= 10000# a constant
        
        n= torch.arange(T, dtype=torch.float32).unsqueeze(1) #(T,1)
        ell = torch.arange(d, dtype=torch.float32).unsqueeze(0) #(1,d)
        theta= n/(alpha**((2*ell)/float(d))) #(T,d)
        
        PE= torch.empty(T,d,dtype= torch.float32) #(T,d)

        for dim in range(d):
            if dim%2==0:
                theta[:,dim]= torch.cos(theta[:,dim])
            else:
                theta[:,dim]= torch.sin(theta[:,dim])
        self.PE= PE
    
    
    def forward(self):
        # Todo return the correct matrix
        return self.PE


class MaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.W_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.W_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x):
        # Todo W_q, W_k and W_v have been initialized for you. Compute and return the output of the masked attention
        #  mechanism. Also ensure that dropout from the config file is correctly applied.
        ##
        # x: input tensor of shape (batch, sequence length, dimension)
        B, T, d = x.shape
        # Linear transformations for Q, K and V
        Q=self.W_q(x)
        K=self.W_k(x)
        V=self.W_v(x)
        
        #Compute attention scores (similarity)
        # what I am doing here is to compute the dot product between the query and key matrices, it measures how similar each query is to each key  
        scores = torch.matmul(Q, K.transpose(-2, -1))/ (d ** 0.5)  # Q is (B, T, d), K is (B, T, d) so K^T is (B, d, T) and scores is (B, T, T)
        # with the scaling factor 1/sqrt(d) to prevent large dot product values and control the variance of the scores
        ##########################
        
        
        #what I am doing here is to create a mask that is lower triangular with 1s in the lower triangle and 0s in the upper triangle
        #then I am using this mask to set the scores in the upper triangle to -inf
        #now, when we apply softmax, the scores in the upper triangle will be 0 and the scores in the lower triangle will be normalized
        #with this way we ensure that each position can only attend to previous positions and itself and not to future positions
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=1).bool()# lower triangular mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        ## Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights) # Apply dropout to attention weights
        
        ## Weighted sum of values
        # (B, T, d): matrix product of two matrices with shapes,
        # attention_weight is the softmax output, V is the value that we aggregate according to the attention weights
        out = torch.matmul(attention_weights, V) 
        
        return out


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([
            config["attention_mechanism"](config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'],
                                config['d_model'])

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        return x


class RoPE_attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_cos, self.rope_sin = self._construct_sin_cos(config)
        self.W_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.W_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.W_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.sqrt_d = 1.0 / torch.sqrt(torch.tensor(config['d_model']))
        self.dropout = nn.Dropout(config['dropout'])

    def _construct_sin_cos(self, config):
        # TODO construct the sin and cosine vectors with their respective m and theta values.
        
        d = config['d_model']
        T = config['context_size']
        
        # d must be even
        if d % 2 != 0:
            raise ValueError("d_model must be even for RoPE attention.") # page 5 of the paper (https://arxiv.org/pdf/2104.09864)
        
        i= torch.arange(d//2, dtype=torch.float32)
        theta_i= (10000**(-2*(i)/d)) #(d/2,)
        
        # m = 0...T-1 (positions)
        m = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        # angel for (m,i), each position and each dimension
        angle_m_i = m * theta_i.unsqueeze(0) #(T, d/2)
        
        cos = torch.cos(angle_m_i).unsqueeze(0)# (1, T, d/2)
        sin = torch.sin(angle_m_i).unsqueeze(0)#(1, T, d/2)
        
        return cos, sin

    def _rotate(self, x):
        # TODO: Implement rotation of the input 'x' using the RoPE (Rotary Position Embedding) mechanism.
        #  You may find it helpful to refer to Equation 34 in the official paper (https://arxiv.org/pdf/2104.09864).
        #  However, feel free to use an alternative equivalent implementation if it simplifies the process.
        B, T, d = x.shape
        half = d // 2

        #Prepare output tensor
        out = torch.empty_like(x)

        # Loop over time positions and 2D pairs:
        # What I am iterate over each time step and each pair of dimensions (2j, 2j+1), then I apply the rotation matrix to the x_even and x_odd,
        # if we see the equation 34 in the paper, the rotation matrix is a 2x2 matrix with a loop over the indices of the matrix,
        # I think having loop in a code is not very efficient and I think we can use vectorized operation here, but for now I will implement this way because of the time constraint
        for t in range(T):
            for j in range(half):
                # fetch cos/sin for this position t and pair j
                c = self.rope_cos[0, t, j]  # scalar
                s = self.rope_sin[0, t, j]  # scalar

                # current pair (B,) each: even = dim 2j, odd = dim 2j+1
                x_even = x[:, t, 2*j]
                x_odd  = x[:, t, 2*j + 1]

                # apply 2x2 rotation:
                # [x_even'; x_odd'] = [[ c, -s],
                #                      [ s,  c]] @ [x_even; x_odd]
                new_even = x_even * c - x_odd * s
                new_odd  = x_even * s + x_odd * c

                # write back
                out[:, t, 2*j]     = new_even
                out[:, t, 2*j + 1] = new_odd
        
        return out

    def forward(self, x):
        # TODO Implement the forward pass using RoPE attention. Use the _rotate function which you have just defined
        #  to rotate the queries and keys. Afterwards implement the standard attention mechanism from scratch.
        #  Dont forget to implement mask and dropout.
        
        B, T, d = x.shape
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Rotate Q and K
        Q_rot = self._rotate(Q)
        K_rot = self._rotate(K)
        
        # score computation
        scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1))/ (d ** 0.5)
        # mask 
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=1).bool()# lower triangular mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) 

        # softmax + dropout
        attention_weights = F.softmax(scores, dim=-1)       # (B, T, T)
        attention_weights = self.dropout(attention_weights) # Apply dropout to attention weights , (B, T, T)      
        out = torch.matmul(attention_weights, V)            # (B, T, d): matrix product of two matrices with shapes,
                                                            # attention_weight is the softmax output, V is the value that we aggregate according to 
        return out


class Standard_FF(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO Implement the architecture as in the task pdf.
        
        d = config['d_model'] # model dimension
        p = config['dropout'] # dropout probability

        # Define the layers in order:
        self.fc1 = nn.Linear(d, 4 * d) # Linear Layer 1
        self.relu = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(4 * d, d) # Linear Layer 2
        self.dropout = nn.Dropout(p)   # Dropout layer

    def forward(self, x):
        # TODO make a forward pass through the network.
        
        # x: (B, T, d)
        x = self.fc1(x)        # (B, T, 4d)
        x = self.relu(x)       # (B, T, 4d)
        x = self.fc2(x)        # (B, T, d)
        x = self.dropout(x)    # (B, T, d)
        return x


class Expert(nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(Expert, self).__init__()
        self.config = config
        # TODO Implement the Expert architecture in accordance with the PDF.
        p = config['dropout']
        d = input_size  # = hidden_size = d_model per assignment

        
        self.fc1 = nn.Linear(d, d)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d, d)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # TODO Do a forward pass through the network.
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MoEModel(nn.Module):
    def __init__(self, config):
        super(MoEModel, self).__init__()
        self.config = config
        # TODO Implement the gating network and the main architecture of the experts.
        d = config['d_model']
        n_experts = config['n_experts']
        top_k = config['top_k']
        p = config['dropout']

        self.top_k = top_k
        self.n_experts = n_experts
        
        # Create a list of experts
        self.experts = nn.ModuleList([
            Expert(config, d, d) for _ in range(n_experts)
        ])
        # Gating network: one linear layer
        self.gate = nn.Linear(d, n_experts)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
            # TODO Implement a forward pass through the network. Ensure that only the output of the sparsely selected
            #  networks is evaluated.
            B, T, d = x.shape  # x: (B, T, d)
        
            # Flatten the input to (B*T, d) for processing
            x_flat = x.view(-1, d)  # (B*T, d)         
        
            # Compute gating scores
            gate_scores = self.gate(x_flat)  # (B*T, n_experts)
        
            #Get the top-k experts for each input
            topk_val, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # both (B*T, top_k)
        
            #compute output for each selected expert
            expert_outputs = torch.zeros_like(x_flat)  # (B*T, d)
        
        
            for i in range(self.top_k):
                expert_indx_i = topk_indices[:, i]  # (B*T,)
                weight_i = topk_val[:, i].unsqueeze(1)  # (B*T, 1)
            
                for j in range(self.n_experts):
                    mask= (expert_indx_i ==j) # This line shows whick tokens go to this expert
                
                    if mask.any():
                        #expert_input = x_flat[mask]  # token for this expert (num_tokens_for_expert, d)
                        # (num_tokens_for_expert, d)
                        x_j = x_flat[mask]
                        y_j = self.experts[j](x_j)  # (num_tokens_for_expert, d)
                        expert_outputs[mask] += expert_outputs[mask] * y_j  # Weighted sum
        
            output = expert_outputs.view(B, T, d)  # Reshape back to (B, T, d)
            output = self.dropout(output)
            return output


@torch.no_grad()
def evaluate_loss(model, config, train_data, val_data):
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(train_data, val_data, split, config['batch_size'], config['context_size'])
            _, loss = model(xb, yb)
            losses.append(loss.item())

        if split == "train":
            loss_train = np.mean(losses)
        elif split == "val":
            loss_val = np.mean(losses)
    model.train()
    return loss_train, loss_val
