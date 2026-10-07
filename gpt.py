import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# extract all the characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character encoding and decoding
itos = {i:ch for i,ch in enumerate(chars)}
stoi = {ch:i for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# encode the whole data
data = torch.tensor(encode(text), dtype=torch.long)

# Split dataset into training and validation
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# Hyper-parameters
# -----------------------------
batch_size = 64
block_size = 256
n_embed = 384
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_layers = 6
num_head = 6
dropout = 0.2
# -----------------------------

torch.manual_seed(1337)
device = 'mps'

def get_batch(split):
    data = train_data if split == 'train' else validation_data
    idxs = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[idx:idx+block_size] for idx in idxs])
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in idxs])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model):
    splits = ['train', 'validate']
    average_loss = {}
    model.eval()

    for split in splits:
        losses = torch.zeros(eval_iters)

        for iter in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[iter] = loss.item()
        
        average_loss[split] = losses.mean()
    
    model.train()
    return average_loss

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = torch.nn.Linear(n_embed, head_size, bias=False)
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        head_size = self.key.out_features
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        affinities = q @ k.transpose(-2, -1) * head_size**-0.5 # Shouldn't this be the head_size instead of C??
        affinities = affinities.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, -1)
        affinities = self.dropout(affinities)
        out = affinities @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()

        head_size = n_embed//num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.frwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.frwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_layer = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_layer = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embed = self.token_embedding_layer(idx)
        position_embed = self.position_embedding_layer(torch.arange(T, device=device))
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_block_size = idx[:, -block_size:]
            logits, loss = self(idx_block_size)
            x = logits[:, -1, :]
            probs = F.softmax(x, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=1)
        
        return idx

# Initialize the model
model = BigramLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters()), 'M parameters')

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_time = time.time()

# Train the model.
for i in range(max_iters):
    if i % eval_interval == 0 or i == max_iters-1:
        losses = estimate_loss(model)
        print(f"Step {i}: training set loss {losses['train']:.4f}, validation set loss {losses['validate']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Time taken to train the model: ", time.time()-start_time)

# Inference, i.e. generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(idx, 5000)
print(decode(output[0].tolist()))