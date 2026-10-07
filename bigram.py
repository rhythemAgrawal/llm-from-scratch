import torch
import torch.nn as nn
from torch.nn import functional as F

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
batch_size = 32
block_size = 8
n_embed = 32
learning_rate = 1e-2
max_iters = 3000
eval_interval = 300
eval_iters = 200
# -----------------------------

torch.manual_seed(1337)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_layer = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_layer = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embed = self.token_embedding_layer(idx)
        position_embed = self.position_embedding_layer(torch.arange(T, device=device))
        x = token_embed + position_embed
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
            logits, loss = self(idx)
            x = logits[:, -1, :]
            probs = F.softmax(x, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=1)
        
        return idx

# Initialize the model
model = BigramLanguageModel()
m = model.to(device)

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model.
for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {i}: training set loss {losses['train']:.4f}, validation set loss {losses['validate']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Inference, i.e. generate from the model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(idx, 500)
print(decode(output[0].tolist()))