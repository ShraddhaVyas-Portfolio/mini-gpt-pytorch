import torch
import torch.nn as nn
import torch.nn.functional as F

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

block_size = 8
batch_size = 4
n_embd = 32
head_size = 16
def get_batch():

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    return x, y
class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )
      

      
    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)

        wei = F.softmax(
            wei,
            dim=-1
        )

        v = self.value(x)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):

        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

    def forward(self, x):

        return torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        


class FeedForward(nn.Module):
    def __init__(self, n_embd):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )

    def forward(self, x):

        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward(
            n_embd
        )


        self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(
        self.ln1(x)
        )

        x = x + self.ffwd(
        self.ln2(x)
        )

        return x
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.token_embedding_table = nn.Embedding(
    vocab_size,
    n_embd
)
        self.position_embedding_table = nn.Embedding(
    block_size,
    n_embd
)

        self.blocks = nn.Sequential(

        Block(n_embd, 2),

        Block(n_embd, 2),

        Block(n_embd, 2)

)

        self.lm_head = nn.Linear(
         n_embd,
         vocab_size
    )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T)
     )

        x = token_emb + pos_emb

        x = self.blocks(x)

        logits = self.lm_head(x)
        if targets is None:
            loss = None

        else:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

        
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(
            logits,
            dim=-1
            )

            idx_next = torch.multinomial(
            probs,
            num_samples=1
            )

            idx = torch.cat(
            (idx, idx_next),
            dim=1
            )

        return idx

    
model = GPTLanguageModel(vocab_size)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

for step in range(5000):

    xb, yb = get_batch()

    logits, loss = model(
        xb,
        yb
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()

if step % 100 == 0:

    print(
        step,
        loss.item()
        )

context = torch.zeros(
    (1, 1),
    dtype=torch.long
)

generated = model.generate(
    context,
    max_new_tokens=300
)

print("\nGenerated Text:\n")

print(
    decode(
        generated[0].tolist()
    )
)


generated = model.generate(
    context,
    max_new_tokens=300
)

print("\nGenerated Text:\n")

print(
    decode(
        generated[0].tolist()
    )
)

