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
    pass

class MultiHeadAttention(nn.Module):
    pass

class FeedForward(nn.Module):
    pass

class Block(nn.Module):
    pass

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

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

            logits, loss = self(idx)

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

model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

for step in range(10000):

    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())

context = torch.zeros(
    (1, 1),
    dtype=torch.long
)

generated = model.generate(
    context,
    max_new_tokens=200
)

print("\nGenerated Text:\n")

print(
    decode(
        generated[0].tolist()
    )
)
# Delete these temporarily

class Head(nn.Module):
    pass

class MultiHeadAttention(nn.Module):
    pass

class FeedForward(nn.Module):
    pass

class Block(nn.Module):
    pass
