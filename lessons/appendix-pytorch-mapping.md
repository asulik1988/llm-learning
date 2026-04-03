# Appendix: microgpt to PyTorch

When you move from microgpt to real implementations, the concepts are identical but the names and tools change. This table maps what you learned to what you'll see in PyTorch.

## Core Mapping

| microgpt | PyTorch | What it does |
|----------|---------|-------------|
| `Value` class with `.data` and `.grad` | `torch.Tensor` with `requires_grad=True` | Stores numbers + tracks gradients automatically |
| `Value.backward()` | `loss.backward()` | Runs backpropagation (chain rule) through the computation graph |
| `linear(x, w)` (dot product per row) | `torch.nn.Linear(in, out)` or `x @ w.T` | Matrix multiplication — many neurons in parallel |
| `softmax(logits)` | `torch.nn.functional.softmax(logits, dim=-1)` | Converts logits to probabilities |
| `relu()` method | `torch.nn.functional.relu()` or `torch.nn.ReLU()` | `max(0, x)` nonlinearity |
| `rmsnorm(x)` | `torch.nn.RMSNorm(dim)` | Normalizes vector to stable scale |
| `state_dict` (dict of matrices) | `model.state_dict()` or `model.parameters()` | All the model's learnable weights |
| `matrix(nout, nin)` | `torch.nn.Parameter(torch.randn(nout, nin) * 0.08)` | Random weight initialization |
| `params = [p for ...]` | `model.parameters()` | Flat list of all trainable parameters |
| Manual Adam loop (lines 208-215) | `torch.optim.Adam(model.parameters(), lr=0.01)` | The optimizer, as a single object |
| `p.grad = 0` | `optimizer.zero_grad()` | Reset gradients before next step |
| Adam update + `loss.backward()` | `loss.backward()` then `optimizer.step()` | The full training update |

## Architecture Mapping

| microgpt concept | PyTorch equivalent | Notes |
|-----------------|-------------------|-------|
| `state_dict['wte']` (embedding table) | `torch.nn.Embedding(vocab_size, n_embd)` | Lookup table for token vectors |
| `state_dict['wpe']` (position table) | `torch.nn.Embedding(block_size, n_embd)` | Lookup table for position vectors |
| The `gpt()` function | A `torch.nn.Module` subclass with a `forward()` method | The model's forward pass |
| KV cache (`keys`, `values` lists) | Managed by inference frameworks (vLLM, HF generate) | Stored K/V for fast autoregressive generation |
| `tokens = [BOS] + [uchars.index(ch) ...] + [BOS]` | Tokenizer (`tiktoken`, `sentencepiece`, HF `AutoTokenizer`) | Text to token IDs |
| The training `for step` loop | `for batch in dataloader:` + optimizer step | Same loop, but batched |
| `loss = -probs[target].log()` | `torch.nn.functional.cross_entropy(logits, targets)` | Same math, fused for efficiency |

## The Key Insight

PyTorch automates what microgpt does by hand:

```
microgpt                          PyTorch
─────────────────────────         ─────────────────────────
# Forward pass                    # Forward pass
logits = gpt(token, pos, ...)     logits = model(input_ids)

# Loss                            # Loss
loss = -probs[target].log()       loss = F.cross_entropy(logits, targets)

# Backward                        # Backward
loss.backward()                   loss.backward()

# Update                          # Update
for i, p in enumerate(params):    optimizer.step()
    # ... Adam math ...           optimizer.zero_grad()
    p.grad = 0
```

The left side is ~15 lines. The right side is ~4 lines. But they do exactly the same thing. PyTorch's `Tensor` is microgpt's `Value`, scaled up to run on GPUs with optimized C++/CUDA kernels instead of pure Python loops.

When you see `nn.Linear`, think "a matrix of neurons doing `linear(x, w)`."
When you see `loss.backward()`, think "the chain rule walking the computation graph."
When you see `optimizer.step()`, think "nudge every dial to reduce the loss."

You already understand what these functions do. PyTorch just makes them fast.
