"""
LAB 14: One vs Four Heads — Why Use Multiple Attention Heads?
==============================================================

CONCEPT:
In multi-head attention, we split the embedding dimension across multiple
"heads." Each head independently computes its own attention pattern — its
own set of Q, K, V projections over its slice of the embedding.

With n_embd=16:
- 1 head x 16 dims: One head sees the full 16-dimensional space.
  It can learn ONE complex attention pattern.
- 4 heads x 4 dims: Four heads, each seeing a 4-dimensional slice.
  Each can learn a DIFFERENT pattern (e.g., one head might focus on
  the previous character, another on vowel positions, another on
  name endings, etc.)
- 16 heads x 1 dim: Sixteen heads, each seeing just 1 dimension.
  Lots of heads, but each has so little information to work with
  that it may not be able to compute useful attention patterns.

The key tradeoff:
- More heads = more chances to specialize on different patterns
- Fewer dims per head = less capacity per head to compute useful patterns

WHAT WE CHANGED (from microgpt.py):

    Line 103 — n_head is now a parameter, tested at 3 values:
    - Original:  n_head = 4
    - Exp 1:     n_head = 1,  head_dim = 16  # one wide head
    - Exp 2:     n_head = 4,  head_dim = 4   # same as original
    - Exp 3:     n_head = 16, head_dim = 1   # many narrow heads

    Line 104 — head_dim derived from the experiment's n_head:
    - Original:  head_dim = n_embd // n_head  (= 4)
    - Changed:   head_dim passed as parameter to gpt()

    Lines 157-162 — gpt() takes n_h and h_dim as parameters:
    - Original:  for h in range(n_head):
                     hs = h * head_dim
    - Changed:   for h in range(n_h):
                     hs = h * h_dim

    That's it. Two hyperparameters varied (n_head, head_dim).
    The model architecture and training are otherwise identical.

    ADDED (not in microgpt.py):
    - gpt() accepts n_h and h_dim as arguments
    - make_model() and train() wrappers for running 3 experiments
    - Loss comparison table and generated names comparison

Run time: ~3-4 minutes
PREDICTION (write down your answers before running!):
-----------------------------------------------------
1. Will 1 big head or 4 smaller heads produce lower loss?
2. Will 16 extremely narrow heads (1 dim each) work at all?
3. Which configuration will generate the most realistic names?
4. Is there a sweet spot, or does more heads always = better?

WHAT YOU SHOULD SEE:
--------------------
- 4 heads (default) should work well — each head has enough capacity
  (4 dims) to compute meaningful attention, and there are enough heads
  to specialize.
- 1 head might work okay but is limited to learning a single attention
  pattern — it can't simultaneously focus on "recent character" and
  "vowel pattern" and "name length."
- 16 heads with 1 dim each is problematic: with only 1 dimension, each
  head computes attention based on a single number, which severely limits
  what patterns it can detect.
- The sweet spot for this tiny model is somewhere in the middle.

Note: n_head * head_dim must always equal n_embd (16).

"""

import os
import math
import random

# --- Data Loading ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

# --- Tokenizer ---
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# --- Autograd Engine ---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# --- Neural Network Operations ---
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# --- Fixed hyperparameters ---
n_layer = 1
n_embd = 16
block_size = 16

# --- GPT forward pass (parameterized by n_head/head_dim) ---
def gpt(token_id, pos_id, keys, values, sd, n_h, h_dim):
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_h):  # LAB CHANGE: use configurable n_head
            hs = h * h_dim    # LAB CHANGE: use configurable head_dim
            q_h = q[hs:hs+h_dim]
            k_h = [ki[hs:hs+h_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+h_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(h_dim)) / h_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(h_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, sd['lm_head'])
    return logits


def make_model():
    """Create fresh parameters (same for all configs — attention weights are shared)."""
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params


def train(label, n_h, h_dim, seed=42):
    """Train model with given head configuration."""
    random.seed(seed)
    sd, params = make_model()

    print(f"\n  {label}")
    print(f"  n_head={n_h}, head_dim={h_dim}, total={n_h * h_dim} (must equal n_embd={n_embd})")
    print(f"  Parameters: {len(params)}")

    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)
    num_steps = 1000
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, sd, n_h, h_dim)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        if (step + 1) % 200 == 0:
            loss_history.append((step + 1, loss.data))
            print(f"    step {step+1:4d} | loss {loss.data:.4f}")

    # Generate names
    temperature = 0.5
    names = []
    for _ in range(10):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values, sd, n_h, h_dim)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        names.append(''.join(sample))

    return loss_history, names


# ============================================================
# RUN ALL 3 EXPERIMENTS
# ============================================================
print("=" * 70)
print("LAB 14: WHY USE MULTIPLE ATTENTION HEADS?")
print("=" * 70)

configs = [
    ("1 head x 16 dims",  1, 16),  # LAB CHANGE: single wide head
    ("4 heads x 4 dims",  4,  4),  # LAB CHANGE: default configuration
    ("16 heads x 1 dim", 16,  1),  # LAB CHANGE: many narrow heads
]

all_results = []
for label, n_h, h_dim in configs:
    print(f"\n--- Training: {label} ---")
    history, names = train(label, n_h, h_dim)
    all_results.append((label, history, names))

# --- Comparison table ---
print("\n" + "=" * 70)
print("LOSS COMPARISON TABLE")
print("=" * 70)
header = f"{'Step':>6}"
for label, _, _ in all_results:
    header += f" | {label:>18}"
print(header)
print("-" * len(header))

for row_idx in range(len(all_results[0][1])):
    step = all_results[0][1][row_idx][0]
    line = f"{step:>6}"
    for _, history, _ in all_results:
        line += f" | {history[row_idx][1]:>18.4f}"
    print(line)

# --- Generated names ---
print("\n" + "=" * 70)
print("GENERATED NAMES COMPARISON")
print("=" * 70)
for label, _, names in all_results:
    print(f"\n  {label}:")
    for i, name in enumerate(names):
        print(f"    {i+1:>2}. {name}")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("""
- ONE HEAD can only learn a single attention pattern. It might learn
  "look at the previous character" but then it can't ALSO learn
  "look at the first character of the name" at the same time.

- MULTIPLE HEADS can specialize: one head learns recent context,
  another learns vowel patterns, another learns position-dependent
  patterns. This is why the original transformer paper was called
  "Attention Is All You Need" — multi-head attention is powerful.

- TOO MANY HEADS with too few dimensions per head becomes a problem:
  with head_dim=1, each head computes attention based on a single
  scalar value. The dot product q*k is just one multiplication —
  not enough to compute nuanced similarity between positions.

- This is a common tradeoff in transformer design: GPT-3 uses 96
  heads with 128 dims each. The sweet spot depends on model size.
""")
