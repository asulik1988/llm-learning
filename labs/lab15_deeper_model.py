"""
LAB 15: Deeper Model — Does Adding Layers Help?
=================================================

CONCEPT:
Each transformer layer consists of:
1. Attention block: lets positions look at each other
2. MLP block: processes information at each position

With 1 layer, the model gets ONE round of "look at context, then process."
With 2 layers, the model gets TWO rounds — the second layer can build on
patterns the first layer found. With 4 layers, even more rounds.

Think of it like reading a sentence:
- 1 layer: "What letters are near me?"
- 2 layers: "What letters are near me, and what patterns do THOSE form?"
- 4 layers: Even deeper reasoning about patterns of patterns.

But more layers also means:
- More parameters (more things to learn)
- Slower training (more computation per step)
- Risk of overfitting (too many parameters for the data we have)
- Vanishing/exploding gradients (though residual connections help)

For our tiny model and small training run, there are likely diminishing
returns. Real LLMs use 32-96 layers, but they also have billions of
parameters and train on trillions of tokens.

WHAT WE CHANGED (from microgpt.py):

    Line 100 — n_layer is now a parameter, tested at 3 values:
    - Original:  n_layer = 1
    - Exp 1:     n_layer = 1   # same as original
    - Exp 2:     n_layer = 2   # double depth
    - Exp 3:     n_layer = 4   # quadruple depth

    Lines 110-116 — layer creation loop uses variable count:
    - Original:  for i in range(n_layer):   # always 1
    - Changed:   for i in range(num_layers):  # 1, 2, or 4

    Line 137 — gpt() takes num_layers as a parameter:
    - Original:  def gpt(token_id, pos_id, keys, values):
    - Changed:   def gpt(token_id, pos_id, keys, values, sd, num_layers):

    That's it. One hyperparameter varied (n_layer). The per-layer
    architecture is identical to microgpt.py.

    ADDED (not in microgpt.py):
    - gpt() and make_model() accept num_layers parameter
    - train() wrapper for running 3 experiments
    - Parameter count breakdown (embedding vs. per-layer)
    - Loss comparison table and generated names comparison

Run time: ~3-5 minutes
PREDICTION (write down your answers before running!):
-----------------------------------------------------
1. How many more parameters does each extra layer add?
2. Will 4 layers train to significantly lower loss than 1 layer?
3. Will extra layers help with name quality in just 1000 steps?
4. Could extra layers actually HURT with so little training?

WHAT YOU SHOULD SEE:
--------------------
- Parameter count grows linearly with layers (each layer adds the same
  number of attention + MLP weights).
- More layers may produce slightly lower loss, but with only 1000 steps
  the deeper models may not have enough training to leverage extra capacity.
- The 4-layer model has 4x the layer parameters, but only sees the same
  1000 training examples — it might underfit (not enough training) even
  though it has more capacity.
- Generated names may not be dramatically different across configs because
  1000 steps is very little training for the deeper models.

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
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# --- GPT forward pass (parameterized by n_layer) ---
def gpt(token_id, pos_id, keys, values, sd, num_layers):
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(num_layers):  # LAB CHANGE: variable number of layers
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
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


def make_model(num_layers):
    """Create fresh parameters with given layer count."""
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(num_layers):  # LAB CHANGE: variable layer count
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params


def train(label, num_layers, seed=42):
    """Train model with given layer count."""
    random.seed(seed)
    sd, params = make_model(num_layers)

    print(f"\n  {label}")
    print(f"  Layers: {num_layers}")
    print(f"  Parameters: {len(params)}")  # LAB CHANGE: show parameter count

    # Calculate parameter breakdown
    embedding_params = vocab_size * n_embd + block_size * n_embd + vocab_size * n_embd
    per_layer_params = (4 * n_embd * n_embd) + (4 * n_embd * n_embd + n_embd * 4 * n_embd)
    print(f"    Embedding params: {embedding_params}")
    print(f"    Per-layer params: {per_layer_params} x {num_layers} = {per_layer_params * num_layers}")

    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)
    num_steps = 1000
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(num_layers)], [[] for _ in range(num_layers)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, sd, num_layers)
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
        keys, values = [[] for _ in range(num_layers)], [[] for _ in range(num_layers)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values, sd, num_layers)
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
print("LAB 15: DOES ADDING LAYERS HELP?")
print("=" * 70)

configs = [
    ("1 layer (default)", 1),  # LAB CHANGE: baseline
    ("2 layers",          2),  # LAB CHANGE: double depth
    ("4 layers",          4),  # LAB CHANGE: quadruple depth
]

all_results = []
for label, num_layers in configs:
    print(f"\n{'=' * 40}")
    print(f"Training: {label}")
    print(f"{'=' * 40}")
    history, names = train(label, num_layers)
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
- Each layer adds the same number of parameters (attention weights +
  MLP weights). The parameter count scales linearly with depth.

- More layers = more "rounds of thinking." Each layer can build on
  what the previous layer discovered. Layer 1 might learn bigram
  patterns, layer 2 might learn trigram patterns built on those, etc.

- But more layers need more training to be useful. With only 1000 steps,
  the deeper models may not fully utilize their extra capacity.

- In practice, depth is one of the most important scaling dimensions.
  GPT-2 used 48 layers, GPT-3 used 96 layers. But they also trained
  on vastly more data with much wider embeddings.

- For this tiny model on this small dataset, 1 layer may actually be
  close to optimal — adding layers gives diminishing returns when the
  data and training budget are limited.
""")
