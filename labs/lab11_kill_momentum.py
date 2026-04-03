"""
LAB 11: Kill Momentum — What Do Adam's Two Tricks Actually Do?
==============================================================

CONCEPT:
The Adam optimizer has two "tricks" beyond plain SGD:

1. MOMENTUM (beta1): Keeps a running average of past gradients.
   Think of it like a ball rolling downhill — it builds up speed in
   consistent directions and smooths out noise. When beta1=0.85,
   85% of the previous momentum is kept, plus 15% of the new gradient.

2. ADAPTIVE SCALING (beta2): Keeps a running average of squared gradients.
   This means parameters that get large gradients frequently get smaller
   updates, and parameters that get small gradients get relatively larger
   updates. It adapts the learning rate per-parameter.

When BOTH are disabled (beta1=0, beta2=0), you basically get plain SGD
(with bias correction, which doesn't do much when betas are 0).

WHAT WE CHANGED (from microgpt.py):

    Line 182 — beta1 and beta2 are now configurable per experiment (4 experiments):
    - Original:  learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    - Exp 1:     beta1=0.85, beta2=0.99   # same as original (full Adam)
    - Exp 2:     beta1=0.0,  beta2=0.99   # no momentum
    - Exp 3:     beta1=0.85, beta2=0.0    # no adaptive scaling
    - Exp 4:     beta1=0.0,  beta2=0.0    # plain SGD

    Lines 210-214 — bias correction handles beta=0 gracefully:
    - Original:  m_hat = m[i] / (1 - beta1 ** (step + 1))
    - Changed:   b1_corr = 1 - b1 ** (step + 1) if b1 > 0 else 1.0
                 m_hat = m_buf[i] / b1_corr
                 (same pattern for b2_corr / v_hat)

    That's it. The model architecture and forward pass are identical.
    Only the optimizer's beta1/beta2 values change between experiments.

    ADDED (not in microgpt.py):
    - make_model() and train() wrapper functions for running multiple experiments
    - gpt() takes state_dict as parameter (sd) instead of using global
    - Loss comparison table and generated names comparison across 4 configs

Run time: ~3-5 minutes
PREDICTION (write down your answers before running!):
-----------------------------------------------------
1. Which trick matters more for final loss: momentum or adaptive scaling?
2. Will disabling momentum make training noisier or smoother?
3. Will any configuration completely fail to learn?
4. How different will the generated names look across configs?

WHAT YOU SHOULD SEE:
--------------------
- Full Adam should train smoothly and reach the lowest loss.
- Without momentum (beta1=0), training is noisier because each step
  reacts only to the current gradient, not the trend. But it still learns
  because adaptive scaling helps.
- Without adaptive scaling (beta2=0), training can be more unstable because
  all parameters get the same effective learning rate regardless of how
  "active" they are. This often hurts more than losing momentum.
- Without either (plain SGD), training is the noisiest and may not converge
  as well, but it can still learn something — SGD is a perfectly valid
  optimizer, just less efficient.
- The generated names will be best from full Adam and progressively worse
  as you remove each trick.

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

# --- Tokenizer (character-level) ---
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

# --- Model Parameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# --- GPT forward pass ---
def gpt(token_id, pos_id, keys, values, sd):
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


def make_model():
    """Create a fresh set of parameters."""
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


def train(label, b1, b2, seed=42):
    """Train with given beta1/beta2 values."""
    random.seed(seed)
    sd, params = make_model()

    learning_rate, eps_adam = 0.01, 1e-8
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
            logits = gpt(token_id, pos_id, keys, values, sd)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)
        loss.backward()

        # LAB CHANGE: Adam with configurable beta1 and beta2
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = b1 * m_buf[i] + (1 - b1) * p.grad
            v_buf[i] = b2 * v_buf[i] + (1 - b2) * p.grad ** 2
            # Bias correction (handles b1=0 or b2=0 gracefully)
            b1_corr = 1 - b1 ** (step + 1) if b1 > 0 else 1.0
            b2_corr = 1 - b2 ** (step + 1) if b2 > 0 else 1.0
            m_hat = m_buf[i] / b1_corr
            v_hat = v_buf[i] / b2_corr
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        if (step + 1) % 200 == 0:
            loss_history.append((step + 1, loss.data))
            print(f"  [{label}] step {step+1:4d} | loss {loss.data:.4f}")

    # Generate names
    temperature = 0.5
    names = []
    for _ in range(5):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values, sd)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        names.append(''.join(sample))

    return loss_history, names


# ============================================================
# RUN ALL 4 EXPERIMENTS
# ============================================================
print("=" * 70)
print("LAB 11: WHAT DO ADAM'S TWO TRICKS DO?")
print("=" * 70)

configs = [
    ("Full Adam (b1=0.85, b2=0.99)", 0.85, 0.99),   # LAB CHANGE: baseline
    ("No momentum (b1=0.0, b2=0.99)", 0.0, 0.99),    # LAB CHANGE: disable momentum
    ("No adaptive (b1=0.85, b2=0.0)", 0.85, 0.0),    # LAB CHANGE: disable adaptive scaling
    ("Plain SGD (b1=0.0, b2=0.0)", 0.0, 0.0),         # LAB CHANGE: both disabled
]

all_results = []

for label, b1, b2 in configs:
    print(f"\n--- Training: {label} ---")
    history, names = train(label, b1, b2)
    all_results.append((label, history, names))

# --- Print comparison table ---
print("\n" + "=" * 70)
print("LOSS COMPARISON TABLE")
print("=" * 70)
header = f"{'Step':>6}"
for label, _, _ in all_results:
    short = label.split("(")[0].strip()
    header += f" | {short:>16}"
print(header)
print("-" * len(header))

for row_idx in range(len(all_results[0][1])):
    step = all_results[0][1][row_idx][0]
    line = f"{step:>6}"
    for _, history, _ in all_results:
        line += f" | {history[row_idx][1]:>16.4f}"
    print(line)

# --- Print generated names ---
print("\n" + "=" * 70)
print("GENERATED NAMES COMPARISON")
print("=" * 70)
for label, _, names in all_results:
    print(f"\n  {label}:")
    for i, name in enumerate(names):
        print(f"    {i+1}. {name}")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("""
- MOMENTUM (beta1) smooths the optimization trajectory. Without it,
  training is noisier but can still work.

- ADAPTIVE SCALING (beta2) adjusts per-parameter learning rates.
  This is often the more important trick — without it, some parameters
  get updates that are too large or too small.

- Full Adam combines both for the best of both worlds.

- Even plain SGD (both disabled) learns something — optimizers don't
  create intelligence, they just make training more efficient.
""")
