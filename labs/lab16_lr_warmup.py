"""
LAB 16: Learning Rate Warmup — Learning Rate Schedules Matter
==============================================================

CONCEPT:
The learning rate controls how big each optimization step is. But should
it be the same size throughout training?

Three strategies:

1. LINEAR DECAY (original microgpt):
   lr starts at lr_base and decreases linearly to 0.
   Simple, but the transition from "big steps" to "small steps" is abrupt
   at the beginning — you start with the maximum learning rate when the
   parameters are still random.

2. COSINE DECAY:
   lr follows a smooth cosine curve from lr_base down to 0.
   Smoother than linear — spends more time at moderate learning rates.
   This is what most modern LLMs use (GPT-3, LLaMA, etc).

3. WARMUP + COSINE DECAY:
   lr ramps UP from 0 to lr_base over the first 100 steps, THEN follows
   cosine decay. The idea: at the very start, parameters are random and
   gradients are noisy. Taking huge steps based on noisy gradients can
   send parameters to bad regions. Warmup lets the model "find its
   footing" with small steps before ramping up.

WHAT WE CHANGED (from microgpt.py):

    Line 208 — learning rate schedule is now a function, tested at 3 variants:
    - Original:  lr_t = learning_rate * (1 - step / num_steps)
    - Exp 1:     lr_t = lr_base * (1 - step / num_steps)              # same as original (linear decay)
    - Exp 2:     lr_t = lr_base * 0.5 * (1 + cos(pi * step / N))      # cosine decay
    - Exp 3:     lr_t = warmup ramp for 100 steps, then cosine decay   # warmup + cosine

    That's it. One line changed -- the learning rate computation.
    The model architecture, optimizer, and all other hyperparameters are identical.

    ADDED (not in microgpt.py):
    - lr_linear_decay(), lr_cosine_decay(), lr_warmup_cosine() schedule functions
    - Learning rate preview table at key steps
    - train() wrapper for running 3 experiments
    - Loss and LR comparison tables, generated names comparison

Run time: ~3-5 minutes
PREDICTION (write down your answers before running!):
-----------------------------------------------------
1. Will warmup make a noticeable difference in just 1000 steps?
2. Will cosine decay beat linear decay?
3. Which schedule will have the lowest final loss?
4. Will the generated names look different between schedules?

WHAT YOU SHOULD SEE:
--------------------
- Cosine decay should be smoother than linear decay because it spends
  more time at moderate learning rates rather than dropping linearly.
- Warmup may help early stability — watch the first few loss values.
  With warmup, early losses should decrease more steadily instead of
  bouncing around.
- The differences may be subtle at 1000 steps. In real LLM training
  (millions of steps), warmup is essentially mandatory — without it,
  training often diverges (loss goes to infinity).
- All three should produce reasonable names, but the quality ordering
  should match the final loss ordering.

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

# --- Model ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

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


# LAB CHANGE: Three different learning rate schedule functions
def lr_linear_decay(step, num_steps, lr_base):
    """Original: linear decay from lr_base to 0."""
    return lr_base * (1 - step / num_steps)

def lr_cosine_decay(step, num_steps, lr_base):
    """Cosine decay from lr_base to 0."""
    return lr_base * 0.5 * (1 + math.cos(math.pi * step / num_steps))

def lr_warmup_cosine(step, num_steps, lr_base, warmup_steps=100):
    """Warmup for warmup_steps, then cosine decay."""
    if step < warmup_steps:
        # LAB CHANGE: Linear ramp from 0 to lr_base
        return lr_base * (step / warmup_steps)
    else:
        # LAB CHANGE: Cosine decay over remaining steps
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return lr_base * 0.5 * (1 + math.cos(math.pi * progress))


def train(label, lr_fn, seed=42):
    """Train with a given learning rate schedule."""
    random.seed(seed)
    sd, params = make_model()

    learning_rate = 0.01
    beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)
    num_steps = 1000
    loss_history = []
    lr_history = []

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

        # LAB CHANGE: Use the learning rate schedule function
        lr_t = lr_fn(step, num_steps, learning_rate)

        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        if (step + 1) % 100 == 0:
            loss_history.append((step + 1, loss.data))
            lr_history.append((step + 1, lr_t))
            print(f"    [{label[:12]:>12}] step {step+1:4d} | loss {loss.data:.4f} | lr {lr_t:.6f}")

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

    return loss_history, lr_history, names


# ============================================================
# RUN ALL 3 EXPERIMENTS
# ============================================================
print("=" * 70)
print("LAB 16: LEARNING RATE SCHEDULES MATTER")
print("=" * 70)

# LAB CHANGE: Show what each schedule looks like at key points
print("\nLearning rate at key steps (lr_base=0.01):")
print(f"{'Step':>6} | {'Linear':>10} | {'Cosine':>10} | {'Warmup+Cos':>10}")
print("-" * 48)
for s in [0, 50, 100, 250, 500, 750, 999]:
    l1 = lr_linear_decay(s, 1000, 0.01)
    l2 = lr_cosine_decay(s, 1000, 0.01)
    l3 = lr_warmup_cosine(s, 1000, 0.01)
    print(f"{s:>6} | {l1:>10.6f} | {l2:>10.6f} | {l3:>10.6f}")

configs = [
    ("Linear Decay", lr_linear_decay),
    ("Cosine Decay", lr_cosine_decay),
    ("Warmup+Cosine", lr_warmup_cosine),
]

all_results = []
for label, lr_fn in configs:
    print(f"\n--- Training: {label} ---")
    history, lr_hist, names = train(label, lr_fn)
    all_results.append((label, history, lr_hist, names))

# --- Comparison table ---
print("\n" + "=" * 70)
print("LOSS COMPARISON TABLE")
print("=" * 70)
header = f"{'Step':>6}"
for label, _, _, _ in all_results:
    header += f" | {label:>14}"
print(header)
print("-" * len(header))

for row_idx in range(len(all_results[0][1])):
    step = all_results[0][1][row_idx][0]
    line = f"{step:>6}"
    for _, history, _, _ in all_results:
        line += f" | {history[row_idx][1]:>14.4f}"
    print(line)

# --- Learning rate comparison at selected steps ---
print("\n" + "=" * 70)
print("LEARNING RATE AT EACH CHECKPOINT")
print("=" * 70)
header = f"{'Step':>6}"
for label, _, _, _ in all_results:
    header += f" | {label:>14}"
print(header)
print("-" * len(header))

for row_idx in range(len(all_results[0][2])):
    step = all_results[0][2][row_idx][0]
    line = f"{step:>6}"
    for _, _, lr_hist, _ in all_results:
        line += f" | {lr_hist[row_idx][1]:>14.6f}"
    print(line)

# --- Generated names ---
print("\n" + "=" * 70)
print("GENERATED NAMES COMPARISON")
print("=" * 70)
for label, _, _, names in all_results:
    print(f"\n  {label}:")
    for i, name in enumerate(names):
        print(f"    {i+1}. {name}")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("""
- LINEAR DECAY is the simplest schedule. It starts at full learning rate
  immediately, which can cause instability early when parameters are random.

- COSINE DECAY is smoother — the learning rate decreases slowly at first,
  then faster in the middle, then slowly again near the end. This is the
  most popular schedule in modern LLM training.

- WARMUP prevents early instability by starting with tiny steps. This is
  critical for large models — without warmup, GPT-3 scale training
  typically diverges (loss explodes to infinity).

- For our tiny model, the differences may be subtle. But the principles
  scale: every major LLM (GPT-4, Claude, LLaMA, Gemini) uses warmup +
  cosine (or similar) decay.

- The warmup phase is typically 0.1-1% of total training steps. We used
  100/1000 = 10% here for visibility, which is more than typical.
""")
