"""
LAB 10: SGD vs Adam -- Why Optimizers Matter
=============================================

CONCEPT:
    The simplest optimizer is "Stochastic Gradient Descent" (SGD):

        p.data -= learning_rate * p.grad

    This is the most direct interpretation of "move downhill." But it has problems:
    - It is noisy: each training example pulls parameters in different directions,
      causing the loss to bounce around rather than smoothly decrease
    - It treats all parameters equally: a parameter that rarely gets large gradients
      is updated at the same rate as one that frequently does

    "SGD with Momentum" adds a running average of past gradients:

        velocity = beta * velocity + (1 - beta) * gradient
        p.data -= learning_rate * velocity

    This smooths out noise: if gradients consistently point the same direction,
    momentum amplifies the signal. If they bounce around, momentum dampens them.
    Think of it like a heavy ball rolling downhill -- it builds speed in consistent
    directions and resists random jitter.

    "Adam" (Adaptive Moment Estimation) adds a second trick on top of momentum:
    per-parameter adaptive learning rates. It tracks:
        - m: running mean of gradients (like momentum)
        - v: running mean of squared gradients (captures gradient "scale")

    Parameters with consistently large gradients get a smaller effective learning
    rate (to prevent overshooting), while parameters with small gradients get a
    larger effective learning rate (to speed up their learning).

    Adam = Momentum + Adaptive per-parameter learning rates + Bias correction

WHAT WE CHANGED (from microgpt.py):

    Lines 207-215 — optimizer step replaced with 3 variants:
    - Original (Adam):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

    - Exp "sgd":           p.data -= lr_t * p.grad
    - Exp "sgd_momentum":  velocity = 0.85 * velocity + 0.15 * p.grad
                           p.data -= lr_t * velocity
    - Exp "adam":           (same as original)

    That's it. The model architecture and forward pass are identical.
    Only the optimizer update rule changes between experiments.

    ADDED (not in microgpt.py):
    - run_experiment() with optimizer_type parameter ("adam", "sgd", "sgd_momentum")
    - Loss comparison table across all 3 optimizers at every 100 steps
    - Generated names comparison and final loss summary

PREDICTION (write your answers before running!):
    1. Which optimizer will reach the lowest loss: SGD, SGD+Momentum, or Adam?
    2. How large will the gap be between SGD and Adam after 1000 steps?
    3. Will plain SGD even produce recognizable names?

WHAT YOU SHOULD SEE:
    A comparison table of loss at every 100 steps for all three optimizers.

    - Plain SGD: loss decreases slowly and noisily. It may still be quite high
      after 1000 steps. Generated names may look random.
    - SGD + Momentum: significantly better than plain SGD. Momentum smooths out
      the noise, allowing more consistent progress.
    - Adam: the best optimizer here. The adaptive learning rates help because
      different parameters (embeddings vs attention vs MLP) naturally have
      different gradient scales.

    The generated names from each experiment illustrate the quality difference.

HOW TO RUN:
    python lab10_sgd_vs_adam.py
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

# --- Model Architecture Constants ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

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


def run_experiment(experiment_name, optimizer_type):
    """
    Run a training experiment with the specified optimizer.
    optimizer_type: "adam", "sgd", or "sgd_momentum"
    """
    random.seed(42)  # LAB CHANGE: same seed for fair comparison
    random.shuffle(docs)

    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    params = [p for mat in state_dict.values() for row in mat for p in row]

    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict['wte'][token_id]
        pos_emb = state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)

            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])
            v = linear(x, state_dict[f'layer{li}.attn_wv'])
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

            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state_dict['lm_head'])
        return logits

    # --- Optimizer setup ---
    learning_rate = 0.01
    num_steps = 1000

    # LAB CHANGE: Different optimizer state depending on type
    if optimizer_type == "adam":
        beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
        m_adam = [0.0] * len(params)
        v_adam = [0.0] * len(params)
    elif optimizer_type == "sgd_momentum":
        momentum_beta = 0.85  # LAB CHANGE: same beta as Adam's beta1 for fair comparison
        velocity = [0.0] * len(params)

    loss_history = {}

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)

        # LAB CHANGE: Apply the selected optimizer
        if optimizer_type == "adam":
            # Adam: momentum + adaptive per-parameter learning rates + bias correction
            for i, p in enumerate(params):
                m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
                v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
                m_hat = m_adam[i] / (1 - beta1 ** (step + 1))
                v_hat = v_adam[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
                p.grad = 0

        elif optimizer_type == "sgd":
            # LAB CHANGE: Plain SGD -- the simplest possible optimizer
            for i, p in enumerate(params):
                p.data -= lr_t * p.grad  # LAB CHANGE: just follow the gradient. That's it.
                p.grad = 0

        elif optimizer_type == "sgd_momentum":
            # LAB CHANGE: SGD + Momentum -- smooth out the noise with a running average
            for i, p in enumerate(params):
                velocity[i] = momentum_beta * velocity[i] + (1 - momentum_beta) * p.grad  # LAB CHANGE: running average of gradients
                p.data -= lr_t * velocity[i]  # LAB CHANGE: use smoothed gradient instead of raw gradient
                p.grad = 0

        # Record loss at milestones
        if (step + 1) % 100 == 0 or step == 0:
            loss_history[step + 1] = loss.data

    # Generate names
    temperature = 0.5
    generated = []
    for sample_idx in range(5):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []

        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])

        generated.append(''.join(sample))

    return loss_history, generated


# ===========================================================================
# LAB CHANGE: Run 3 experiments with different optimizers
# ===========================================================================

experiments = [
    ("1. Adam",           "adam"),
    ("2. Plain SGD",      "sgd"),
    ("3. SGD + Momentum", "sgd_momentum"),
]

all_results = {}
all_names = {}

for name, opt_type in experiments:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    loss_history, generated = run_experiment(name, opt_type)
    all_results[name] = loss_history
    all_names[name] = generated

    for step, lv in sorted(loss_history.items()):
        print(f"  step {step:>5d} | loss {lv:.4f}")
    print(f"  Generated names:")
    for i, n in enumerate(generated):
        print(f"    sample {i+1}: {n}")

# ===========================================================================
# LAB CHANGE: Comparison table
# ===========================================================================

print(f"\n{'='*60}")
print("LOSS COMPARISON TABLE")
print(f"{'='*60}")

steps_to_show = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
header = f"{'Step':>6}"
for name, _ in experiments:
    header += f" | {name:>18}"
print(f"\n{header}")
print(f"{'-'*6}" + "".join(f"-+-{'-'*18}" for _ in experiments))

for s in steps_to_show:
    row = f"{s:>6}"
    for name, _ in experiments:
        val = all_results[name].get(s, None)
        if val is not None:
            row += f" | {val:>18.4f}"
        else:
            row += f" | {'---':>18}"
    print(row)

print(f"\n{'='*60}")
print("GENERATED NAMES COMPARISON")
print(f"{'='*60}")

for name, _ in experiments:
    print(f"\n  {name}:")
    for i, n in enumerate(all_names[name]):
        print(f"    {i+1}. {n}")

# Final analysis
print(f"\n{'='*60}")
print("FINAL LOSS COMPARISON")
print(f"{'='*60}")
for name, _ in experiments:
    final = all_results[name].get(1000, all_results[name].get(max(all_results[name].keys()), 0))
    print(f"  {name:<25} final loss: {final:.4f}")

print("""
KEY TAKEAWAYS:
- Plain SGD: p -= lr * grad
  Simple but noisy. Each training example sends the gradient in a slightly
  different direction, so progress is jerky and slow.

- SGD + Momentum: velocity = beta * velocity + (1-beta) * grad; p -= lr * velocity
  Like a heavy ball rolling downhill. Consistent gradient directions get amplified,
  random jitter gets dampened. Much smoother convergence.

- Adam: momentum + per-parameter adaptive learning rates
  On top of momentum, Adam tracks how "active" each parameter's gradient is.
  Parameters with consistently large gradients get smaller learning rates (prevents
  overshooting). Parameters with small gradients get larger learning rates (speeds
  them up). This is why Adam is the default optimizer for transformers.

  The key insight: different parameters (embeddings vs attention vs MLP) naturally
  have very different gradient magnitudes. A single learning rate is a bad fit.
  Adam's per-parameter adaptation handles this automatically.
""")
