"""
LAB 09: Removing Nonlinearity -- Why ReLU Matters
==================================================

CONCEPT:
    The MLP in a transformer has two linear layers with a ReLU in between:

        hidden = ReLU(x @ W1)
        output = hidden @ W2

    Why is ReLU there? Without it, you have:

        output = (x @ W1) @ W2 = x @ (W1 @ W2)

    Two linear transformations in sequence are mathematically equivalent to
    a SINGLE linear transformation. The matrix product W1 @ W2 is just another
    matrix. So without ReLU, you have effectively lost an entire layer --
    the model's capacity is strictly reduced.

    ReLU is what makes the network "deep" rather than just "wide". It introduces
    nonlinearity, allowing the network to learn curved decision boundaries and
    complex patterns that no single linear transformation can capture.

    However, ReLU has a problem: "dead neurons". When a neuron's input is
    negative, ReLU outputs exactly 0, and its gradient is also 0. If a neuron
    gets stuck outputting 0 for all inputs, it can never recover -- it is "dead."

    Leaky ReLU fixes this by using max(0.01*x, x) instead of max(0, x).
    For negative inputs, the output is small but nonzero (0.01*x), so the
    gradient is 0.01 instead of 0 -- dead neurons can still recover.

WHAT WE CHANGED (from microgpt.py):

    Line 57 — added leaky_relu() method to Value class:
    - Original:  (no leaky_relu method exists)
    - Changed:   def leaky_relu(self, alpha=0.01):
                     if self.data > 0: return Value(self.data, (self,), (1.0,))
                     else: return Value(alpha * self.data, (self,), (alpha,))

    Line 174 — activation function is now experiment-dependent (3 experiments):
    - Experiment "relu":       x = [xi.relu() for xi in x]        # same as original
    - Experiment "none":       (line deleted — no activation)      # two linear layers collapse
    - Experiment "leaky_relu": x = [xi.leaky_relu() for xi in x]  # new activation

    Line 186 — num_steps reduced:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 500

    That's it. The core architecture change is line 174: what happens between
    the two MLP linear layers. Everything else is identical.

    ADDED (not in microgpt.py):
    - leaky_relu() method on Value class
    - run_experiment() wrapper to run each activation variant
    - Loss comparison table and generated names comparison across 3 experiments

PREDICTION (write your answers before running!):
    1. How will removing ReLU entirely affect the loss?
    2. Will the model without ReLU still generate recognizable names?
    3. Will leaky ReLU perform better, worse, or the same as standard ReLU?

WHAT YOU SHOULD SEE:
    Three experiments training for 500 steps each:

    1. ReLU (baseline): reasonable loss, decent names
    2. No nonlinearity: likely higher loss. The MLP's two layers collapse into
       one effective linear layer, reducing model capacity. Names will be
       worse quality.
    3. Leaky ReLU: should perform similarly to ReLU, possibly slightly better.
       The 0.01 slope for negative inputs prevents dead neurons while preserving
       most of the nonlinear benefit.

    The loss trajectory tells the story: removing ReLU should clearly hurt
    performance, proving that nonlinearity is essential for depth to matter.

HOW TO RUN:
    python lab09_remove_relu.py
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

    # LAB CHANGE: Add leaky_relu operation
    def leaky_relu(self, alpha=0.01):
        if self.data > 0:
            return Value(self.data, (self,), (1.0,))       # positive: gradient = 1
        else:
            return Value(alpha * self.data, (self,), (alpha,))  # negative: gradient = alpha (small but nonzero!)

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


def run_experiment(experiment_name, activation_mode):
    """
    Run a training experiment with the specified activation function.
    activation_mode: "relu", "none", or "leaky_relu"
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

            # LAB CHANGE: Apply different activations based on experiment
            if activation_mode == "relu":
                x = [xi.relu() for xi in x]              # Original
            elif activation_mode == "none":
                pass                                       # LAB CHANGE: NO activation! Two linear layers collapse into one
            elif activation_mode == "leaky_relu":
                x = [xi.leaky_relu() for xi in x]         # LAB CHANGE: leaky ReLU -- dead neurons can recover

            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state_dict['lm_head'])
        return logits

    # Training
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_adam = [0.0] * len(params)
    v_adam = [0.0] * len(params)

    num_steps = 500
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
        for i, p in enumerate(params):
            m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
            v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_adam[i] / (1 - beta1 ** (step + 1))
            v_hat = v_adam[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        # LAB CHANGE: Record loss at key milestones
        if step + 1 in [1, 100, 200, 500]:
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
# LAB CHANGE: Run 3 experiments with different activations
# ===========================================================================

experiments = [
    ("1. ReLU (baseline)",    "relu"),
    ("2. No activation",      "none"),
    ("3. Leaky ReLU",         "leaky_relu"),
]

all_results = {}
all_names = {}

for name, mode in experiments:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    loss_history, generated = run_experiment(name, mode)
    all_results[name] = loss_history
    all_names[name] = generated

    for step, lv in sorted(loss_history.items()):
        print(f"  step {step:>4d} | loss {lv:.4f}")
    print(f"  Generated names:")
    for i, n in enumerate(generated):
        print(f"    sample {i+1}: {n}")

# ===========================================================================
# LAB CHANGE: Comparison table
# ===========================================================================

print(f"\n{'='*60}")
print("LOSS COMPARISON TABLE")
print(f"{'='*60}")

steps_to_show = [1, 100, 200, 500]
header = f"{'Experiment':<25}" + "".join(f" | {'Step '+str(s):>10}" for s in steps_to_show)
print(f"\n{header}")
print(f"{'-'*25}" + "".join(f"-+-{'-'*10}" for _ in steps_to_show))

for name, _ in experiments:
    row = f"{name:<25}"
    for s in steps_to_show:
        row += f" | {all_results[name].get(s, float('nan')):>10.4f}"
    print(row)

print(f"\n{'='*60}")
print("GENERATED NAMES COMPARISON")
print(f"{'='*60}")

for name, _ in experiments:
    print(f"\n  {name}:")
    for i, n in enumerate(all_names[name]):
        print(f"    {i+1}. {n}")

print("""
KEY TAKEAWAYS:
- Without activation functions, stacked linear layers collapse into ONE layer:
    (x @ W1) @ W2 = x @ (W1 * W2)
  This is just another matrix multiply -- no extra capacity from the second layer!
- ReLU (max(0, x)) introduces nonlinearity, making each layer genuinely add capacity
- But ReLU has "dead neurons": once a neuron outputs 0 for all inputs, it stays dead
  because the gradient through ReLU at 0 is exactly 0
- Leaky ReLU (max(0.01*x, x)) fixes this: negative inputs get a small nonzero gradient
  (0.01), so neurons can recover from being "dead"
- This is why nonlinear activation functions are essential to deep learning:
  without them, depth is an illusion
""")
