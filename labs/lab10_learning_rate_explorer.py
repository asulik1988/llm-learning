"""
=============================================================================
LAB 10: Learning Rate Explorer -- Walking Downhill
=============================================================================

CONCEPT:
    Gradient descent is beautifully simple:

        new_value = old_value - learning_rate * gradient

    The GRADIENT tells you which direction is downhill (toward lower loss).
    The LEARNING RATE tells you how big of a step to take.

    Think of it like being blindfolded on a hilly landscape:
    - You feel the slope under your feet (that's the gradient)
    - You take a step downhill (gradient descent)
    - The learning rate is your STEP SIZE

    Too small a step? You'll eventually reach the bottom, but it will take
    forever. Too big a step? You might overshoot the valley entirely and
    end up on an even higher hill!

    This lab uses PLAIN gradient descent only. No Adam, no momentum, no
    adaptive rates. Just the raw, fundamental algorithm:

        p.data -= learning_rate * p.grad

    This is the purest form of learning. Everything else builds on this.

WHAT WE CHANGED (from microgpt.py):

    Lines 207-215 -- optimizer replaced with plain gradient descent:
    - Original (Adam):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    - Changed:   p.data -= lr * p.grad   # plain gradient descent, no Adam

    Line 208 -- learning rate is now a variable looped over 4 values:
    - Original:  lr_t = learning_rate * (1 - step / num_steps)  (= 0.01 with decay)
    - Changed:   learning_rates = [0.0001, 0.005, 0.05, 0.5]   (no decay, constant lr)

    Line 186 -- num_steps reduced:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 500

    That's it. The optimizer is simplified to raw SGD, and the learning rate
    is tested at four values. The model architecture is identical.

    ADDED (not in microgpt.py):
    - make_model(seed) and generate_names() helper functions
    - NaN/overflow detection for diverging experiments
    - Loss comparison table across 4 learning rates
    - Generated names comparison

PREDICTION (write down your answers before running!):
    - With learning rate 0.0001 (tiny), will the model learn much in 500 steps?
    - With learning rate 0.5 (huge), what do you think will happen?
    - Which learning rate will produce the best names?
    - Will ANY of these match Adam's performance?

WHAT YOU SHOULD SEE:
    1. lr=0.0001 (too small): Loss barely decreases. The model takes such
       tiny steps that it hardly moves downhill in 500 steps. Generated
       names will be mostly gibberish.

    2. lr=0.005 (reasonable): Loss steadily decreases. The model finds a
       decent path downhill. Generated names should look somewhat name-like.

    3. lr=0.05 (aggressive): Loss might decrease faster initially but could
       become unstable. The big steps help early on but risk overshooting.

    4. lr=0.5 (way too big): Loss likely explodes or oscillates wildly.
       Each step is so huge that the model jumps over the valley and lands
       somewhere worse. Generated names will be nonsense.

    This directly shows why step size matters: the same gradient, scaled
    differently, can mean the difference between learning and chaos.
=============================================================================
"""

import os
import math
import random

# --- Data Loading ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# We'll load data once and reuse it for all experiments
base_seed = 42
random.seed(base_seed)
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# --- Tokenizer (character-level) ---
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

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

# --- Model setup helpers ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

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

def gpt(token_id, pos_id, keys, values, sd):
    """Forward pass -- takes state_dict as argument so each experiment has its own."""
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

# LAB CHANGE: Function to create a fresh model with a specific random seed
def make_model(seed):
    """Create a fresh model. Same seed = same initial weights."""
    random.seed(seed)
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    all_params = [p for mat in sd.values() for row in mat for p in row]
    return sd, all_params

# LAB CHANGE: Function to generate names from a model
def generate_names(sd, n_samples=5, temperature=0.5):
    """Generate names using the given state_dict."""
    names = []
    for _ in range(n_samples):
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
    return names

# =========================================================================
# LAB BEGINS: Run 4 experiments with different learning rates
# =========================================================================

print("\n" + "=" * 70)
print("  LEARNING RATE EXPLORER: Plain Gradient Descent")
print("=" * 70)
print("""
The update rule is the simplest possible:

    new_weight = old_weight - learning_rate * gradient

No tricks. No Adam. Just: compute the slope, take a step downhill.
The learning rate controls HOW BIG that step is.
""")

# LAB CHANGE: The 4 learning rates to compare
learning_rates = [0.0001, 0.005, 0.05, 0.5]
lr_labels = ["0.0001 (tiny)", "0.005 (reasonable)", "0.05 (aggressive)", "0.5 (huge)"]
num_steps = 500

# Store results: loss history and generated names for each experiment
all_loss_histories = {}
all_generated_names = {}

for exp_idx, (lr, lr_label) in enumerate(zip(learning_rates, lr_labels)):
    print(f"\n{'=' * 60}")
    print(f"  Experiment {exp_idx + 1}/4: learning_rate = {lr_label}")
    print(f"{'=' * 60}")

    # LAB CHANGE: Fresh model with same initial weights for fair comparison
    sd, all_params = make_model(seed=42)

    loss_history = []
    training_failed = False

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

        # Check for NaN or explosion
        if math.isnan(loss.data) or math.isinf(loss.data) or loss.data > 100:
            print(f"  step {step+1:4d}: loss = {loss.data:.4f} -- DIVERGED! Stopping early.")
            loss_history.append((step + 1, loss.data))
            training_failed = True
            break

        loss.backward()

        # LAB CHANGE: PLAIN gradient descent -- no Adam, no momentum
        # Just: step in the opposite direction of the gradient
        for p in all_params:
            p.data -= lr * p.grad   # <-- this is THE update rule
            p.grad = 0              # reset gradient for next step

        if (step + 1) % 100 == 0:
            loss_history.append((step + 1, loss.data))
            print(f"  step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    all_loss_histories[lr] = loss_history

    # Generate names (even if training diverged, to see the effect)
    random.seed(999)  # Same seed for generation so names are comparable
    if not training_failed:
        names = generate_names(sd, n_samples=5)
    else:
        names = generate_names(sd, n_samples=5)
    all_generated_names[lr] = names

    print(f"\n  Generated names (lr={lr}):")
    for i, name in enumerate(names):
        print(f"    {i+1}. {name}")

# =========================================================================
# LAB CHANGE: Comparison table
# =========================================================================

print("\n\n" + "=" * 70)
print("  COMPARISON TABLE: Loss over time for each learning rate")
print("=" * 70)

# Header
header = f"\n{'Step':>6}"
for lr_label in lr_labels:
    header += f"  {lr_label:>20}"
print(header)
print("-" * (6 + 22 * len(lr_labels)))

# We recorded loss at steps 100, 200, 300, 400, 500
steps_to_show = [100, 200, 300, 400, 500]
for step_target in steps_to_show:
    row = f"{step_target:>6}"
    for lr in learning_rates:
        history = all_loss_histories[lr]
        found = False
        for s, l in history:
            if s == step_target:
                if math.isnan(l) or math.isinf(l) or l > 100:
                    row += f"  {'DIVERGED':>20}"
                else:
                    row += f"  {l:>20.4f}"
                found = True
                break
        if not found:
            row += f"  {'DIVERGED':>20}"
    print(row)

# =========================================================================
# LAB CHANGE: Generated names comparison
# =========================================================================

print("\n\n" + "=" * 70)
print("  GENERATED NAMES COMPARISON")
print("=" * 70)

for lr, lr_label in zip(learning_rates, lr_labels):
    names = all_generated_names[lr]
    print(f"\n  lr = {lr_label}:")
    for i, name in enumerate(names):
        print(f"    {i+1}. {name}")

# =========================================================================
# LAB CHANGE: Explain what we saw
# =========================================================================

print("\n\n" + "=" * 70)
print("  WHAT JUST HAPPENED")
print("=" * 70)
print("""
Every experiment used the SAME gradient (the slope of the loss landscape).
The ONLY difference was the step size (learning rate):

  lr = 0.0001 (tiny steps):
      The model barely moved downhill in 500 steps. It would eventually
      learn, but you'd need tens of thousands of steps. Like taking
      baby steps on a mountain -- safe, but painfully slow.

  lr = 0.005 (reasonable):
      Steady, consistent improvement. Each step is big enough to make
      progress but small enough to stay on course. This is the sweet
      spot for plain gradient descent.

  lr = 0.05 (aggressive):
      Faster initial progress, but the large steps might cause the
      loss to bounce around. Like jogging downhill -- faster, but
      you might stumble.

  lr = 0.5 (way too big):
      The steps are so enormous that the model jumps OVER the valley
      and lands somewhere worse. The loss increases instead of
      decreasing. This is like leaping blindly -- you overshoot
      the bottom every time.
""")

print("=" * 70)
print("  THE BLINDFOLDED HILL ANALOGY")
print("=" * 70)
print("""
Imagine you're blindfolded on a hilly landscape, trying to find the
lowest point (the bottom of a valley):

  1. You feel the slope under your feet      --> that's the GRADIENT
  2. You take a step in the downhill direction --> that's GRADIENT DESCENT
  3. How big a step you take                  --> that's the LEARNING RATE

The gradient tells you WHERE to go. The learning rate tells you HOW FAR.

    new_position = old_position - learning_rate * slope

This one line is the entire algorithm. Everything else in deep learning
(Adam, momentum, learning rate schedules) just makes this basic step
work better. But at its core, learning = walking downhill.
""")
