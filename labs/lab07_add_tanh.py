"""
LAB 07: Adding tanh() to the Autograd Engine -- The Chain Rule in Action
========================================================================

CONCEPT:
    The beauty of an autograd engine is that you can add ANY differentiable
    function as a new operation, as long as you provide two things:
        1. The forward computation: what does the function output?
        2. The local gradient: what is the derivative of the output w.r.t. the input?

    The chain rule handles everything else. When backward() runs, it multiplies
    each local gradient by the upstream gradient and passes it along.

    In this lab, we add tanh() to the Value class.

    tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Its derivative has a beautiful form:
        d/dx tanh(x) = 1 - tanh(x)^2

    This means the local gradient only depends on the OUTPUT of tanh, not the
    input. This is computationally convenient.

    We then replace ReLU with tanh in the MLP and verify everything still works
    by running gradient checking on the new operation.

    Historically, tanh was the dominant activation function before ReLU. It has
    the advantage of outputting values in [-1, 1] (centered around zero), but
    the disadvantage of "vanishing gradients" -- when |x| is large, tanh
    saturates and the gradient approaches zero, slowing learning.

WHAT WE CHANGED (from microgpt.py):

    Line 57 — added tanh() method to Value class:
    - Original:  (no tanh method exists)
    - Changed:   def tanh(self):
                     t = math.tanh(self.data)
                     return Value(t, (self,), (1 - t**2,))

    Line 174 — activation function swapped:
    - Original:  x = [xi.relu() for xi in x]
    - Changed:   x = [xi.tanh() for xi in x]

    Line 186 — num_steps reduced:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 500

    That's it. Three lines changed. One new method, one activation swap,
    one step count adjustment.

    ADDED (not in microgpt.py):
    - Gradient checking section (same approach as lab06) to verify tanh gradients
    - compute_loss_for_doc() helper function

PREDICTION (write your answers before running!):
    1. Will tanh train faster or slower than ReLU for this task?
    2. Will the final loss be better or worse?
    3. Will gradient checking pass for our tanh implementation?

WHAT YOU SHOULD SEE:
    First, gradient checking runs on 5 parameters to verify that the tanh
    local gradient (1 - tanh^2) is correctly implemented. All should pass.

    Then you will see training for 500 steps. The loss should decrease,
    though possibly at a different rate than ReLU. tanh can sometimes train
    slightly differently because:
    - Its output range [-1, 1] vs ReLU's [0, inf] changes the scale of activations
    - It does not "kill" neurons the way ReLU does (ReLU outputs exactly 0 for
      negative inputs), but it can suffer from vanishing gradients for large inputs

    At the end, 5 generated names are shown.

HOW TO RUN:
    python lab07_add_tanh.py
"""

import os
import math
import random

random.seed(42)

# --- Data Loading ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

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

    # LAB CHANGE: Add tanh() operation with correct local gradient
    def tanh(self):
        t = math.tanh(self.data)           # forward: compute tanh(x)
        return Value(t, (self,), (1 - t**2,))  # local gradient: 1 - tanh(x)^2

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

# --- Model Parameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

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
print(f"num params: {len(params)}")

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

# --- The GPT Model (forward pass) ---
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
        x = [xi.tanh() for xi in x]  # LAB CHANGE: tanh instead of relu (was: xi.relu())
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ===========================================================================
# LAB CHANGE: First, run gradient checking to verify tanh gradients are correct
# ===========================================================================

def compute_loss_for_doc(doc):
    """Run one forward pass on a single document and return the loss Value."""
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
    return loss

print(f"\n{'='*60}")
print("GRADIENT CHECKING: Verifying tanh gradients are correct")
print(f"{'='*60}")

doc = docs[0]
loss = compute_loss_for_doc(doc)
loss.backward()

epsilon = 1e-5
random.seed(999)
check_indices = random.sample(range(len(params)), 5)

print(f"\n{'Param #':>8} | {'Backprop Grad':>14} | {'Numerical Grad':>14} | {'Relative Error':>14} | {'Match?':>6}")
print(f"{'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*6}")

all_match = True
for idx in check_indices:
    p = params[idx]
    analytical_grad = p.grad

    for pp in params:
        pp.grad = 0

    original_data = p.data
    p.data = original_data + epsilon
    loss_plus = compute_loss_for_doc(doc)
    numerical_grad = (loss_plus.data - loss.data) / epsilon
    p.data = original_data

    if abs(analytical_grad) + abs(numerical_grad) > 0:
        rel_error = abs(analytical_grad - numerical_grad) / (abs(analytical_grad) + abs(numerical_grad) + 1e-15)
    else:
        rel_error = 0.0

    match = rel_error < 1e-3
    if not match:
        all_match = False
    print(f"{idx:>8} | {analytical_grad:>14.8f} | {numerical_grad:>14.8f} | {rel_error:>14.2e} | {'  OK' if match else ' FAIL':>6}")

if all_match:
    print("\nAll gradient checks PASSED -- tanh local gradient (1 - tanh^2) is correct!")
else:
    print("\nWARNING: Some gradient checks FAILED!")

# Reset all gradients for training
for p in params:
    p.grad = 0

# ===========================================================================
# LAB CHANGE: Train with tanh for 500 steps
# ===========================================================================

print(f"\n{'='*60}")
print("TRAINING WITH TANH ACTIVATION (500 steps)")
print(f"{'='*60}\n")

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_adam = [0.0] * len(params)
v_adam = [0.0] * len(params)

num_steps = 500

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

    if step == 0 or (step + 1) % 100 == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# --- Inference ---
temperature = 0.5
print(f"\n{'='*60}")
print("GENERATED NAMES (with tanh activation)")
print(f"{'='*60}")

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

    print(f"  sample {sample_idx+1}: {''.join(sample)}")

print("""
KEY TAKEAWAYS:
- Adding a new operation to autograd requires ONLY the local gradient
- tanh local gradient: d/dx tanh(x) = 1 - tanh(x)^2
- Gradient checking confirmed our implementation is correct
- The chain rule composes all the local gradients automatically
- tanh was the dominant activation before ReLU, but ReLU is simpler
  and avoids the vanishing gradient problem for large |x|
- You can add ANY differentiable function the same way: forward + local gradient
""")
