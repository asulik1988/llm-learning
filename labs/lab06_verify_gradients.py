"""
LAB 06: Gradient Checking -- Verifying Backpropagation is Correct
=================================================================

CONCEPT:
    Backpropagation uses the chain rule to compute gradients analytically.
    But how do we know our backward() implementation is actually correct?

    The answer is "gradient checking" (also called "numerical gradient verification").
    The idea is simple: the derivative of a function f at point x is defined as:

        df/dx = lim(epsilon->0) [f(x + epsilon) - f(x)] / epsilon

    So we can NUMERICALLY estimate any gradient by:
        1. Recording the current loss
        2. Nudging one parameter by a tiny epsilon (e.g., 1e-5)
        3. Recomputing the loss
        4. Computing (new_loss - old_loss) / epsilon

    If backprop is correct, the analytical gradient (.grad) should match the
    numerical estimate to within floating-point tolerance.

    This technique was crucial in the early days of deep learning -- before
    autograd frameworks existed, researchers hand-derived gradients and used
    gradient checking to verify them. It is still used today when implementing
    custom operations.

WHAT WE CHANGED (from microgpt.py):

    Nothing. The model, architecture, and all parameters are identical
    to microgpt.py. Zero lines changed in the core code.
    Training is not run at all -- only a single forward+backward pass is used.

    ADDED (not in microgpt.py):
    - compute_loss_for_doc() helper to run a single forward pass
    - Gradient checking loop: for 5 random parameters, nudges each by
      epsilon=1e-5, recomputes loss, and compares numerical gradient
      (loss_plus - loss) / epsilon to the analytical gradient from .grad
    - Relative error computation and pass/fail reporting

PREDICTION (write your answers before running!):
    1. Will the analytical and numerical gradients match exactly, or only approximately?
    2. How close do you expect them to be? (1e-2? 1e-5? 1e-10?)
    3. Could there be cases where they don't match well even if backprop is correct?

WHAT YOU SHOULD SEE:
    You will see 5 randomly chosen parameters, each showing:
    - The analytical gradient from backprop (.grad value)
    - The numerical gradient from the finite-difference method
    - The relative error between them

    The gradients should match to about 5-7 decimal places (relative error < 1e-4).
    They won't match EXACTLY because:
    - Numerical gradients use a finite epsilon, not a true infinitesimal limit
    - Floating-point arithmetic has limited precision
    - But they should be close enough to confirm backprop is correct

    If you ever see relative errors > 1e-2, something is wrong with the
    gradient computation. This is your smoke test for autograd correctness.

HOW TO RUN:
    python lab06_verify_gradients.py
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

# --- Tokenizer (character-level) ---
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)  # beginning/end of sequence token
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
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ===========================================================================
# LAB CHANGE: Gradient checking -- verify backprop matches numerical gradients
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

# Use just one short document for gradient checking
doc = docs[0]
print(f"\n{'='*60}")
print(f"GRADIENT CHECKING")
print(f"{'='*60}")
print(f"Using document: '{doc}'")

# Step 1: Forward + backward to get analytical gradients
print(f"\nStep 1: Running forward + backward pass...")
loss = compute_loss_for_doc(doc)
loss.backward()
print(f"Loss: {loss.data:.6f}")

# Step 2: Pick 5 random parameters to check
num_checks = 5  # LAB CHANGE: number of parameters to verify
epsilon = 1e-5  # LAB CHANGE: size of the numerical nudge

# Choose random parameter indices (skip any with zero grad to get interesting ones)
random.seed(123)  # LAB CHANGE: deterministic selection for reproducibility
check_indices = random.sample(range(len(params)), num_checks)

print(f"\nStep 2: Checking {num_checks} random parameters (epsilon={epsilon})")
print(f"\n{'Param #':>8} | {'Backprop Grad':>14} | {'Numerical Grad':>14} | {'Relative Error':>14} | {'Match?':>6}")
print(f"{'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*6}")

all_match = True
for idx in check_indices:
    p = params[idx]
    analytical_grad = p.grad  # LAB CHANGE: save the gradient from backprop

    # Step 3: Numerically estimate the gradient
    # Zero all gradients first
    for pp in params:
        pp.grad = 0

    # Nudge this parameter up by epsilon
    original_data = p.data
    p.data = original_data + epsilon  # LAB CHANGE: nudge parameter
    loss_plus = compute_loss_for_doc(doc)  # LAB CHANGE: recompute loss

    # Compute numerical gradient: (loss_new - loss_old) / epsilon
    numerical_grad = (loss_plus.data - loss.data) / epsilon  # LAB CHANGE: finite difference

    # Restore original parameter value
    p.data = original_data  # LAB CHANGE: undo the nudge

    # Compute relative error
    if abs(analytical_grad) + abs(numerical_grad) > 0:
        rel_error = abs(analytical_grad - numerical_grad) / (abs(analytical_grad) + abs(numerical_grad) + 1e-15)
    else:
        rel_error = 0.0

    match = rel_error < 1e-3  # LAB CHANGE: tolerance for matching
    if not match:
        all_match = False

    print(f"{idx:>8} | {analytical_grad:>14.8f} | {numerical_grad:>14.8f} | {rel_error:>14.2e} | {'  OK' if match else ' FAIL':>6}")

# Summary
print(f"\n{'='*60}")
if all_match:
    print("ALL GRADIENTS MATCH! Backpropagation is computing correct gradients.")
else:
    print("WARNING: Some gradients did not match! There may be a bug in backward().")
print(f"{'='*60}")

print("""
KEY TAKEAWAYS:
- Gradient checking is the "unit test" for autograd implementations
- Analytical gradients (from backprop) should match numerical gradients
- Relative error < 1e-4 is generally considered a pass
- This technique works for ANY differentiable computation graph
- In practice, you run this once to verify, then rely on backprop
  (because numerical gradients are ~1000x slower -- one forward pass per parameter!)
""")
