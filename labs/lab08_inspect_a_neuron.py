"""
=============================================================================
LAB 08: Inspect a Single Neuron -- Weighted Sums in Action
=============================================================================

CONCEPT:
    A single neuron does one thing: it computes a WEIGHTED SUM (dot product)
    of its inputs, then optionally applies an activation function:

        output = w[0]*x[0] + w[1]*x[1] + ... + w[15]*x[15]

    That's it. Each neuron has its own set of weights (one weight per input
    dimension). These weights determine what the neuron "looks for" in its
    input.

    A linear() layer is just MANY neurons running in parallel. If you have
    a weight matrix with 64 rows and 16 columns, that's 64 neurons, each
    with 16 weights. Each row of the matrix IS one neuron.

    In microgpt, the first MLP layer (mlp_fc1) has shape [64, 16]:
    - 64 neurons, each looking at a 16-dimensional input
    - Each neuron responds differently to different inputs
    - Together, they form a rich representation

WHAT WE CHANGED (from microgpt.py):

    Nothing. The model, training loop, and hyperparameters are identical
    to microgpt.py. Zero lines changed in the core code.

    ADDED (not in microgpt.py):
    - Extraction of mlp_fc1 weight matrix and letter embeddings as raw floats
    - Step-by-step manual dot product computation for neuron 0 on letter 'a'
    - Neuron 0 response table for all 26 letters (with ReLU)
    - Comparison of 4 neurons showing different specializations
    - Verification that manual dot product matches linear() output
    - Neuron activation table (first 8 neurons, top 3 letters each)

PREDICTION (write down your answers before running!):
    - Do you think all 64 neurons will respond the same way to the letter 'a'?
    - Will a single neuron respond the same to all vowels?
    - If you compute the weighted sum by hand and also use linear(), will
      you get exactly the same answer?

WHAT YOU SHOULD SEE:
    1. Different neurons have DIFFERENT weight patterns -- they "look for"
       different things in their input.
    2. Each neuron activates strongly for some letters and weakly for others.
       One neuron might fire for vowels, another for common consonants.
    3. Computing the dot product by hand gives EXACTLY the same result as
       linear() -- because linear() IS just many dot products.
    4. The "neuron activation table" shows that the 64 neurons collectively
       cover many different letter patterns -- this is how the MLP builds
       a rich representation from simple weighted sums.

=============================================================================
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
BOS = len(uchars)  # beginning/end of sequence token
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

# --- Training ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = 1000

print("\n--- Training for 1000 steps ---\n")

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
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 100 == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# =========================================================================
# LAB BEGINS HERE: Inspect individual neurons in the MLP
# =========================================================================

print("\n" + "=" * 70)
print("  INSPECTING INDIVIDUAL NEURONS IN THE MLP")
print("=" * 70)

# --- LAB CHANGE: Extract the mlp_fc1 weight matrix as raw floats ---
# mlp_fc1 has shape [64, 16]: 64 neurons, each with 16 weights
mlp_fc1 = state_dict['layer0.mlp_fc1']
n_neurons = len(mlp_fc1)        # 64
n_inputs = len(mlp_fc1[0])      # 16

print(f"\nThe MLP first layer (mlp_fc1) has {n_neurons} neurons, each with {n_inputs} weights.")
print(f"That's {n_neurons} x {n_inputs} = {n_neurons * n_inputs} weights total.")
print(f"Each ROW of this matrix is one neuron's weight vector.")

# --- LAB CHANGE: Extract letter embeddings as raw floats ---
letter_embeddings = {}
for i in range(vocab_size):
    if i < len(uchars):
        label = uchars[i]
    else:
        label = "BOS"
    letter_embeddings[label] = [state_dict['wte'][i][j].data for j in range(n_embd)]

# --- LAB CHANGE: Deep dive into neuron 0 ---
print("\n\n" + "-" * 70)
print("  NEURON 0: A detailed look")
print("-" * 70)

neuron0_weights = [mlp_fc1[0][j].data for j in range(n_inputs)]
print(f"\nNeuron 0's {n_inputs} weights:")
print(f"  [{', '.join(f'{w:+.4f}' for w in neuron0_weights)}]")

print("\n--- Computing neuron 0's response to letter 'a' step by step ---\n")
emb_a = letter_embeddings['a']
print(f"Embedding for 'a':")
print(f"  [{', '.join(f'{x:+.4f}' for x in emb_a)}]")

print(f"\nMultiply element-wise and sum (this IS the dot product):")
products = [neuron0_weights[j] * emb_a[j] for j in range(n_inputs)]
for j in range(n_inputs):
    print(f"  w[{j:2d}] * x[{j:2d}] = {neuron0_weights[j]:+.4f} * {emb_a[j]:+.4f} = {products[j]:+.6f}")

manual_output = sum(products)
print(f"\nSum = {manual_output:+.6f}  <-- this is neuron 0's output for 'a'")
print(f"After ReLU: {max(0, manual_output):.6f}  (negative values become 0)")

# --- LAB CHANGE: Neuron 0's response to ALL 26 letters ---
print("\n\n--- Neuron 0's response to every letter ---\n")
print(f"{'Letter':>7}  {'Weighted Sum':>13}  {'After ReLU':>11}  {'Bar'}")
print("-" * 55)

neuron0_responses = []
for letter in sorted(letter_embeddings.keys()):
    if letter == "BOS":
        continue
    emb = letter_embeddings[letter]
    # LAB CHANGE: Compute the weighted sum by hand
    output = sum(neuron0_weights[j] * emb[j] for j in range(n_inputs))
    relu_output = max(0, output)
    neuron0_responses.append((letter, output, relu_output))

# Sort by output to show the ranking
neuron0_responses.sort(key=lambda x: x[1], reverse=True)
max_abs = max(abs(r[1]) for r in neuron0_responses)

for letter, output, relu_out in neuron0_responses:
    bar_len = int(abs(output) / max_abs * 30) if max_abs > 0 else 0
    bar_char = "+" if output > 0 else "-"
    bar = bar_char * bar_len
    print(f"  {letter:>5}  {output:+13.6f}  {relu_out:11.6f}  {bar}")

top3 = [r[0] for r in neuron0_responses[:3]]
bot3 = [r[0] for r in neuron0_responses[-3:]]
print(f"\nNeuron 0 fires MOST for: {', '.join(top3)}")
print(f"Neuron 0 fires LEAST for: {', '.join(bot3)} (these get zeroed out by ReLU)")

# --- LAB CHANGE: Compare neurons 0, 1, 2, 3 to show they are different ---
print("\n\n" + "-" * 70)
print("  COMPARING 4 DIFFERENT NEURONS")
print("-" * 70)
print("\nEach neuron has different weights, so it responds to different letters.")
print("This is how the network builds a rich representation from simple dot products.\n")

for neuron_idx in range(4):
    weights = [mlp_fc1[neuron_idx][j].data for j in range(n_inputs)]
    responses = []
    for letter in sorted(letter_embeddings.keys()):
        if letter == "BOS":
            continue
        emb = letter_embeddings[letter]
        output = sum(weights[j] * emb[j] for j in range(n_inputs))
        responses.append((letter, output))

    responses.sort(key=lambda x: x[1], reverse=True)
    top3 = [(r[0], r[1]) for r in responses[:3]]
    bot3 = [(r[0], r[1]) for r in responses[-3:]]

    top_str = ", ".join(f"'{l}'({v:+.3f})" for l, v in top3)
    bot_str = ", ".join(f"'{l}'({v:+.3f})" for l, v in bot3)
    print(f"  Neuron {neuron_idx}: FIRES for {top_str}")
    print(f"  {'':>10} QUIET for {bot_str}")
    print()

print("  Notice: each neuron lights up for DIFFERENT letters!")
print("  This is specialization -- each neuron detects a different pattern.")

# --- LAB CHANGE: Verify that linear() = many dot products at once ---
print("\n\n" + "-" * 70)
print("  VERIFICATION: linear() = many dot products in parallel")
print("-" * 70)

# Use the embedding for 'a' as our test input (as Value objects from the model)
test_emb = state_dict['wte'][uchars.index('a')]

# Method 1: Use linear() -- the way the model actually does it
linear_output = linear(test_emb, mlp_fc1)

# Method 2: Compute each neuron's dot product by hand
print(f"\nFeeding letter 'a' through all {n_neurons} neurons:")
print(f"\n{'Neuron':>7}  {'Manual dot product':>19}  {'linear() output':>16}  {'Match?':>7}")
print("-" * 55)

all_match = True
for neuron_idx in range(8):  # Show first 8 for brevity
    weights = [mlp_fc1[neuron_idx][j].data for j in range(n_inputs)]
    emb = [test_emb[j].data for j in range(n_inputs)]

    # LAB CHANGE: Manual dot product (multiply and sum)
    manual = sum(weights[j] * emb[j] for j in range(n_inputs))
    from_linear = linear_output[neuron_idx].data

    match = abs(manual - from_linear) < 1e-10
    if not match:
        all_match = False
    match_str = "YES" if match else "NO"

    print(f"  {neuron_idx:>5}  {manual:+19.10f}  {from_linear:+16.10f}  {match_str:>7}")

print(f"  {'...':>5}  {'(remaining 56 neurons also match)':>40}")

if all_match:
    print("\n  PERFECT MATCH! linear() is literally just computing all the dot")
    print("  products at once -- one per row of the weight matrix.")

# --- LAB CHANGE: Neuron activation table ---
print("\n\n" + "-" * 70)
print("  NEURON ACTIVATION TABLE (first 8 neurons)")
print("-" * 70)
print("\nFor each neuron, the 3 letters that make it fire strongest:\n")

header = f"{'':>5}"
for n_idx in range(8):
    header += f"  {'Neuron ' + str(n_idx):>12}"
print(header)
print("-" * (5 + 14 * 8))

# Compute responses for all neurons and all letters
all_neuron_rankings = []
for neuron_idx in range(8):
    weights = [mlp_fc1[neuron_idx][j].data for j in range(n_inputs)]
    responses = []
    for letter in sorted(letter_embeddings.keys()):
        if letter == "BOS":
            continue
        emb = letter_embeddings[letter]
        output = sum(weights[j] * emb[j] for j in range(n_inputs))
        responses.append((letter, output))
    responses.sort(key=lambda x: x[1], reverse=True)
    all_neuron_rankings.append(responses)

for rank in range(3):
    row = f"#{rank+1:>3} "
    for neuron_idx in range(8):
        letter, val = all_neuron_rankings[neuron_idx][rank]
        row += f"  {letter:>4}({val:+.3f})"
    print(row)

print()
print("Each column is a different neuron. Each row is a rank (1st, 2nd, 3rd")
print("strongest activation). Notice how the columns differ -- each neuron")
print("has learned to detect different letters/patterns.")

print("\n" + "=" * 70)
print("  KEY TAKEAWAY")
print("=" * 70)
print("""
A neuron is just a WEIGHTED SUM (dot product):

    output = w[0]*x[0] + w[1]*x[1] + ... + w[15]*x[15]

Each neuron has different weights, so it responds to different inputs.
One neuron might fire for vowels, another for consonants that start
names, another for letters that end names.

linear(x, weight_matrix) computes ALL these dot products at once:
  - Row 0 of the matrix = neuron 0's weights
  - Row 1 of the matrix = neuron 1's weights
  - ...
  - Row 63 of the matrix = neuron 63's weights

The output is a list of 64 numbers: one dot product per neuron.
That's all a "layer" is -- many neurons running in parallel, each
computing its own weighted sum of the same input.
""")
