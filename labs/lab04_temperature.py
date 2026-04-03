"""
=============================================================================
LAB 04: Temperature Controls Randomness
=============================================================================

CONCEPT:
    After the model computes logits (raw scores for each possible next token),
    we apply softmax to turn them into probabilities. But before softmax, we
    can divide the logits by a "temperature" value:

        probs = softmax(logits / temperature)

    Temperature controls the SHAPE of the probability distribution:

    - Low temperature (0.01): Makes the distribution VERY peaked. The highest-
      scoring token gets nearly 100% probability. Output becomes deterministic
      and repetitive -- the model always picks its "best guess."

    - Temperature = 1.0: Uses the raw probabilities as-is. This is what the
      model actually learned.

    - High temperature (3.0): Makes the distribution VERY flat. All tokens get
      similar probabilities, even bad ones. Output becomes random and chaotic.

    Think of it like a confidence dial:
    - Low temp = "I'm VERY sure, always pick my top choice"
    - High temp = "I'm NOT sure, let's try anything"

WHAT WE CHANGED (from microgpt.py):

    Line 220 -- temperature is now a variable looped over 5 values:
    - Original:  temperature = 0.5
    - Changed:   temperatures = [0.01, 0.3, 0.5, 1.0, 3.0]  # tested in a loop

    That's it. One parameter explored at five values. The model, training,
    and all hyperparameters are identical to microgpt.py.

    ADDED (not in microgpt.py):
    - Loop over 5 temperatures during inference
    - Side-by-side results table
    - Per-temperature analysis (avg name length, unique count, pattern notes)
    - Probability distribution visualization at each temperature for BOS->first letter

PREDICTION (write down your answers before running!):
    - At temperature 0.01, will all 10 names be the same?
    - At temperature 3.0, will the names look like real names at all?
    - Which temperature do you think produces the best-looking names?

WHAT YOU SHOULD SEE:
    temp=0.01: Almost identical names, very short, very "safe." The model
               picks the single most likely letter every time. You might see
               the exact same name 10 times.

    temp=0.3:  Good names but conservative. Popular patterns dominate.
               Less diversity than you'd want.

    temp=0.5:  A nice balance. Names look plausible and varied.
               (This is the default in microgpt.)

    temp=1.0:  More creative/unusual names. Some are great, some are weird.
               This is the "true" distribution the model learned.

    temp=3.0:  Near-random garbage. Letters are chosen almost uniformly.
               Names look like someone mashed the keyboard.

    The probability distribution table at the end really drives this home:
    at low temperature, one letter has ~100% probability. At high temperature,
    every letter has roughly equal probability (~3-4% each).
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

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.seed(42)
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


# --- The GPT Model ---
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


# --- Training Loop ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000

print(f"\nTraining for {num_steps} steps...")
print("(We train ONCE, then sample at 5 different temperatures)\n")

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

    if (step + 1) % 200 == 0 or step == 0:
        print(f"  step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

print(f"\n  Training complete! Final loss: {loss.data:.4f}")


# ======================================================================
# LAB CHANGE: Generate names at different temperatures
# ======================================================================

temperatures = [0.01, 0.3, 0.5, 1.0, 3.0]  # LAB CHANGE
num_samples = 10

print(f"\n{'='*70}")
print("  TEMPERATURE EXPERIMENT")
print("="*70)
print(f"\n  Generating {num_samples} names at each of {len(temperatures)} temperatures.\n")

# LAB CHANGE: collect all results for side-by-side display
results = {}
for temp in temperatures:
    results[temp] = []
    random.seed(123)  # LAB CHANGE: same seed per temperature for fair comparison

    for sample_idx in range(num_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []

        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temp for l in logits])  # LAB CHANGE: use variable temperature
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])

        results[temp].append(''.join(sample) if sample else "<empty>")

# LAB CHANGE: print results as a side-by-side table
header = "  {:>4s}".format("#")
for temp in temperatures:
    header += f"  |  T={temp:<4}"
print(header)
print("  " + "-" * (len(header) - 2))

for i in range(num_samples):
    row = f"  {i+1:>4d}"
    for temp in temperatures:
        name = results[temp][i]
        row += f"  |  {name:<12}"
    print(row)

# LAB CHANGE: print observations about each temperature
print(f"\n{'='*70}")
print("  ANALYSIS: How temperature changes the output")
print("="*70)

for temp in temperatures:
    names = results[temp]
    avg_len = sum(len(n) for n in names) / len(names)
    unique = len(set(names))
    print(f"\n  Temperature = {temp}:")
    print(f"    Average name length: {avg_len:.1f}")
    print(f"    Unique names: {unique}/{num_samples}")
    if temp <= 0.1:
        print(f"    Pattern: VERY DETERMINISTIC -- model picks its top choice every time")  # LAB CHANGE
    elif temp <= 0.5:
        print(f"    Pattern: Conservative -- mostly common, safe-sounding names")  # LAB CHANGE
    elif temp <= 1.0:
        print(f"    Pattern: Balanced -- varied names, some creative, some conventional")  # LAB CHANGE
    else:
        print(f"    Pattern: CHAOTIC -- near-random letter choices, names look like noise")  # LAB CHANGE


# ======================================================================
# LAB CHANGE: Show probability distribution at different temperatures
# ======================================================================

print(f"\n\n{'='*70}")
print("  PROBABILITY DISTRIBUTION: First token prediction")
print("  (What letter should a name start with, given BOS?)")
print("="*70)

# Get logits for the first position (BOS -> first letter)
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
logits = gpt(BOS, 0, keys, values)

for temp in temperatures:
    probs = softmax([l / temp for l in logits])
    prob_data = [(uchars[i] if i < len(uchars) else 'BOS', probs[i].data) for i in range(vocab_size)]
    prob_data.sort(key=lambda x: -x[1])

    print(f"\n  Temperature = {temp}:")
    print(f"  {'Letter':>6}  {'Prob':>7}  Bar")
    print(f"  {'------':>6}  {'-----':>7}  ---")

    # Show top 10 and their probability bars
    for letter, prob in prob_data[:10]:
        bar_len = int(prob * 60)
        bar = '#' * bar_len
        print(f"  {letter:>6}  {prob:>6.1%}  {bar}")

    # Show how concentrated the distribution is
    top1_prob = prob_data[0][1]
    top5_prob = sum(p for _, p in prob_data[:5])
    print(f"  ... (remaining {vocab_size - 10} tokens share {1 - sum(p for _, p in prob_data[:10]):.1%})")
    print(f"  Top-1 probability: {top1_prob:.1%}  |  Top-5 cumulative: {top5_prob:.1%}")


print(f"\n\n{'='*70}")
print("  KEY TAKEAWAY")
print("="*70)
print("""
  Temperature is a simple but powerful knob:

  - At T=0.01, the distribution is a SPIKE: one token has ~100%% probability.
    The output is deterministic and repetitive. Useful when you want the
    single "best" answer (e.g., code completion, factual questions).

  - At T=1.0, you see the TRUE learned distribution. This is what the model
    actually believes. Good for understanding what the model learned.

  - At T=3.0, the distribution is nearly UNIFORM: every token is equally
    likely. The output is random noise. The model's knowledge is being
    drowned out by randomness.

  In practice, T=0.3-0.8 is the sweet spot for most tasks. ChatGPT and
  similar systems use temperature (along with top-p sampling) to control
  the creativity vs. reliability trade-off.

  MATH: dividing logits by T before softmax is equivalent to raising
  each probability to the power (1/T) and renormalizing. Low T sharpens,
  high T flattens.
""")
