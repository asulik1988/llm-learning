"""
=============================================================================
LAB 02: Visualize What the Model Learns About Letters
=============================================================================

CONCEPT:
    Every letter in our model gets an "embedding" -- a list of numbers that
    represents that letter. During training, the model adjusts these numbers
    so that letters used in similar contexts end up with similar embeddings.

    Normally embeddings have many dimensions (16 in microgpt), which makes
    them impossible to visualize directly. In this lab, we set n_embd=2,
    so each letter's embedding is just an (x, y) coordinate that we can
    plot on a 2D scatter plot.

    The trade-off: with only 2 dimensions the model has very little capacity,
    so it won't generate great names. But we get to SEE what it learned.

WHAT WE CHANGED (from microgpt.py):

    Line 101 -- n_embd shrunk to 2:
    - Original:  n_embd = 16
    - Changed:   n_embd = 2   # so each embedding is an (x, y) coordinate we can plot

    Line 103 -- n_head reduced to match:
    - Original:  n_head = 4
    - Changed:   n_head = 1   # need head_dim = n_embd/n_head >= 1

    That's it. Two lines changed. Everything else is identical.

    ADDED (not in microgpt.py):
    - Embedding extraction and printing after training
    - Scatter plot generation (embeddings_2d.png) using matplotlib
    - Vowel/consonant color coding and BOS token visualization
    - Observations section explaining what to look for in the plot

PREDICTION (write down your answers before running!):
    - Do you think vowels (a, e, i, o, u) will cluster together?
    - Will common consonants (s, t, n, r) be near each other?
    - Will rarely-used letters (q, x, z) be far from everything else?

WHAT YOU SHOULD SEE:
    After training, the scatter plot (saved to labs/embeddings_2d.png) should
    show some structure -- letters that appear in similar positions in names
    should end up near each other. Vowels often cluster because they appear
    in similar contexts (between consonants). Common consonants that start
    names may cluster together.

    The clustering won't be perfect (2D is very limited), but you should see
    that the model is NOT placing letters randomly. It has learned something
    about which letters behave similarly.

    This is the foundation of ALL modern NLP: words (or letters) that appear
    in similar contexts get similar representations. This is how models
    "understand" language -- not through definitions, but through patterns
    of usage.
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
n_embd = 2     # LAB CHANGE: reduced from 16 to 2 so we can plot embeddings!
block_size = 16
n_head = 1     # LAB CHANGE: reduced from 4 (need head_dim = n_embd/n_head >= 1)
head_dim = n_embd // n_head

print(f"\n*** LAB CHANGE: n_embd = {n_embd} (normally 16) ***")
print(f"*** LAB CHANGE: n_head = {n_head} (normally 4) ***")
print(f"*** Each letter's embedding is just 2 numbers: an (x, y) coordinate ***\n")

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


# --- Training Loop ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000

print(f"\nTraining for {num_steps} steps with n_embd=2...")
print("(This will be slower than normal because 2D embeddings are very limited)\n")

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

    if (step + 1) % 100 == 0 or step == 0:
        print(f"  step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")


# ======================================================================
# LAB CHANGE: Extract and visualize the learned embeddings
# ======================================================================

print("\n" + "="*70)
print("  LEARNED LETTER EMBEDDINGS (2D)")
print("="*70)

vowels = set('aeiou')

# Print the raw embeddings
print(f"\n  {'Letter':>6}  {'X':>8}  {'Y':>8}  {'Type':>10}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}")

embed_data = []
for idx, ch in enumerate(uchars):
    x_val = state_dict['wte'][idx][0].data
    y_val = state_dict['wte'][idx][1].data
    letter_type = "VOWEL" if ch in vowels else "consonant"
    print(f"  {ch:>6}  {x_val:>8.4f}  {y_val:>8.4f}  {letter_type:>10}")
    embed_data.append((ch, x_val, y_val, ch in vowels))

# Also show BOS token
bos_x = state_dict['wte'][BOS][0].data
bos_y = state_dict['wte'][BOS][1].data
print(f"  {'BOS':>6}  {bos_x:>8.4f}  {bos_y:>8.4f}  {'special':>10}")

# --- Create the scatter plot ---
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot vowels and consonants in different colors
for ch, x, y, is_vowel in embed_data:
    color = '#e74c3c' if is_vowel else '#3498db'  # red for vowels, blue for consonants
    marker = 'o' if is_vowel else 's'  # circle for vowels, square for consonants
    size = 120 if is_vowel else 80
    ax.scatter(x, y, c=color, s=size, marker=marker, zorder=3, edgecolors='black', linewidth=0.5)
    ax.annotate(ch, (x, y), textcoords="offset points", xytext=(8, 8),
                fontsize=14, fontweight='bold', color=color)

# Plot BOS token
ax.scatter(bos_x, bos_y, c='#2ecc71', s=150, marker='*', zorder=3, edgecolors='black', linewidth=0.5)
ax.annotate('BOS', (bos_x, bos_y), textcoords="offset points", xytext=(8, 8),
            fontsize=10, fontweight='bold', color='#2ecc71')

ax.set_title('Learned Letter Embeddings (n_embd=2)\nRed circles = vowels, Blue squares = consonants', fontsize=14)
ax.set_xlabel('Embedding dimension 0', fontsize=12)
ax.set_ylabel('Embedding dimension 1', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Vowels (a,e,i,o,u)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=10, label='Consonants'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71', markersize=12, label='BOS token'),
]
ax.legend(handles=legend_elements, loc='best', fontsize=11)

plt.tight_layout()

# Save to labs directory
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(script_dir, 'embeddings_2d.png')
plt.savefig(plot_path, dpi=150)
print(f"\n  Plot saved to: {plot_path}")

# --- Generate some names to see how well this tiny model does ---
print(f"\n{'='*70}")
print("  GENERATED NAMES (with n_embd=2)")
print("="*70)
print("  (Don't expect great results -- 2D embeddings are very limited!)\n")

temperature = 0.5
for sample_idx in range(10):
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

    print(f"  {sample_idx+1:2d}. {''.join(sample)}")

# --- Observations ---  # LAB CHANGE
print(f"\n{'='*70}")
print("  OBSERVATIONS")
print("="*70)
print("""
  Look at the scatter plot (embeddings_2d.png). Some things to notice:

  1. VOWELS vs CONSONANTS: Do the red circles (vowels) tend to cluster
     in one region? If so, the model learned that vowels appear in
     similar contexts (typically between consonants in names).

  2. COMMON vs RARE LETTERS: Letters like 'a', 'e', 'n' appear in many
     names. Letters like 'q', 'x', 'z' appear rarely. Do common letters
     end up in a different region than rare ones?

  3. THE BOS TOKEN: The BOS (beginning/end of sequence) token has a
     special role -- it appears before the first letter and after the
     last letter of every name. Is it far from the regular letters?

  4. NEARBY LETTERS: Look for pairs of letters that ended up close
     together. Can you explain why? (Hint: think about which letters
     appear in similar positions in names.)

  With only 2 dimensions, the model can't capture much. Real models
  use 768+ dimensions, allowing them to encode many different
  relationships simultaneously. But the principle is the same:
  CONTEXT DETERMINES MEANING.
""")
