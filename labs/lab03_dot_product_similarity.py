"""
=============================================================================
LAB 03: The Dot Product -- Measuring Similarity Between Letters
=============================================================================

CONCEPT:
    The dot product is the fundamental operation in neural networks. Given two
    vectors (lists of numbers), you multiply them element-wise and sum the
    results:

        dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]

    A HIGH dot product means the vectors point in the same direction -- they
    are SIMILAR. A dot product near ZERO means the vectors are unrelated. A
    NEGATIVE dot product means they point in opposite directions.

    In this lab, we train microgpt so its letter embeddings learn real
    structure, then we compute the dot product between every pair of letter
    embeddings. Letters that appear in similar contexts (like vowels, which
    all appear between consonants) should have HIGH dot products with each
    other.

    We also compute cosine similarity, which is the dot product divided by
    the magnitudes of both vectors. This gives a fairer comparison because
    it ignores how "big" each vector is and focuses purely on direction.
    Cosine similarity ranges from -1 (opposite) to +1 (identical direction).

WHAT WE CHANGED (from microgpt.py):

    Nothing. The model, training loop, and hyperparameters are identical
    to microgpt.py. Zero lines changed in the core code.

    ADDED (not in microgpt.py):
    - dot_product(), magnitude(), cosine_similarity() helper functions
    - Step-by-step dot product example ('a' dot 'e')
    - All pairwise cosine similarity computation (top 10, bottom 10)
    - Vowel vs. consonant average similarity analysis
    - Per-vowel nearest neighbors
    - BOS token similarity analysis

PREDICTION (write down your answers before running!):
    - Will vowels (a, e, i, o, u) have higher dot products with each other
      than with consonants?
    - Which pair of letters do you think will be MOST similar?
    - Will the average vowel-vowel similarity be higher than the average
      consonant-consonant similarity?
    - What do you think the dot product of a random pair will be? (hint: the
      embeddings are initialized near zero with std=0.08)

WHAT YOU SHOULD SEE:
    After training, you should see that:
    1. The top most-similar pairs tend to be letters that appear in similar
       positions in names (e.g., vowels with vowels, common ending consonants
       with each other).
    2. The least similar pairs tend to be letters with very different roles
       (e.g., a common vowel vs. a rare consonant).
    3. The average vowel-vowel cosine similarity is noticeably higher than
       the average vowel-consonant similarity.

    This is the dot product in action: it MEASURES how similar two vectors
    are. This exact same operation is what attention uses to decide "which
    other tokens should I pay attention to?" -- it computes the dot product
    between query and key vectors.
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

# --- Training Loop (Adam optimizer) ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000

print("\n--- Training for 1000 steps (so embeddings learn real structure) ---\n")

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
# LAB BEGINS HERE: Exploring dot product similarity between letter embeddings
# =========================================================================

print("\n" + "=" * 70)
print("  DOT PRODUCT SIMILARITY BETWEEN LETTER EMBEDDINGS")
print("=" * 70)

# --- LAB CHANGE: Extract the raw embedding data for all 27 tokens ---
# Each embedding is a list of 16 Value objects. We extract the .data floats.
embeddings = {}
labels = []
for i in range(vocab_size):
    if i < len(uchars):
        label = uchars[i]
    else:
        label = "BOS"
    labels.append(label)
    embeddings[label] = [state_dict['wte'][i][j].data for j in range(n_embd)]

# --- LAB CHANGE: Define dot product and cosine similarity functions ---
def dot_product(a, b):
    """Multiply element-wise and sum. This is THE fundamental operation."""
    return sum(ai * bi for ai, bi in zip(a, b))

def magnitude(a):
    """Length of a vector: sqrt(sum of squares)."""
    return math.sqrt(sum(ai * ai for ai in a))

def cosine_similarity(a, b):
    """Dot product normalized by magnitudes. Ranges from -1 to +1."""
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)

# --- LAB CHANGE: Show how dot product works with a concrete example ---
print("\n--- Step-by-step dot product example: 'a' dot 'e' ---\n")
emb_a = embeddings['a']
emb_e = embeddings['e']
print(f"Embedding for 'a': [{', '.join(f'{x:+.3f}' for x in emb_a)}]")
print(f"Embedding for 'e': [{', '.join(f'{x:+.3f}' for x in emb_e)}]")
print()
print("Element-wise multiply:")
products = [ai * ei for ai, ei in zip(emb_a, emb_e)]
for j in range(n_embd):
    print(f"  dim {j:2d}: {emb_a[j]:+.3f} * {emb_e[j]:+.3f} = {products[j]:+.4f}")
print(f"\nSum of all products = {sum(products):.4f}  <-- this is the dot product!")
print(f"(Verify) dot_product(a, e) = {dot_product(emb_a, emb_e):.4f}")
print(f"Cosine similarity(a, e) = {cosine_similarity(emb_a, emb_e):.4f}")

# --- LAB CHANGE: Compute all pairwise cosine similarities ---
# We use cosine similarity for ranking because it normalizes for magnitude.
# (A letter with a large embedding would dominate raw dot products.)
letter_labels = [l for l in labels if l != "BOS"]  # just the 26 letters

print("\n\n--- All pairwise cosine similarities (26 letters) ---\n")

all_pairs = []
for i, l1 in enumerate(letter_labels):
    for j, l2 in enumerate(letter_labels):
        if i < j:
            cs = cosine_similarity(embeddings[l1], embeddings[l2])
            dp = dot_product(embeddings[l1], embeddings[l2])
            all_pairs.append((l1, l2, cs, dp))

# Sort by cosine similarity
all_pairs.sort(key=lambda x: x[2], reverse=True)

print("TOP 10 MOST SIMILAR letter pairs (highest cosine similarity):")
print("-" * 55)
print(f"{'Pair':>8}  {'Cosine Sim':>12}  {'Dot Product':>12}")
print("-" * 55)
for l1, l2, cs, dp in all_pairs[:10]:
    print(f"  {l1} - {l2}      {cs:+.4f}        {dp:+.4f}")

print()
print("TOP 10 LEAST SIMILAR letter pairs (lowest cosine similarity):")
print("-" * 55)
print(f"{'Pair':>8}  {'Cosine Sim':>12}  {'Dot Product':>12}")
print("-" * 55)
for l1, l2, cs, dp in all_pairs[-10:]:
    print(f"  {l1} - {l2}      {cs:+.4f}        {dp:+.4f}")

# --- LAB CHANGE: Vowel vs consonant analysis ---
vowels = set('aeiou')
consonants = set(letter_labels) - vowels

print("\n\n--- Vowel vs. Consonant similarity comparison ---\n")

vowel_vowel = []
vowel_consonant = []
consonant_consonant = []

for l1, l2, cs, dp in all_pairs:
    l1_is_vowel = l1 in vowels
    l2_is_vowel = l2 in vowels
    if l1_is_vowel and l2_is_vowel:
        vowel_vowel.append(cs)
    elif l1_is_vowel or l2_is_vowel:
        vowel_consonant.append(cs)
    else:
        consonant_consonant.append(cs)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

print(f"Average cosine similarity between two VOWELS:      {avg(vowel_vowel):+.4f}  ({len(vowel_vowel)} pairs)")
print(f"Average cosine similarity VOWEL-CONSONANT:         {avg(vowel_consonant):+.4f}  ({len(vowel_consonant)} pairs)")
print(f"Average cosine similarity between two CONSONANTS:  {avg(consonant_consonant):+.4f}  ({len(consonant_consonant)} pairs)")

print()
if avg(vowel_vowel) > avg(vowel_consonant):
    print(">>> Vowels are MORE similar to each other than to consonants!")
    print("    The model learned that vowels play a similar ROLE in names.")
else:
    print(">>> Interesting -- vowels are not clearly more similar to each other.")
    print("    With only 1000 training steps, the model may need more time to")
    print("    separate vowels from consonants cleanly.")

# --- LAB CHANGE: Show specific vowel-vowel pairs ---
print("\n\n--- All vowel-vowel similarities ---\n")
print(f"{'Pair':>8}  {'Cosine Sim':>12}")
print("-" * 25)
for l1, l2, cs, dp in all_pairs:
    if l1 in vowels and l2 in vowels:
        print(f"  {l1} - {l2}      {cs:+.4f}")

# --- LAB CHANGE: Show which letter is most similar to each vowel ---
print("\n\n--- For each vowel, its 3 most similar letters ---\n")
for v_letter in sorted(vowels):
    sims = []
    for other in letter_labels:
        if other != v_letter:
            cs = cosine_similarity(embeddings[v_letter], embeddings[other])
            sims.append((other, cs))
    sims.sort(key=lambda x: x[1], reverse=True)
    top3 = sims[:3]
    top3_str = ", ".join(f"{l}({cs:+.3f})" for l, cs in top3)
    print(f"  '{v_letter}' is most similar to: {top3_str}")

# --- LAB CHANGE: Show the BOS token's relationship ---
print("\n\n--- BOS (start/end) token similarity to letters ---\n")
bos_sims = []
for l in letter_labels:
    cs = cosine_similarity(embeddings["BOS"], embeddings[l])
    bos_sims.append((l, cs))
bos_sims.sort(key=lambda x: x[1], reverse=True)
print("Most similar to BOS (start/end of name):")
for l, cs in bos_sims[:5]:
    print(f"  '{l}': {cs:+.4f}")
print("Least similar to BOS:")
for l, cs in bos_sims[-5:]:
    print(f"  '{l}': {cs:+.4f}")
print()
print("Letters similar to BOS tend to appear at the START or END of names,")
print("because they appear next to the BOS token in training data.")

print("\n" + "=" * 70)
print("  KEY TAKEAWAY")
print("=" * 70)
print("""
The dot product MEASURES SIMILARITY between vectors:
  - Multiply element-wise, then sum
  - High value = similar vectors (point in same direction)
  - Low/negative value = dissimilar vectors

The model learned embeddings where letters with similar ROLES in names
(like vowels) have similar vectors. This is not programmed -- it
EMERGES from training.

This is exactly what attention does: it uses the dot product between
query and key vectors to figure out which tokens are relevant to
each other. The dot product is the "similarity detector" of the
entire transformer.
""")
