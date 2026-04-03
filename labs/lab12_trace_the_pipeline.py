"""
=============================================================================
LAB 12: Trace the Full Pipeline -- From Text to Vectors
=============================================================================

CONCEPT:
    Before the transformer can process text, the text must be converted into
    numbers (vectors). This conversion has several steps, and each one is
    important. This lab traces the ENTIRE pipeline for the name "anna":

        "anna"
          |
          v
        [26, 0, 13, 13, 0, 26]        <-- Step 1-2: Tokenization
          |
          v
        Token embeddings (from wte)     <-- Step 3: Look up learned vectors
          |
          v
        + Positional embeddings (wpe)   <-- Step 4: Add position information
          |
          v
        Combined vector                 <-- Step 5: tok_emb + pos_emb
          |
          v
        RMSNorm                         <-- Step 6: Normalize the vector
          |
          v
        READY FOR THE TRANSFORMER       <-- Step 7: This enters the layers

    The key insight: the SAME letter at DIFFERENT positions gets a DIFFERENT
    vector, because the positional embedding changes. The two 'a's in "anna"
    start with the same token embedding but end up as different vectors
    because they are at different positions.

WHAT WE CHANGED (from microgpt.py):

    Nothing. The model, training loop, and hyperparameters are identical
    to microgpt.py. Zero lines changed in the core code.

    ADDED (not in microgpt.py):
    - Step-by-step pipeline trace for the name "anna":
      Step 1: Vocabulary display (character-to-integer mapping)
      Step 2: Tokenization (string to token IDs with BOS markers)
      Step 3: Token embedding lookup (wte table)
      Step 4: Positional embedding lookup (wpe table)
      Step 5: Element-wise addition of tok_emb + pos_emb
      Step 6: RMSNorm (with manual computation shown)
      Step 7: Ready-for-transformer summary
    - Comparison of same letter at different positions ('n' at pos 2 vs 3,
      'a' at pos 1 vs 4) showing different vectors from positional embeddings
    - fmt_vec() and magnitude() helper functions

PREDICTION (write down your answers before running!):
    - The two 'n's in "anna" are at positions 2 and 3. Will their vectors
      entering the transformer be different? Why or why not?
    - The two 'a's are at positions 1 and 4. How will they differ?
    - What does RMSNorm do to the magnitude (length) of the vector?
    - BOS appears at position 0 and position 5. Same or different vectors?

WHAT YOU SHOULD SEE:
    1. The vocabulary maps each character to a fixed integer (a=0, b=1, ...).
    2. Token embeddings are LEARNED vectors -- after training, they encode
       meaning (similar letters have similar embeddings, as you saw in Lab 3).
    3. Positional embeddings are also LEARNED. Each position has its own
       vector that gets ADDED to the token embedding.
    4. The same letter at different positions produces a different combined
       vector. This is how the model knows "this 'a' is the second letter"
       vs "this 'a' is the fifth letter."
    5. RMSNorm rescales the vector so its values have a consistent magnitude.
       This stabilizes training.
    6. The final normalized vector is what enters the transformer layers
       (attention + MLP). Every subsequent computation builds on these vectors.
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

print("\n--- Training for 1000 steps (so embeddings are learned, not random) ---\n")

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
# LAB BEGINS HERE: Trace the full tokenization + embedding pipeline
# =========================================================================

name = "anna"

print("\n\n" + "=" * 70)
print(f"  TRACING THE FULL PIPELINE FOR \"{name}\"")
print("=" * 70)

# =========================================================================
# STEP 1: The Vocabulary
# =========================================================================
print("\n" + "-" * 70)
print("  STEP 1: The Vocabulary")
print("-" * 70)
print("\nEvery character gets a fixed integer ID. This is our vocabulary:\n")

for i, ch in enumerate(uchars):
    end = "\n" if (i + 1) % 9 == 0 else ""
    print(f"  '{ch}' = {i:2d}", end=end)
print(f"\n  BOS = {BOS:2d}  (beginning/end of sequence)")
print(f"\nTotal vocabulary size: {vocab_size} tokens ({len(uchars)} letters + 1 special token)")

# =========================================================================
# STEP 2: Tokenization
# =========================================================================
print("\n\n" + "-" * 70)
print(f"  STEP 2: Tokenize \"{name}\"")
print("-" * 70)

# LAB CHANGE: Build the token sequence step by step
tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
token_labels = ["BOS"] + list(name) + ["BOS"]

print(f"\nThe name \"{name}\" becomes a sequence of token IDs:")
print(f"\n  Add BOS at the start (marks beginning of name)")
print(f"  Convert each character to its ID")
print(f"  Add BOS at the end (marks end of name)")
print()
for i, (tok, label) in enumerate(zip(tokens, token_labels)):
    if label == "BOS":
        print(f"  Position {i}: '{label}' --> token ID {tok}")
    else:
        print(f"  Position {i}: '{label}'   --> token ID {tok}  (because '{label}' is letter #{tok} in the alphabet)")

print(f"\n  Final token sequence: {tokens}")
print(f"  In symbols:           {token_labels}")

# =========================================================================
# STEPS 3-7: Process each token through the embedding pipeline
# =========================================================================

# LAB CHANGE: Helper to format a vector nicely
def fmt_vec(vec, width=7):
    """Format a list of floats/Values for printing."""
    vals = [v.data if isinstance(v, Value) else v for v in vec]
    return "[" + ", ".join(f"{v:+{width}.4f}" for v in vals) + "]"

def magnitude(vec):
    """Compute the magnitude (length) of a vector."""
    vals = [v.data if isinstance(v, Value) else v for v in vec]
    return math.sqrt(sum(v * v for v in vals))

# Store the final vectors for comparison later
final_vectors = {}

for pos_id in range(len(tokens) - 1):  # Process each token (not the last BOS, since it has no target)
    token_id = tokens[pos_id]
    label = token_labels[pos_id]

    print(f"\n\n{'=' * 70}")
    print(f"  PROCESSING TOKEN AT POSITION {pos_id}: '{label}' (token ID = {token_id})")
    print(f"{'=' * 70}")

    # --- STEP 3: Token Embedding Lookup ---
    print(f"\n  STEP 3: Look up token embedding from wte[{token_id}]")
    tok_emb = state_dict['wte'][token_id]
    print(f"    tok_emb = wte[{token_id}]")
    print(f"    = {fmt_vec(tok_emb)}")
    print(f"    magnitude = {magnitude(tok_emb):.4f}")

    # --- STEP 4: Positional Embedding Lookup ---
    print(f"\n  STEP 4: Look up positional embedding from wpe[{pos_id}]")
    pos_emb = state_dict['wpe'][pos_id]
    print(f"    pos_emb = wpe[{pos_id}]")
    print(f"    = {fmt_vec(pos_emb)}")
    print(f"    magnitude = {magnitude(pos_emb):.4f}")

    # --- STEP 5: Add them element-wise ---
    print(f"\n  STEP 5: Add token embedding + positional embedding")
    combined = [t.data + p.data for t, p in zip(tok_emb, pos_emb)]
    print(f"    x = tok_emb + pos_emb")
    print(f"    = {fmt_vec(combined)}")
    print(f"    magnitude = {magnitude(combined):.4f}")

    # Show the addition for the first couple of dimensions
    print(f"\n    Showing first 4 dimensions of the element-wise addition:")
    for j in range(4):
        t_val = tok_emb[j].data
        p_val = pos_emb[j].data
        print(f"      dim {j}: {t_val:+.4f} + {p_val:+.4f} = {combined[j]:+.4f}")

    # --- STEP 6: RMSNorm ---
    print(f"\n  STEP 6: Apply RMSNorm (normalize the vector)")

    # LAB CHANGE: Compute RMSNorm step by step
    sum_of_squares = sum(x * x for x in combined)
    mean_square = sum_of_squares / len(combined)
    scale_factor = 1.0 / math.sqrt(mean_square + 1e-5)

    normalized = [x * scale_factor for x in combined]

    print(f"    Sum of squares: {sum_of_squares:.6f}")
    print(f"    Mean square:    {mean_square:.6f}")
    print(f"    Scale factor:   1 / sqrt({mean_square:.6f} + 1e-5) = {scale_factor:.4f}")
    print(f"    Normalized x = each value * {scale_factor:.4f}")
    print(f"    = {fmt_vec(normalized)}")
    print(f"    magnitude AFTER norm = {magnitude(normalized):.4f}")

    # --- STEP 7: Ready for transformer ---
    print(f"\n  STEP 7: This vector enters the transformer!")
    print(f"    It encodes BOTH:")
    print(f"      - What token this is ('{label}', from the token embedding)")
    print(f"      - Where it is in the sequence (position {pos_id}, from the positional embedding)")

    final_vectors[(label, pos_id)] = normalized

# =========================================================================
# LAB CHANGE: Compare the same letter at different positions
# =========================================================================

print("\n\n" + "=" * 70)
print("  KEY COMPARISON: Same letter, different positions")
print("=" * 70)

# Compare the two 'n's (positions 2 and 3)
print("\n--- The two 'n's (positions 2 and 3) ---\n")
n_pos2 = final_vectors[('n', 2)]
n_pos3 = final_vectors[('n', 3)]

print(f"  'n' at position 2: {fmt_vec(n_pos2)}")
print(f"  'n' at position 3: {fmt_vec(n_pos3)}")
print(f"\n  Difference at each dimension:")
diffs_n = [n_pos2[j] - n_pos3[j] for j in range(n_embd)]
print(f"  {fmt_vec(diffs_n)}")
max_diff_n = max(abs(d) for d in diffs_n)
print(f"\n  Max absolute difference: {max_diff_n:.4f}")
print(f"  These are DIFFERENT vectors! Same letter, but the model knows one")
print(f"  is at position 2 and the other is at position 3.")

# Compare the two 'a's (positions 1 and 4)
print("\n\n--- The two 'a's (positions 1 and 4) ---\n")
a_pos1 = final_vectors[('a', 1)]
a_pos4 = final_vectors[('a', 4)]

print(f"  'a' at position 1: {fmt_vec(a_pos1)}")
print(f"  'a' at position 4: {fmt_vec(a_pos4)}")
print(f"\n  Difference at each dimension:")
diffs_a = [a_pos1[j] - a_pos4[j] for j in range(n_embd)]
print(f"  {fmt_vec(diffs_a)}")
max_diff_a = max(abs(d) for d in diffs_a)
print(f"\n  Max absolute difference: {max_diff_a:.4f}")
print(f"  Again DIFFERENT! The 'a' at position 1 (after BOS, starting the name)")
print(f"  gets a different vector than the 'a' at position 4 (before the final BOS).")

# Show WHY they differ: same tok_emb, different pos_emb
print("\n\n--- WHY they differ: same token embedding + different positional embedding ---\n")
tok_emb_a = [state_dict['wte'][uchars.index('a')][j].data for j in range(n_embd)]
pos_emb_1 = [state_dict['wpe'][1][j].data for j in range(n_embd)]
pos_emb_4 = [state_dict['wpe'][4][j].data for j in range(n_embd)]

print(f"  Token embedding for 'a' (same for both): {fmt_vec(tok_emb_a)}")
print(f"  Positional emb for pos 1:                {fmt_vec(pos_emb_1)}")
print(f"  Positional emb for pos 4:                {fmt_vec(pos_emb_4)}")
print(f"\n  tok_emb is identical. pos_emb is different. That's the whole trick!")
print(f"  The model encodes WHERE a letter appears by ADDING a position-specific")
print(f"  vector to the letter's embedding.")

# Compare BOS at position 0 vs the 'a' at position 1
print("\n\n--- BOS (position 0) vs 'a' (position 1) ---\n")
bos_pos0 = final_vectors[('BOS', 0)]
a_pos1_vec = final_vectors[('a', 1)]
print(f"  BOS at position 0: {fmt_vec(bos_pos0)}")
print(f"  'a' at position 1: {fmt_vec(a_pos1_vec)}")
print(f"  These differ in BOTH token embedding AND positional embedding.")

print("\n\n" + "=" * 70)
print("  SUMMARY: The Full Pipeline")
print("=" * 70)
print(f"""
For the name "anna", here is what happened:

  "anna"
    |
    v
  Characters: [BOS, a, n, n, a, BOS]
    |
    v
  Token IDs:  [{', '.join(str(t) for t in tokens)}]
    |
    v  (look up each ID in the wte table)
  Token embeddings: 6 vectors of 16 numbers each
    |
    v  (look up each position in the wpe table)
  + Positional embeddings: 6 vectors of 16 numbers each
    |
    v  (element-wise addition)
  Combined vectors: 6 vectors of 16 numbers each
    |
    v  (divide by root-mean-square to normalize)
  RMSNorm'd vectors: 6 vectors of 16 numbers each
    |
    v
  READY FOR THE TRANSFORMER (attention + MLP layers)

Key insight: the two 'n's got DIFFERENT vectors (different positions).
             the two 'a's got DIFFERENT vectors (different positions).
             Position matters! "anna" is not the same as "naan" because
             the same letters at different positions produce different
             input vectors.

This is the entire front end of a language model. Everything after
this point (attention, MLP, output) operates on these vectors.
""")
