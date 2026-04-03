"""
LAB 13: Remove Attention — What Does Attention Actually Contribute?
====================================================================

CONCEPT:
Attention is the mechanism that lets each position "look at" all previous
positions in the sequence. When generating the 5th character of a name,
attention lets the model consider what the 1st, 2nd, 3rd, and 4th
characters were — and decide what patterns to continue.

Without attention, each position can only use:
- Its own token embedding (what character am I?)
- Its positional embedding (what position am I at?)

That means the model becomes essentially a POSITION-DEPENDENT UNIGRAM model:
"Given that I'm at position 3 and the current character is 'a', what comes
next?" — but it can't see that positions 1 and 2 were "ch", so it doesn't
know we're in the middle of "cha..." (maybe "charles"? "charlotte"?).

The MLP can still learn things like "after 'a' at position 3, common next
characters are..." but it can't learn conditional patterns like "after 'ch'
followed by 'a', the next character tends to be 'r' or 'n'."

WHAT WE CHANGED (from microgpt.py):

    Lines 144-177 — the entire attention block is removed in experiment 2:
    - Original:  (lines 144-168 in microgpt.py: Q/K/V projections, multi-head
                  attention computation, output projection, residual connection)
    - Changed:   All of that is DELETED. The forward pass goes straight from
                  rmsnorm to the MLP, skipping attention entirely.

    The "with attention" experiment is identical to microgpt.py. The "without
    attention" experiment deletes lines 146-168 of the original (the entire
    attention sub-block within the layer loop).

    That's it. One block deleted. Everything else is identical.

    ADDED (not in microgpt.py):
    - gpt_with_attention() and gpt_no_attention() as separate forward functions
    - make_model() with include_attention flag (skips creating Q/K/V/O weights)
    - train() wrapper for running both experiments
    - Loss comparison table and side-by-side generated names

Run time: ~2-4 minutes
PREDICTION (write down your answers before running!):
-----------------------------------------------------
1. How much higher will the no-attention loss be?
2. Will the no-attention model generate anything recognizable?
3. What kinds of patterns CAN it learn without attention?
4. Will the names be shorter or longer without attention?

WHAT YOU SHOULD SEE:
--------------------
- The full model should generate reasonable-looking names.
- The no-attention model will generate gibberish or very simple patterns.
  It might learn that certain characters are common at certain positions
  (e.g., vowels are common at position 2), but it can't learn that "th"
  should be followed by certain characters.
- The no-attention loss will be significantly higher because the model
  simply can't predict as well without seeing context.
- This demonstrates that attention is what gives transformers their power
  to model sequential dependencies.

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
random.shuffle(docs)

# --- Tokenizer ---
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

# --- Model ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head


def gpt_with_attention(token_id, pos_id, keys, values, sd):
    """Standard GPT forward pass (baseline)."""
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


def gpt_no_attention(token_id, pos_id, keys, values, sd):
    """GPT with attention REMOVED — each position is on its own."""
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        # LAB CHANGE: Skip the entire attention block!
        # No Q, K, V projections. No attention weights. No looking at history.
        # Just go straight to the MLP with the residual.

        # MLP (feed-forward network) — this still works, but only sees current position
        x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # residual connection

    logits = linear(x, sd['lm_head'])
    return logits


def make_model(include_attention=True):
    """Create fresh parameters. Skip attention weights if not needed."""
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        if include_attention:
            sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params


def train(label, use_attention, seed=42):
    """Train model with or without attention."""
    random.seed(seed)
    sd, params = make_model(include_attention=use_attention)
    forward_fn = gpt_with_attention if use_attention else gpt_no_attention

    print(f"\n  {label}")
    print(f"  Parameters: {len(params)}")

    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)
    num_steps = 1000
    loss_history = []

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = forward_fn(token_id, pos_id, keys_cache, values_cache, sd)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        if (step + 1) % 200 == 0:
            loss_history.append((step + 1, loss.data))
            print(f"    step {step+1:4d} | loss {loss.data:.4f}")

    # Generate names
    temperature = 0.5
    names = []
    for _ in range(10):
        keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = forward_fn(token_id, pos_id, keys_cache, values_cache, sd)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        names.append(''.join(sample))

    return loss_history, names


# ============================================================
# RUN BOTH EXPERIMENTS
# ============================================================
print("=" * 70)
print("LAB 13: WHAT DOES ATTENTION ACTUALLY CONTRIBUTE?")
print("=" * 70)

print("\n--- Experiment 1: Full model WITH attention (baseline) ---")
attn_history, attn_names = train("With Attention", use_attention=True)

print("\n--- Experiment 2: Model WITHOUT attention ---")
no_attn_history, no_attn_names = train("Without Attention", use_attention=False)

# --- Comparison ---
print("\n" + "=" * 70)
print("LOSS COMPARISON")
print("=" * 70)
print(f"{'Step':>6} | {'With Attention':>16} | {'Without Attention':>18}")
print("-" * 48)
for i in range(len(attn_history)):
    step = attn_history[i][0]
    print(f"{step:>6} | {attn_history[i][1]:>16.4f} | {no_attn_history[i][1]:>18.4f}")

print("\n" + "=" * 70)
print("GENERATED NAMES COMPARISON")
print("=" * 70)
print(f"{'#':>3}  {'With Attention':>20}  |  {'Without Attention':<20}")
print("-" * 50)
for i in range(10):
    print(f"{i+1:>3}  {attn_names[i]:>20}  |  {no_attn_names[i]:<20}")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("""
- WITHOUT attention, the model can't see what came before the current
  position. Each character prediction is based ONLY on:
    1. What the current input character is (token embedding)
    2. What position we're at (positional embedding)

- This means it can learn things like "at position 0, 'a' is common"
  and "after seeing 'e', 'n' is likely" (bigram patterns from the
  token embedding). But it CANNOT learn "after 'ch', 'r' is likely"
  because it can't see two positions back.

- Attention is what makes transformers powerful: it lets every position
  attend to all previous positions, enabling the model to learn
  multi-character patterns and long-range dependencies.

- The loss difference shows exactly how much value attention adds!
""")
