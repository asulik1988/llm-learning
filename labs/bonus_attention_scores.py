"""
=============================================================================
LAB 03: See What Attention Focuses On
=============================================================================

CONCEPT:
    Attention is the mechanism that lets a GPT "look back" at previous tokens
    when predicting the next one. At each position, the model computes
    attention weights -- a probability distribution over all previous positions
    that determines how much each one matters for the current prediction.

    With multi-head attention, the model has several independent "heads" that
    can learn different patterns. For example:
    - One head might learn to always focus on the immediately previous token
      ("what letter did I just generate?")
    - Another might focus on the first token ("what letter did the name
      start with?")
    - Another might focus on tokens 2 positions back

    In this lab, we train the model normally, then during inference we
    intercept the attention weights and print them out for every position
    of every generated name.

WHAT WE CHANGED (from microgpt.py):

    Line 137 — gpt() takes an extra capture_attention parameter:
    - Original:  def gpt(token_id, pos_id, keys, values):
    - Changed:   def gpt(token_id, pos_id, keys, values, capture_attention=None):

    Lines 163-168 — attention weight recording added inside attention loop:
    - Original:  (no recording)
    - Changed:   if capture_attention is not None:
                     capture_attention.append({'layer': li, 'head': h, 'pos': pos_id,
                                               'weights': [w.data for w in attn_weights]})

    That's it. Two additions to gpt(). The model architecture, training,
    and all hyperparameters are identical to microgpt.py.

    ADDED (not in microgpt.py):
    - capture_attention parameter and recording logic in gpt()
    - Detailed attention weight printout for 3 generated names
    - Per-head pattern analysis (self, previous, first/BOS, position N)
    - Head behavior summary section

PREDICTION (write down your answers before running!):
    - Do you think all 4 attention heads will learn the same pattern?
    - Which position do you think gets the most attention: the very first
      token, the most recent token, or something else?
    - Will the attention pattern change depending on the position?

WHAT YOU SHOULD SEE:
    For each generated name, you'll see the attention weights at every
    position. Look for:

    1. SPECIALIZATION: Different heads focus on different positions. This is
       the whole point of multi-head attention -- each head captures a
       different relationship.

    2. RECENCY BIAS: At least one head probably puts high weight on the
       most recent token. This makes sense: to predict the next letter in
       a name, knowing the previous letter is very useful (e.g., 'q' is
       almost always followed by 'u').

    3. POSITION EFFECTS: Early positions (when little context is available)
       might show more uniform attention, while later positions (with more
       context) might be more selective.

    4. BOS ATTENTION: Some heads might consistently attend to the BOS
       (start of sequence) token, which acts as a kind of "default" or
       "reset" signal.

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
# LAB CHANGE: Added capture_attention parameter to record attention weights
def gpt(token_id, pos_id, keys, values, capture_attention=None):
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

            # LAB CHANGE: capture the attention weights
            if capture_attention is not None:
                capture_attention.append({
                    'layer': li,
                    'head': h,
                    'pos': pos_id,
                    'weights': [w.data for w in attn_weights],
                })

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
print("(Attention will be captured during inference, not during training)\n")

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
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


# ======================================================================
# LAB CHANGE: Inference with attention capture and display
# ======================================================================

print(f"\n{'='*70}")
print("  ATTENTION ANALYSIS: What does each head focus on?")
print("="*70)

temperature = 0.5

def token_label(token_id, uchars, BOS):
    """Return a human-readable label for a token."""
    if token_id == BOS:
        return "BOS"
    return uchars[token_id]


for sample_idx in range(3):  # LAB CHANGE: only 3 names for detailed output
    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    all_tokens = [BOS]  # track all tokens for labeling
    all_attention = []   # LAB CHANGE: collect attention data

    for pos_id in range(block_size):
        attn_data = []  # LAB CHANGE: per-step attention capture
        logits = gpt(token_id, pos_id, keys_cache, values_cache, capture_attention=attn_data)
        all_attention.append(attn_data)

        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        all_tokens.append(token_id)

        if token_id == BOS:
            break
        sample.append(uchars[token_id])

    name = ''.join(sample) if sample else "<empty>"

    # LAB CHANGE: print the attention analysis for this name
    print(f"\n{'─'*70}")
    print(f"  Name #{sample_idx+1}: {name}")
    token_labels = [token_label(t, uchars, BOS) for t in all_tokens]
    print(f"  Tokens: {' '.join(token_labels)}")
    print(f"{'─'*70}")

    for pos_id, step_attn in enumerate(all_attention):
        current_token = token_labels[pos_id]
        next_token = token_labels[pos_id + 1] if pos_id + 1 < len(token_labels) else "?"
        context_labels = token_labels[:pos_id + 1]

        print(f"\n  Position {pos_id} (token='{current_token}', predicting='{next_token}')")
        print(f"  Attending to: [{', '.join(context_labels)}]")

        # Group by head
        heads = {}
        for entry in step_attn:
            h = entry['head']
            heads[h] = entry['weights']

        for h in sorted(heads.keys()):
            weights = heads[h]
            # Format weights with the token they attend to
            weight_strs = []
            for i, w in enumerate(weights):
                label = context_labels[i] if i < len(context_labels) else "?"
                weight_strs.append(f"{label}:{w:.2f}")

            # Find which token gets the most attention
            max_idx = weights.index(max(weights))
            max_label = context_labels[max_idx] if max_idx < len(context_labels) else "?"
            max_weight = weights[max_idx]

            # Describe the pattern
            if max_idx == pos_id:
                pattern = "self"
            elif max_idx == pos_id - 1 and pos_id > 0:
                pattern = "previous"
            elif max_idx == 0:
                pattern = "first (BOS)"
            else:
                pattern = f"pos {max_idx}"

            print(f"    Head {h}: [{', '.join(weight_strs)}]  "
                  f"<-- focuses on '{max_label}' ({pattern}, {max_weight:.0%})")


# ======================================================================
# LAB CHANGE: Summarize head behaviors across all generated names
# ======================================================================

print(f"\n\n{'='*70}")
print("  HEAD BEHAVIOR SUMMARY")
print("="*70)
print("""
  Look at the attention patterns above and ask yourself:

  1. DOES EACH HEAD HAVE A CONSISTENT PATTERN?
     For example, does Head 0 always focus on the same relative position?
     If Head 0 usually attends to the previous token, that's a "bigram head"
     -- it's learning which letter typically follows which.

  2. DO DIFFERENT HEADS SPECIALIZE?
     The whole point of multi-head attention is that each head can learn
     a different pattern. If all heads did the same thing, we'd only need
     one head. Look for diversity: one head on BOS, one on previous, etc.

  3. HOW DOES ATTENTION CHANGE WITH POSITION?
     At position 0 (the first token), there's only one thing to attend to
     (BOS). But at later positions, heads have more choices. Do they become
     more selective (focused on one token) or more spread out?

  4. WHAT ABOUT THE BOS TOKEN?
     BOS is a special "anchor" token. Some heads might always partially
     attend to it, treating it as a default or baseline. This is similar
     to how attention models use [CLS] tokens in BERT.

  In real transformers with many layers, the patterns become much more
  complex: some heads attend to tokens with specific syntactic roles,
  others track long-range dependencies, and others seem to encode
  positional information. But even in our tiny 1-layer model, you can
  see the beginning of specialization.
""")
