"""
=============================================================================
LAB 05: Overfitting -- Memorization vs. Generalization
=============================================================================

CONCEPT:
    When a model trains on a SMALL dataset, something interesting happens:
    it can memorize the training data perfectly (training loss goes to ~0)
    while completely failing on NEW data it hasn't seen (validation loss
    stays high or even increases).

    This is called OVERFITTING. The model hasn't learned the GENERAL PATTERNS
    of how names work -- it has just memorized the specific names it was
    trained on, like a student who memorizes answers instead of understanding
    the material.

    In this lab, we train on only 20 names (instead of 32,000+). We track
    both training loss and validation loss to watch the gap grow. Then we
    check if the model just regurgitates its 20 training names.

WHAT WE CHANGED (from microgpt.py):

    Lines 20-21 — training data reduced to 20 names:
    - Original:  docs = [line.strip() for line in open('input.txt') ...] (32K+ names)
    - Changed:   train_docs = all_docs[:20]      # only 20 names for training!
                 val_docs = all_docs[20:120]      # 100 names for validation

    Line 186 — num_steps tripled:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 3000  (more steps to see overfitting develop)

    Line 189 — training loop cycles over 20 docs:
    - Original:  doc = docs[step % len(docs)]   # cycles through 32K names
    - Changed:   doc = train_docs[step % len(train_docs)]  # cycles through 20 names

    That's it. Three lines changed. The model architecture and forward pass
    are identical to microgpt.py.

    ADDED (not in microgpt.py):
    - Train/val data split (20 train, 100 val)
    - compute_avg_loss() function for evaluation without gradient updates
    - Periodic train AND val loss evaluation with gap tracking
    - Overfitting detection annotations in output
    - Memorization check: comparing generated names to training set
    - ASCII loss curve visualization

PREDICTION (write down your answers before running!):
    - How low do you think training loss will go with only 20 names?
    - Will validation loss also go down, or will it diverge from training?
    - When you generate 10 names, how many will be exact copies of the
      20 training names?

WHAT YOU SHOULD SEE:
    Training loss: Drops steadily, possibly to very low values. The model
    can memorize 20 names easily -- it has thousands of parameters and
    only 20 examples to fit.

    Validation loss: May drop initially (the model learns some general
    patterns), but then STOPS DECREASING or even INCREASES. This divergence
    is the hallmark of overfitting.

    Generated names: Many (possibly most) will be exact copies of the 20
    training names. The model learned "these specific names" rather than
    "how to make names."

    This is why real ML systems need:
    1. Large datasets (more examples = harder to memorize)
    2. Regularization (techniques to prevent memorization)
    3. Train/val/test splits (to detect overfitting)
    4. Early stopping (stop training when val loss stops improving)

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

all_docs = [line.strip() for line in open('input.txt') if line.strip()]
random.seed(42)
random.shuffle(all_docs)

# ======================================================================
# LAB CHANGE: Split data into tiny training set and validation set
# ======================================================================
train_docs = all_docs[:20]     # LAB CHANGE: only 20 names for training!
val_docs = all_docs[20:120]    # LAB CHANGE: 100 names for validation

print("="*70)
print("  LAB 05: Overfitting -- Memorization vs. Generalization")
print("="*70)
print(f"\n  Training set:   {len(train_docs)} names (TINY!)")
print(f"  Validation set: {len(val_docs)} names")
print(f"\n  Training names:")
for i, name in enumerate(train_docs):
    print(f"    {i+1:2d}. {name}")
print()

# --- Tokenizer (character-level) ---
# Use ALL docs for vocabulary so val names don't have unknown chars
uchars = sorted(set(''.join(all_docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"  vocab size: {vocab_size}")

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
print(f"  num params: {len(params)}")
print(f"\n  Ratio: {len(params)} parameters learning from only {len(train_docs)} names.")
print(f"  That's {len(params) // len(train_docs)} parameters per training example -- MASSIVELY overparameterized!\n")


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


# ======================================================================
# LAB CHANGE: Function to compute average loss over a set of documents
# ======================================================================
def compute_avg_loss(doc_list):
    """Compute average loss over a list of documents WITHOUT updating gradients."""
    total_loss = 0.0
    total_tokens = 0

    for doc in doc_list:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys_cache, values_cache)
            probs = softmax(logits)
            # Use .data to avoid building the computation graph
            prob_val = probs[target_id].data
            if prob_val > 0:
                total_loss += -math.log(prob_val)
            else:
                total_loss += 20.0  # cap for log(0)
            total_tokens += 1

    return total_loss / total_tokens if total_tokens > 0 else float('inf')


# --- Training Loop ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_opt = [0.0] * len(params)
v_opt = [0.0] * len(params)

num_steps = 3000  # LAB CHANGE: more steps to see overfitting develop

print(f"  Training for {num_steps} steps on {len(train_docs)} names...")
print(f"  Evaluating train AND val loss every 100 steps.\n")

# LAB CHANGE: track losses for the summary
train_losses = []
val_losses = []
eval_steps = []

print(f"  {'Step':>6}  |  {'Train Loss':>10}  |  {'Val Loss':>10}  |  {'Gap':>10}  |  Notes")
print(f"  {'-'*6}  |  {'-'*10}  |  {'-'*10}  |  {'-'*10}  |  -----")

for step in range(num_steps):
    # LAB CHANGE: cycle through training docs
    doc = train_docs[step % len(train_docs)]
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
        m_opt[i] = beta1 * m_opt[i] + (1 - beta1) * p.grad
        v_opt[i] = beta2 * v_opt[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_opt[i] / (1 - beta1 ** (step + 1))
        v_hat = v_opt[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    # LAB CHANGE: evaluate both train and val loss periodically
    if (step + 1) % 100 == 0 or step == 0:
        t_loss = compute_avg_loss(train_docs)
        v_loss = compute_avg_loss(val_docs[:20])  # use 20 val docs for speed
        gap = v_loss - t_loss
        eval_steps.append(step + 1)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # LAB CHANGE: annotate important moments
        note = ""
        if len(train_losses) >= 2:
            if val_losses[-1] > val_losses[-2] and train_losses[-1] < train_losses[-2]:
                note = "<-- OVERFITTING! Val going UP while train going DOWN"
            elif gap > 0.5:
                note = "<-- Gap is growing"

        print(f"  {step+1:>6}  |  {t_loss:>10.4f}  |  {v_loss:>10.4f}  |  {gap:>+10.4f}  |  {note}")


# ======================================================================
# LAB CHANGE: Generate names and check for memorization
# ======================================================================

print(f"\n\n{'='*70}")
print("  MEMORIZATION CHECK: Are generated names copies of training data?")
print("="*70)

temperature = 0.5
generated_names = []
train_set = set(name.lower() for name in train_docs)

print(f"\n  Generating 10 names and checking against the {len(train_docs)} training names...\n")

for sample_idx in range(10):
    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])

    name = ''.join(sample)
    generated_names.append(name)
    is_memorized = name.lower() in train_set
    marker = "  <<< MEMORIZED from training data!" if is_memorized else ""
    print(f"  {sample_idx+1:2d}. {name}{marker}")

memorized_count = sum(1 for n in generated_names if n.lower() in train_set)
print(f"\n  Result: {memorized_count}/{len(generated_names)} generated names are exact copies of training data.")

if memorized_count >= 5:
    print("  The model is HEAVILY memorizing -- it learned these 20 specific names,")  # LAB CHANGE
    print("  not the general patterns of how names work.")
elif memorized_count >= 2:
    print("  The model is PARTIALLY memorizing -- some outputs are copies, some are novel.")  # LAB CHANGE
else:
    print("  Surprisingly little memorization! The model may have learned some general patterns.")  # LAB CHANGE

# ======================================================================
# LAB CHANGE: Final summary with the loss curve data
# ======================================================================

print(f"\n\n{'='*70}")
print("  LOSS CURVE SUMMARY")
print("="*70)

# Simple ASCII visualization of the gap
print(f"\n  Train vs Val loss over time:\n")
max_loss = max(max(train_losses), max(val_losses))
chart_width = 40

for i in range(len(eval_steps)):
    step = eval_steps[i]
    t = train_losses[i]
    v = val_losses[i]
    t_bar = int((t / max_loss) * chart_width)
    v_bar = int((v / max_loss) * chart_width)
    print(f"  Step {step:>5}: Train |{'#' * t_bar:<{chart_width}}| {t:.3f}")
    print(f"             Val   |{'=' * v_bar:<{chart_width}}| {v:.3f}")
    if i < len(eval_steps) - 1:
        print()

print(f"\n  (# = training loss, = = validation loss)")

print(f"\n\n{'='*70}")
print("  KEY TAKEAWAY")
print("="*70)
print(f"""
  With {len(train_docs)} training names and {len(params)} parameters, the model has
  ~{len(params) // len(train_docs)} parameters per training example. It can easily memorize
  every training name without learning any general patterns.

  This is why:
  1. We always split data into train/val/test sets
  2. We monitor validation loss, not just training loss
  3. When val loss stops improving (or starts increasing), we should STOP
     training -- this is called "early stopping"
  4. More training data is almost always better than a bigger model

  In the real world:
  - GPT-4 was trained on trillions of tokens
  - Even so, it can still memorize common sequences (like famous quotes)
  - The balance between memorization and generalization is one of the
    central challenges in machine learning
""")
