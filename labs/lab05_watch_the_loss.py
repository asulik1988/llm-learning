"""
=============================================================================
LAB 05: Watch the Loss -- The Model's Report Card
=============================================================================

CONCEPT:
    The loss is the ONLY signal that drives learning. It is defined as:

        loss = -log(probability the model assigns to the correct answer)

    This single formula is worth staring at:
    - If the model gives the correct answer probability 0.95 (very confident
      and correct), loss = -log(0.95) = 0.05.  Very small. Good job!
    - If the model gives the correct answer probability 0.01 (almost zero),
      loss = -log(0.01) = 4.61.  Very large. Terrible prediction!
    - If the model has no idea and guesses uniformly among 27 options,
      loss = -log(1/27) = 3.30.  This is the "random guess" baseline.

    The model tries to MINIMIZE the loss. That is its entire objective.
    Everything else (learning rate, architecture, data) just serves this goal.

WHAT WE CHANGED (from microgpt.py):

    Nothing. The model, training loop, and hyperparameters are identical
    to microgpt.py. Zero lines changed in the core code.

    ADDED (not in microgpt.py):
    - -log(p) table showing the loss formula at different probabilities
    - analyze_name() function that runs inference on a specific name and
      prints per-position loss (input, target, P(correct), loss, difficulty)
    - Pre-training analysis of "anna" (random model baseline)
    - Post-training analysis of "anna", "james", and "zyx"
    - Before vs. after loss comparison

PREDICTION (write down your answers before running!):
    - Before training, what loss do you expect? (hint: 27 equally likely tokens)
    - After 1000 steps of training, will the loss be lower?
    - In the name "anna", which positions do you think will be EASIEST for the
      model (lowest loss)? Which will be hardest?
    - What is -log(0.5)? What about -log(0.99)?

WHAT YOU SHOULD SEE:
    1. Before training: loss at every position is around 3.3 (= -log(1/27)),
       because the model is randomly guessing among 27 tokens.
    2. After training: loss drops, especially at "easy" positions. For example,
       predicting 'a' after 'n' in "anna" might be easier than predicting the
       first letter after BOS (many names could start).
    3. The -log(p) table shows the curve: probability near 0 gives huge loss,
       probability near 1 gives tiny loss. The curve is NOT linear -- going
       from 0.01 to 0.05 saves way more loss than going from 0.80 to 0.84.
    4. Some positions remain hard even after training, because the training
       data contains many different names starting with different letters.
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

# =========================================================================
# LAB CHANGE: First, examine loss BEFORE training (random model)
# =========================================================================

def token_label(tok_id):
    """Convert a token ID to a readable label."""
    if tok_id == BOS:
        return "BOS"
    return uchars[tok_id]

def analyze_name(name_str, description):
    """Run a name through the model and show loss at every position."""
    tokens = [BOS] + [uchars.index(ch) for ch in name_str] + [BOS]

    print(f"\n{'=' * 65}")
    print(f"  {description}: \"{name_str}\"")
    print(f"  Tokens: {[token_label(t) for t in tokens]}")
    print(f"{'=' * 65}")
    print(f"{'Pos':>4}  {'Input':>6}  {'Target':>7}  {'P(correct)':>11}  {'Loss':>8}  {'Notes'}")
    print("-" * 65)

    keys_eval, values_eval = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    total_loss = 0.0
    n = len(tokens) - 1

    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        logits = gpt(token_id, pos_id, keys_eval, values_eval)
        probs = softmax(logits)
        prob_correct = probs[target_id].data
        loss_val = -math.log(prob_correct)
        total_loss += loss_val

        # Classify difficulty
        if prob_correct > 0.3:
            note = "<-- EASY (model knows this pattern)"
        elif prob_correct > 0.1:
            note = "<-- medium"
        elif prob_correct < 0.05:
            note = "<-- HARD (many possibilities here)"
        else:
            note = ""

        print(f"{pos_id:4d}  {token_label(token_id):>6}  {token_label(target_id):>7}  {prob_correct:11.4f}  {loss_val:8.4f}  {note}")

    avg_loss = total_loss / n
    print("-" * 65)
    print(f"Average loss for \"{name_str}\": {avg_loss:.4f}")
    return avg_loss

print("\n" + "#" * 70)
print("#  PART 1: The -log(p) curve -- understanding the loss formula")
print("#" * 70)

print("""
The loss formula is:  loss = -log(probability of the correct answer)

Let's see what this looks like for different probabilities:
""")

# LAB CHANGE: Print the -log(p) table
print(f"{'Probability':>12}  {'Loss = -log(p)':>15}  {'Interpretation'}")
print("-" * 60)
prob_table = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99]
for p in prob_table:
    loss_val = -math.log(p)
    if p <= 0.05:
        interp = "Model is nearly clueless"
    elif p <= 0.20:
        interp = "Weak guess"
    elif p <= 0.50:
        interp = "Decent guess"
    elif p <= 0.80:
        interp = "Good prediction"
    elif p <= 0.95:
        interp = "Strong prediction"
    else:
        interp = "Near-perfect confidence"
    print(f"{p:12.2f}  {loss_val:15.4f}  {interp}")

print()
print("Notice the curve shape:")
print("  - Going from p=0.01 to p=0.05 saves 1.6 points of loss")
print(f"  - Going from p=0.80 to p=0.84 saves only {-math.log(0.84) - (-math.log(0.80)):.2f} points of loss")
print("  - The loss PUNISHES low confidence much more than it rewards high confidence")
print(f"\nRandom guessing among 27 tokens: p = 1/27 = {1/27:.4f}, loss = -log(1/27) = {-math.log(1/27):.4f}")

# =========================================================================
# LAB CHANGE: Analyze a name BEFORE training
# =========================================================================

print("\n\n" + "#" * 70)
print("#  PART 2: Loss BEFORE training (random model)")
print("#" * 70)
print("\nThe model has random weights. It should assign roughly equal probability")
print(f"to all 27 tokens, giving loss near {-math.log(1/27):.2f} at every position.\n")

avg_before = analyze_name("anna", "BEFORE TRAINING (random weights)")

print(f"\nExpected loss for random guessing: -log(1/27) = {-math.log(1/27):.4f}")
print(f"Actual average loss: {avg_before:.4f}")
print("These should be close! The model knows nothing yet.")

# =========================================================================
# Now train the model
# =========================================================================

print("\n\n" + "#" * 70)
print("#  PART 3: Training for 1000 steps")
print("#" * 70)

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_opt = [0.0] * len(params)
v_opt = [0.0] * len(params)
num_steps = 1000

# LAB CHANGE: Reset gradients before training (they accumulated during the pre-training analysis)
for p in params:
    p.grad = 0

print()
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
        m_opt[i] = beta1 * m_opt[i] + (1 - beta1) * p.grad
        v_opt[i] = beta2 * v_opt[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_opt[i] / (1 - beta1 ** (step + 1))
        v_hat = v_opt[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 100 == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# =========================================================================
# LAB CHANGE: Analyze names AFTER training
# =========================================================================

print("\n\n" + "#" * 70)
print("#  PART 4: Loss AFTER training (trained model)")
print("#" * 70)
print("\nNow the model has learned patterns from 1000 names. Let's see how")
print("much better it predicts each position.\n")

test_names = ["anna", "james", "zyx"]  # LAB CHANGE: common, common, and gibberish

for name in test_names:
    # Check all chars are in vocabulary
    if all(ch in uchars for ch in name):
        analyze_name(name, f"AFTER TRAINING")
    else:
        print(f"\nSkipping '{name}' -- contains characters not in vocabulary")

# =========================================================================
# LAB CHANGE: Compare before vs after
# =========================================================================

print("\n\n" + "#" * 70)
print("#  PART 5: Before vs After comparison")
print("#" * 70)

# Re-analyze "anna" to get the after-training loss
keys_eval, values_eval = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
tokens = [BOS] + [uchars.index(ch) for ch in "anna"] + [BOS]
n = len(tokens) - 1
total_after = 0.0
for pos_id in range(n):
    token_id = tokens[pos_id]
    target_id = tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys_eval, values_eval)
    probs = softmax(logits)
    prob_correct = probs[target_id].data
    total_after += -math.log(prob_correct)
avg_after = total_after / n

print(f"""
Loss for "anna":
  Before training: {avg_before:.4f}  (random guessing)
  After training:  {avg_after:.4f}  (learned patterns)
  Improvement:     {avg_before - avg_after:.4f}  ({(avg_before - avg_after) / avg_before * 100:.1f}% reduction)

Random-guess baseline: -log(1/27) = {-math.log(1/27):.4f}
Perfect prediction:    -log(1.0)  = 0.0000
""")

print("=" * 70)
print("  KEY TAKEAWAY")
print("=" * 70)
print("""
The loss = -log(P(correct answer)) is the model's ENTIRE report card:
  - Random guessing: loss = 3.30 (for 27-token vocab)
  - After training: loss drops as the model learns patterns
  - Easy positions (common patterns) get low loss
  - Hard positions (rare/ambiguous patterns) keep high loss

The loss function has an important shape:
  - Being WRONG is punished severely (-log(0.01) = 4.6)
  - Being RIGHT is rewarded modestly (-log(0.99) = 0.01)
  - This asymmetry forces the model to avoid confident wrong answers

Every single thing the model learns -- embeddings, attention patterns,
MLP weights -- exists to make this one number smaller.
""")
