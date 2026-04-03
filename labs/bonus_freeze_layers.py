"""
LAB 08: Freezing Layers -- Which Parts of the Model Matter Most?
================================================================

CONCEPT:
    A transformer has several distinct components:
        - Embeddings (wte, wpe): convert tokens/positions into vectors
        - Attention (attn_wq, attn_wk, attn_wv, attn_wo): let tokens look at each other
        - MLP (mlp_fc1, mlp_fc2): process each token independently
        - LM head (lm_head): convert vectors back to token probabilities

    Which of these components matters most for learning? We can find out by
    "freezing" a component -- letting forward and backward passes run normally,
    but zeroing out the gradients for that component before the optimizer step.
    This means the frozen parameters never change from their random initialization.

    This is also the basis of "transfer learning" and "fine-tuning": in practice,
    people freeze most of a pretrained model and only train a few layers on new data.

    By comparing final loss across experiments, we see which component carries
    the most learning capacity for this task.

WHAT WE CHANGED (from microgpt.py):

    Lines 209-215 — gradients zeroed for frozen parameters before optimizer step:
    - Original:  (no freezing logic)
    - Changed:   for p in params:
                     if id(p) in frozen_params:
                         p.grad = 0   # this is what "freezing" means

    Line 186 — num_steps reduced:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 500

    The model architecture and forward pass are identical. The only change
    is zeroing gradients for selected parameter groups before the optimizer
    step, tested across 4 experiments:
    - Exp 1: freeze nothing (baseline)
    - Exp 2: freeze attention (attn_wq, attn_wk, attn_wv, attn_wo)
    - Exp 3: freeze MLP (mlp_fc1, mlp_fc2)
    - Exp 4: freeze embeddings (wte, wpe)

    That's it. One change: conditionally zero gradients. Everything else identical.

    ADDED (not in microgpt.py):
    - run_experiment() with freeze_set parameter
    - Frozen parameter identification by key name matching
    - Results comparison table (final loss per experiment)

PREDICTION (write your answers before running!):
    1. Which component do you think matters most: embeddings, attention, or MLP?
    2. Will any frozen experiment still learn something useful?
    3. Which will produce the worst loss when frozen?

WHAT YOU SHOULD SEE:
    4 experiments, each training for 500 steps:
    1. Baseline (train everything): should get a reasonable loss
    2. Freeze attention: the model can still learn embeddings and MLP,
       but attention stays random, so tokens can't "communicate"
    3. Freeze MLP: attention can still learn, but the per-token processing
       after attention is stuck at random linear projections
    4. Freeze embeddings: attention and MLP can learn, but the input
       representations are random -- like trying to learn with scrambled inputs

    The component whose freezing causes the WORST loss is the most important one.
    You will also see 5 generated names from each experiment. Experiments with
    higher loss will generate worse (more random-looking) names.

HOW TO RUN:
    python lab08_freeze_layers.py
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

# --- Tokenizer (character-level) ---
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

# --- Model Architecture Constants ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

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


def run_experiment(experiment_name, freeze_set):
    """Run a training experiment, freezing parameters whose keys contain any string in freeze_set."""
    random.seed(42)  # LAB CHANGE: same seed each time for fair comparison
    random.shuffle(docs)

    # Build fresh model parameters
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

    # LAB CHANGE: Identify which parameters to freeze
    frozen_params = set()
    for key, mat in state_dict.items():
        if any(f in key for f in freeze_set):
            for row in mat:
                for p in row:
                    frozen_params.add(id(p))

    num_frozen = sum(1 for p in params if id(p) in frozen_params)
    num_trainable = len(params) - num_frozen
    print(f"  Parameters: {len(params)} total, {num_trainable} trainable, {num_frozen} frozen")

    # GPT forward pass (uses this experiment's state_dict via closure)
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

    # Training
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_adam = [0.0] * len(params)
    v_adam = [0.0] * len(params)

    num_steps = 500
    final_loss = 0

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

        # LAB CHANGE: Zero gradients for frozen parameters BEFORE optimizer step
        for p in params:
            if id(p) in frozen_params:
                p.grad = 0  # LAB CHANGE: this is what "freezing" means

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
            v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_adam[i] / (1 - beta1 ** (step + 1))
            v_hat = v_adam[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        final_loss = loss.data
        if step == 0 or (step + 1) % 100 == 0:
            print(f"    step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    # Generate names
    temperature = 0.5
    generated = []
    for sample_idx in range(5):
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

        name = ''.join(sample)
        generated.append(name)
        print(f"    sample {sample_idx+1}: {name}")

    return final_loss


# ===========================================================================
# LAB CHANGE: Run 4 experiments with different components frozen
# ===========================================================================

results = {}

print(f"\n{'='*60}")
print("EXPERIMENT 1: Train everything (baseline)")
print(f"{'='*60}")
results['1. Baseline (all trainable)'] = run_experiment(
    "Baseline",
    freeze_set=[]  # LAB CHANGE: nothing frozen
)

print(f"\n{'='*60}")
print("EXPERIMENT 2: Freeze ATTENTION (train embeddings + MLP + lm_head)")
print(f"{'='*60}")
results['2. Freeze attention'] = run_experiment(
    "Freeze attention",
    freeze_set=['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo']  # LAB CHANGE: freeze all attention weights
)

print(f"\n{'='*60}")
print("EXPERIMENT 3: Freeze MLP (train embeddings + attention + lm_head)")
print(f"{'='*60}")
results['3. Freeze MLP'] = run_experiment(
    "Freeze MLP",
    freeze_set=['mlp_fc1', 'mlp_fc2']  # LAB CHANGE: freeze both MLP layers
)

print(f"\n{'='*60}")
print("EXPERIMENT 4: Freeze EMBEDDINGS (train attention + MLP + lm_head)")
print(f"{'='*60}")
results['4. Freeze embeddings'] = run_experiment(
    "Freeze embeddings",
    freeze_set=['wte', 'wpe']  # LAB CHANGE: freeze token and position embeddings
)

# ===========================================================================
# LAB CHANGE: Summary comparison
# ===========================================================================

print(f"\n{'='*60}")
print("RESULTS COMPARISON")
print(f"{'='*60}")
print(f"\n{'Experiment':<35} | {'Final Loss':>10}")
print(f"{'-'*35}-+-{'-'*10}")
for name, loss in results.items():
    print(f"{name:<35} | {loss:>10.4f}")

# Find worst
worst = max(results.items(), key=lambda x: x[1])
best = min(results.items(), key=lambda x: x[1])
print(f"\nBest result:  {best[0]} (loss={best[1]:.4f})")
print(f"Worst result: {worst[0]} (loss={worst[1]:.4f})")

print("""
KEY TAKEAWAYS:
- "Freezing" = zeroing gradients so parameters never update from random init
- The component whose freezing hurts most is the most important for the task
- Embeddings are crucial: they define how the model "sees" the input tokens
- Attention allows tokens to communicate -- without it, each position is processed
  independently (like a bag of characters rather than a sequence)
- MLP provides nonlinear processing capacity at each position
- In practice, fine-tuning often freezes early layers and trains later ones,
  because early layers learn general features and later layers learn task-specific ones
""")
