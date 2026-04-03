"""
=============================================================================
LAB 01: Why Random Initialization Matters
=============================================================================

CONCEPT:
    Before a neural network can learn anything, its parameters (weights) need
    starting values. This seems like a small detail, but it turns out to be
    one of the most important decisions in deep learning. The wrong
    initialization can make it IMPOSSIBLE for the network to learn.

    We'll run three experiments:
    1. Normal init (std=0.08) -- the baseline that works
    2. Zero init (std=0.0)   -- every parameter starts at exactly 0
    3. Large init (std=5.0)  -- every parameter starts at a huge random value

WHAT WE CHANGED (from microgpt.py):

    Line 106 -- matrix lambda now parameterized by init_std, wrapped in build_model():
    - Original:  matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) ...
    - Changed:   matrix = lambda nout, nin, std=init_std: [[Value(random.gauss(0, std) if std > 0 else 0.0) ...

    Line 137 -- gpt() now takes state_dict as a parameter:
    - Original:  def gpt(token_id, pos_id, keys, values):
    - Changed:   def gpt(token_id, pos_id, keys, values, state_dict):

    Line 186 -- num_steps reduced:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 200

    Lines 188-217 -- training loop wrapped in train_and_generate() function, runs
    three experiments with init_std = 0.08, 0.0, and 5.0. Adds error handling for
    OverflowError/ValueError. Reports loss at specific steps {1, 50, 100, 200}.

    That's it. The model architecture and forward pass are identical.
    The only real change is the initialization std, tested at three values.

    ADDED (not in microgpt.py):
    - build_model(init_std) function to create fresh models per experiment
    - train_and_generate() function to run each experiment end-to-end
    - Parameter statistics printout (mean, std, min, max) before training
    - OverflowError/ValueError handling for exploding gradients
    - Summary section at the end

PREDICTION (write down your answers before running!):
    - What do you think will happen to the loss for zero initialization?
      Will it go down? Stay flat? Go up?
    - What do you think will happen to the loss for large initialization?
    - Will any of the broken experiments still generate recognizable names?

WHAT YOU SHOULD SEE:
    Experiment 1 (Normal, std=0.08):
        Loss starts around 3.3 and decreases toward ~2.5. Generated names
        look somewhat name-like, even after only 200 steps.

    Experiment 2 (Zero, std=0.0):
        Loss BARELY moves. It might decrease slightly, but stays very high.
        Generated names are garbage -- often the same letter repeated, or
        random-looking sequences. WHY? Because when every weight is 0, every
        neuron computes the exact same output. They all receive the same
        gradients during backprop. So they all update identically. They can
        NEVER specialize -- neuron #1 can never learn something different from
        neuron #2. This is called the "symmetry problem." The network is
        mathematically stuck.

    Experiment 3 (Large, std=5.0):
        Loss starts extremely high or becomes NaN/inf almost immediately.
        Large weights mean large activations, which mean large gradients,
        which mean even larger weight updates, which mean even larger
        activations... This positive feedback loop is called "exploding
        gradients." The math literally overflows.

    The lesson: initialization isn't just a technicality. It's the difference
    between a network that can learn and one that's broken from the start.
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
random.shuffle(docs)

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


# --- Model Hyperparameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head


def build_model(init_std):  # LAB CHANGE: parameterized initialization
    """Build a fresh model with a given initialization std."""
    matrix = lambda nout, nin, std=init_std: [[Value(random.gauss(0, std) if std > 0 else 0.0) for _ in range(nin)] for _ in range(nout)]

    state_dict = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(block_size, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    params = [p for mat in state_dict.values() for row in mat for p in row]
    return state_dict, params


def gpt(token_id, pos_id, keys, values, state_dict):
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


def train_and_generate(init_std, experiment_name):
    """Train a model with the given init_std and generate names."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Initialization std = {init_std}")
    print(f"{'='*70}")

    random.seed(42)  # LAB CHANGE: reset seed so data order is consistent
    random.shuffle(docs)

    state_dict, params = build_model(init_std)
    print(f"  num params: {len(params)}")

    # --- Check initial parameter statistics ---  # LAB CHANGE
    param_vals = [p.data for p in params]
    print(f"  param mean: {sum(param_vals)/len(param_vals):.6f}")
    print(f"  param std:  {(sum((x - sum(param_vals)/len(param_vals))**2 for x in param_vals)/len(param_vals))**0.5:.6f}")
    print(f"  param min:  {min(param_vals):.6f}")
    print(f"  param max:  {max(param_vals):.6f}")
    print()

    # --- Training ---
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    num_steps = 200  # LAB CHANGE: reduced from 1000 for speed
    report_steps = {1, 50, 100, 200}  # LAB CHANGE: steps where we print loss

    loss_went_nan = False

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            try:
                logits = gpt(token_id, pos_id, keys_cache, values_cache, state_dict)
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)
            except (OverflowError, ValueError) as e:
                # LAB CHANGE: catch math errors from exploding values
                print(f"  step {step+1:4d}: MATH ERROR -- {e}")
                loss_went_nan = True
                break

        if loss_went_nan:
            break

        loss = (1 / n) * sum(losses)

        try:
            loss.backward()
        except (OverflowError, ValueError) as e:
            print(f"  step {step+1:4d}: BACKWARD ERROR -- {e}")
            loss_went_nan = True
            break

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        # LAB CHANGE: print at specific steps
        if (step + 1) in report_steps:
            loss_val = loss.data
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"  step {step+1:4d} / {num_steps:4d} | loss = {loss_val}  <-- BROKEN!")
                loss_went_nan = True
            else:
                print(f"  step {step+1:4d} / {num_steps:4d} | loss = {loss_val:.4f}")

    if loss_went_nan:
        print(f"\n  Training FAILED: loss exploded or became NaN.")

    # --- Inference ---
    print(f"\n  Generated names:")
    temperature = 0.5
    for sample_idx in range(5):
        keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []

        for pos_id in range(block_size):
            try:
                logits = gpt(token_id, pos_id, keys_cache, values_cache, state_dict)
                probs = softmax([l / temperature for l in logits])
                token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
                if token_id == BOS:
                    break
                sample.append(uchars[token_id])
            except (OverflowError, ValueError):
                sample.append("<!ERROR!>")
                break

        name = ''.join(sample) if sample else "<empty>"
        print(f"    {sample_idx+1}. {name}")

    print()


# ============================
# RUN ALL THREE EXPERIMENTS
# ============================
print("="*70)
print("  LAB 01: Why Random Initialization Matters")
print("="*70)
print()
print("We're about to train THREE identical models with different")
print("initialization strategies. Watch how the loss behaves!")
print()

train_and_generate(0.08, "NORMAL INIT (std=0.08) -- the baseline")
train_and_generate(0.0,  "ZERO INIT (std=0.0) -- the symmetry problem")
train_and_generate(5.0,  "LARGE INIT (std=5.0) -- the exploding problem")

print("="*70)
print("  SUMMARY")
print("="*70)
print("""
  Normal init (std=0.08): Loss decreases smoothly. Names look plausible.
  Zero init   (std=0.0):  Loss barely moves. All neurons are stuck computing
                          the same thing. Names are nonsense.
  Large init  (std=5.0):  Loss explodes to NaN/inf. The math overflows.
                          Names are errors or garbage.

  KEY TAKEAWAY: Initialization breaks the symmetry between neurons so each
  one can specialize. Too small = stuck. Too large = explosion. Just right
  (like std=0.08) = learning can happen.
""")
