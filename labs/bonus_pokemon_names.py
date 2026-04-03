"""
LAB 12: Pokemon Names — The Model Learns Whatever Patterns Exist in the Data
=============================================================================

CONCEPT:
A neural network has no idea what "names" are. It doesn't know about humans
or pokemon or any concept at all. It simply learns the statistical patterns
in whatever data you give it:
- Which characters tend to follow which other characters
- How long sequences typically are
- What characters are common at the start vs. end

If you feed it human names, it learns human name patterns (lots of "tion",
"ley", "son", vowel-consonant alternation, etc).

If you feed it pokemon names, it learns pokemon patterns (lots of "chu",
"saur", "zard", unusual consonant clusters, longer names, etc).

SAME model. SAME code. SAME architecture. DIFFERENT data = DIFFERENT output.

This is one of the most important ideas in deep learning: the model is a
blank slate that absorbs structure from data.

WHAT WE CHANGED (from microgpt.py):

    Lines 15-19 -- data loading replaced with inline pokemon name list:
    - Original:  docs = [line.strip() for line in open('input.txt') if line.strip()]
    - Changed:   pokemon_names = ["bulbasaur", "ivysaur", ...]  (200 pokemon names)
                 Also runs a second experiment on original human names for comparison.

    Lines 25-27 -- tokenizer built per-dataset (different vocab per dataset):
    - Original:  uchars = sorted(set(''.join(docs)))
    - Changed:   uchars_local = sorted(set(''.join(dataset_docs)))
                 (computed inside train_on_data() for each dataset separately)

    Line 186 -- num_steps increased:
    - Original:  num_steps = 1000
    - Changed:   num_steps = 2000  (more steps to learn from small pokemon dataset)

    That's it. The model architecture and forward pass are identical.
    The only real change is WHAT DATA is fed in.

    ADDED (not in microgpt.py):
    - Inline pokemon_names list (200 names)
    - train_on_data() function parameterized by dataset
    - Side-by-side comparison of pokemon vs. human generated names

Run time: ~3-5 minutes
PREDICTION (write down your answers before running!):
-----------------------------------------------------
1. Will the pokemon model generate names that sound pokemon-like?
2. Will the human model generate names that sound human-like?
3. Will the two models ever produce the same name?
4. Which dataset will be harder to learn (higher loss) and why?

WHAT YOU SHOULD SEE:
--------------------
- The pokemon model will generate names with pokemon-like sounds: unusual
  consonant combinations, "chu"/"saur"/"zard" patterns, perhaps longer names.
- The human model will generate familiar-sounding names with common English
  patterns.
- The pokemon dataset is smaller (~200 names vs ~32K) so the model may
  memorize it more, but it also has less data to learn general patterns from.
- The two outputs should look VERY different despite identical architectures.

"""

import os
import math
import random

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

# --- Model hyperparameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# --- GPT forward pass ---
def gpt(token_id, pos_id, keys, values, sd, v_size, n_h, h_dim):
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
        for h in range(n_h):
            hs = h * h_dim
            q_h = q[hs:hs+h_dim]
            k_h = [ki[hs:hs+h_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+h_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(h_dim)) / h_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(h_dim)]
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


# LAB CHANGE: Pokemon name dataset (inline)
pokemon_names = [
    "bulbasaur", "ivysaur", "venusaur", "charmander", "charmeleon",
    "charizard", "squirtle", "wartortle", "blastoise", "caterpie",
    "metapod", "butterfree", "weedle", "kakuna", "beedrill",
    "pidgey", "pidgeotto", "pidgeot", "rattata", "raticate",
    "spearow", "fearow", "ekans", "arbok", "pikachu",
    "raichu", "sandshrew", "sandslash", "nidoran", "nidorina",
    "nidoqueen", "nidorino", "nidoking", "clefairy", "clefable",
    "vulpix", "ninetales", "jigglypuff", "wigglytuff", "zubat",
    "golbat", "oddish", "gloom", "vileplume", "paras",
    "parasect", "venonat", "venomoth", "diglett", "dugtrio",
    "meowth", "persian", "psyduck", "golduck", "mankey",
    "primeape", "growlithe", "arcanine", "poliwag", "poliwhirl",
    "poliwrath", "abra", "kadabra", "alakazam", "machop",
    "machoke", "machamp", "bellsprout", "weepinbell", "victreebel",
    "tentacool", "tentacruel", "geodude", "graveler", "golem",
    "ponyta", "rapidash", "slowpoke", "slowbro", "magnemite",
    "magneton", "farfetchd", "doduo", "dodrio", "seel",
    "dewgong", "grimer", "muk", "shellder", "cloyster",
    "gastly", "haunter", "gengar", "onix", "drowzee",
    "hypno", "krabby", "kingler", "voltorb", "electrode",
    "exeggcute", "exeggutor", "cubone", "marowak", "hitmonlee",
    "hitmonchan", "lickitung", "koffing", "weezing", "rhyhorn",
    "rhydon", "chansey", "tangela", "kangaskhan", "horsea",
    "seadra", "goldeen", "seaking", "staryu", "starmie",
    "scyther", "jynx", "electabuzz", "magmar", "pinsir",
    "tauros", "magikarp", "gyarados", "lapras", "ditto",
    "eevee", "vaporeon", "jolteon", "flareon", "porygon",
    "omanyte", "omastar", "kabuto", "kabutops", "aerodactyl",
    "snorlax", "articuno", "zapdos", "moltres", "dratini",
    "dragonair", "dragonite", "mewtwo", "mew", "chikorita",
    "bayleef", "meganium", "cyndaquil", "quilava", "typhlosion",
    "totodile", "croconaw", "feraligatr", "sentret", "furret",
    "hoothoot", "noctowl", "ledyba", "ledian", "spinarak",
    "ariados", "crobat", "chinchou", "lanturn", "pichu",
    "cleffa", "igglybuff", "togepi", "togetic", "natu",
    "xatu", "mareep", "flaaffy", "ampharos", "bellossom",
    "marill", "azumarill", "sudowoodo", "politoed", "hoppip",
    "skiploom", "jumpluff", "aipom", "sunkern", "sunflora",
    "yanma", "wooper", "quagsire", "espeon", "umbreon",
    "murkrow", "slowking", "misdreavus", "girafarig", "pineco",
    "forretress", "dunsparce", "gligar", "steelix", "snubbull",
    "granbull", "qwilfish", "scizor", "shuckle", "heracross",
    "sneasel", "teddiursa", "ursaring", "slugma", "magcargo",
    "swinub", "piloswine", "corsola",
]

def train_on_data(dataset_docs, dataset_label, num_steps=2000, num_generate=20, seed=42):
    """Train microgpt on a given dataset and generate names."""
    random.seed(seed)

    # Build tokenizer from this dataset
    uchars_local = sorted(set(''.join(dataset_docs)))
    bos_local = len(uchars_local)
    vocab_local = len(uchars_local) + 1

    print(f"\n  Dataset: {dataset_label}")
    print(f"  Num examples: {len(dataset_docs)}")
    print(f"  Vocab size: {vocab_local}")
    print(f"  Characters: {''.join(uchars_local)}")

    # Create model
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {'wte': matrix(vocab_local, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_local, n_embd)}
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in sd.values() for row in mat for p in row]

    # Train
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    loss_history = []

    for step in range(num_steps):
        doc = dataset_docs[step % len(dataset_docs)]
        tokens = [bos_local] + [uchars_local.index(ch) for ch in doc] + [bos_local]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, sd, vocab_local, n_head, head_dim)
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

        if (step + 1) % 400 == 0:
            loss_history.append((step + 1, loss.data))
            print(f"    step {step+1:4d} / {num_steps} | loss {loss.data:.4f}")

    # Generate
    temperature = 0.5
    names = []
    for _ in range(num_generate):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = bos_local
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values, sd, vocab_local, n_head, head_dim)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_local), weights=[p.data for p in probs])[0]
            if token_id == bos_local:
                break
            sample.append(uchars_local[token_id])
        names.append(''.join(sample))

    return loss_history, names


# ============================================================
# RUN BOTH EXPERIMENTS
# ============================================================
print("=" * 70)
print("LAB 12: SAME MODEL, DIFFERENT DATA")
print("=" * 70)

# --- Experiment 1: Pokemon names ---
print("\n" + "-" * 70)
print("EXPERIMENT 1: Training on POKEMON names")
print("-" * 70)
pokemon_history, pokemon_generated = train_on_data(pokemon_names, "Pokemon Names")

# --- Experiment 2: Human names ---
print("\n" + "-" * 70)
print("EXPERIMENT 2: Training on HUMAN names")
print("-" * 70)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
human_docs = [line.strip() for line in open('input.txt') if line.strip()]
random.seed(42)
random.shuffle(human_docs)
human_history, human_generated = train_on_data(human_docs, "Human Names")

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("SIDE-BY-SIDE: GENERATED NAMES")
print("=" * 70)
print(f"{'#':>3}  {'Pokemon Model':>20}  |  {'Human Model':<20}")
print("-" * 50)
for i in range(20):
    pname = pokemon_generated[i] if i < len(pokemon_generated) else ""
    hname = human_generated[i] if i < len(human_generated) else ""
    print(f"{i+1:>3}  {pname:>20}  |  {hname:<20}")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("""
- The EXACT SAME architecture produces completely different outputs
  depending on what data it was trained on.

- The pokemon model learned pokemon-like character patterns:
  unusual consonant clusters, "saur"/"chu"/"zard" suffixes, etc.

- The human model learned human name patterns:
  common syllables, vowel-consonant alternation, familiar endings.

- Neural networks are blank slates. They don't "know" anything —
  they learn whatever statistical patterns exist in the training data.

- This is why training data is so important in AI: the model becomes
  a reflection of whatever you feed it.
""")
