# Lesson 17: Experiments -- Turning the Knobs

Previous: [Lesson 16](./16-full-training-step.md)

## Now You Understand the Machine

Over the past 16 lessons, you have learned what every part of microgpt does and why. Now comes the fun part: changing things and seeing what happens.

microgpt is small enough to run on any computer in a few minutes. Each experiment below tells you exactly what to change, what to expect, and why.

## Experiment 1: Change the Embedding Size (n_embd)

### What to change

At `microgpt.py:101`:

```python
n_embd = 16
```

Change `16` to a different number. You must also verify that `n_embd` is divisible by `n_head` (line 103), since `head_dim = n_embd // n_head`.

### What n_embd controls

The embedding size is the number of dimensions used to represent each token. It controls how much information a single token vector can carry.

Think of it like describing a person:

- `n_embd = 2`: you can say "tall" and "old" -- very limited
- `n_embd = 16`: you can describe height, age, hair color, etc. -- decent
- `n_embd = 64`: you can describe many subtle attributes -- rich

### What to try

| Setting | Parameters | What to expect |
|---------|-----------|---------------|
| `n_embd = 4, n_head = 1` | ~350 | Very small. Model can barely represent characters. Names will sound random. |
| `n_embd = 8, n_head = 2` | ~1,200 | Small. Learns basic patterns (common letters, simple pairs). |
| `n_embd = 16, n_head = 4` | ~4,200 | Default. Generates recognizable name-like strings. |
| `n_embd = 32, n_head = 4` | ~14,500 | Larger. Noticeably better names. Slower training. |
| `n_embd = 64, n_head = 4` | ~53,000 | Much larger. Best quality, but 10x slower per step. |

### Why it matters

Every weight matrix in the model depends on `n_embd`. The Q, K, V projections are `n_embd x n_embd`. The MLP expands to `4 * n_embd`. So changing `n_embd` has a quadratic effect on parameter count -- doubling `n_embd` roughly quadruples the parameters.

More parameters means more capacity to learn patterns, but also more data and steps needed to train them well.

## Experiment 2: Change the Number of Layers (n_layer)

### What to change

At `microgpt.py:100`:

```python
n_layer = 1
```

Change `1` to `2`, `3`, or more.

### What n_layer controls

Each layer is one complete attention-then-MLP block (Lesson 15, steps 5-16). Adding layers gives the model more rounds of processing.

With 1 layer, the model gets one chance to look at past tokens (attention) and one chance to transform the information (MLP). With 2 layers, it gets two chances at each -- and the second layer can work with the refined information from the first.

### What to try

| Setting | Parameters | What to expect |
|---------|-----------|---------------|
| `n_layer = 1` | ~4,200 | Default. One round of processing. |
| `n_layer = 2` | ~7,100 | Two rounds. Can learn more complex patterns like 3-character sequences. |
| `n_layer = 3` | ~10,000 | Three rounds. Diminishing returns on this small dataset. |
| `n_layer = 4` | ~12,900 | Might overfit -- too much capacity for the data. |

### Why more layers help

Consider predicting the next letter after `emm`. With 1 layer:

- Attention looks at past tokens, notices the double `m`
- The MLP processes this and makes a prediction

With 2 layers:

- Layer 1 attention might notice `em` is a common pair
- Layer 1 MLP might encode "we are in a double-consonant pattern"
- Layer 2 attention can now attend to this enriched representation
- Layer 2 MLP can make a more informed prediction

Each layer builds on the previous one's output, enabling increasingly abstract reasoning. But for a small dataset of names, 1-2 layers usually suffice.

## Experiment 3: Change the Number of Attention Heads (n_head)

### What to change

At `microgpt.py:103`:

```python
n_head = 4
```

The value must evenly divide `n_embd`. With `n_embd = 16`, valid choices are `1`, `2`, `4`, `8`, or `16`.

### What n_head controls

The number of heads determines how many different attention patterns the model can compute simultaneously (Lesson 14). More heads means more "perspectives" but smaller dimension per head.

### What to try

| Setting | head_dim | What to expect |
|---------|---------|---------------|
| `n_head = 1` | `16` | One attention pattern. Simpler but less flexible. |
| `n_head = 2` | `8` | Two patterns. Can track two things at once. |
| `n_head = 4` | `4` | Default. Four patterns, each 4-dim. |
| `n_head = 8` | `2` | Eight patterns, but each only 2-dim. Very limited per head. |
| `n_head = 16` | `1` | Sixteen patterns with 1 dim each. Each head can only compute a scalar attention. |

### The tradeoff

More heads means more parallel attention patterns, but each head has fewer dimensions to work with. A head with `head_dim = 1` can only compute a single-number representation -- extremely limited. A head with `head_dim = 16` gets the full embedding space but there is only one head.

The sweet spot depends on the task. For microgpt with `n_embd = 16`, `n_head = 4` gives `head_dim = 4`, which is a reasonable balance.

Note that changing `n_head` does **not** change the total parameter count. The Q, K, V, and output matrices are always `n_embd x n_embd` regardless of how many heads slice them up.

## Experiment 4: Change the Training Duration (num_steps)

### What to change

At `microgpt.py:186`:

```python
num_steps = 1000
```

### What num_steps controls

How many names the model trains on. Each step processes one name and updates all parameters once.

### What to try

| Setting | What to expect |
|---------|---------------|
| `num_steps = 100` | Severely undertrained. Model has seen only 100 names. Outputs will be semi-random. |
| `num_steps = 500` | Partially trained. Learns common patterns but misses subtleties. |
| `num_steps = 1000` | Default. Decent quality for this model size. |
| `num_steps = 5000` | More training. Better quality, but diminishing returns. |
| `num_steps = 10000` | Might see slight further improvement. The model's capacity (4,192 params) limits how much it can learn regardless of training time. |

### Diminishing returns

There are two reasons training stops helping after a while:

1. **Model capacity**: 4,192 parameters can only represent so much. Even with infinite training, a small model cannot perfectly model the English name distribution.

2. **Dataset size**: the names dataset has about 32,000 names. After seeing most of them, the model has extracted the main patterns. Seeing them again helps less.

This is why real LLMs are both large (billions of parameters) and trained on massive data (trillions of tokens). Both axes matter.

## Experiment 5: Change the Training Data

### What to change

The data comes from `input.txt` (`microgpt.py:15-18`). You can replace this file with any list of words, one per line.

### Ideas to try

**Country names**: create an `input.txt` with country names like:

```
france
germany
brazil
japan
canada
```

**Pokemon names**: a list of Pokemon:

```
pikachu
charizard
bulbasaur
squirtle
```

**Random words**: any list of words from a dictionary.

### What to expect

The model learns whatever statistical patterns exist in the data. Train on country names and it will generate country-sounding strings. Train on Pokemon names and it will generate Pokemon-sounding strings.

This reveals something profound: the model has no concept of "names" or "countries." It only sees character sequences and learns their statistical regularities. The architecture is completely general -- the same attention-MLP-softmax pipeline works regardless of what text you feed it.

### How to do it

Replace the data download section (`microgpt.py:15-18`) with your own data:

```python
# Replace the download block with:
# (put your words in input.txt, one per line)
```

Or simply create your own `input.txt` file in the same directory as `microgpt.py` before running. The script only downloads the file if `input.txt` does not already exist (`microgpt.py:15`):

```python
if not os.path.exists('input.txt'):
```

So if you place your file there first, it will use yours.

## Experiment 6: Change the Temperature

### What to change

At `microgpt.py:220`:

```python
temperature = 0.5
```

This only affects inference (generation), not training.

### What temperature does

Temperature controls how "confident" the model acts when choosing the next character. It appears in `microgpt.py:230`:

```python
probs = softmax([l / temperature for l in logits])
```

Every logit is divided by the temperature before softmax. This changes the probability distribution.

### Concrete example

Suppose the model's raw logits for the next character are:

```
logit for 'a': 2.0
logit for 'n': 1.5
logit for 'e': 1.0
all others:    ~0
```

Here is what different temperatures do:

**temperature = 0.1** (very cold, very confident):

Divide logits by `0.1`: `[20.0, 15.0, 10.0, ...]`

After softmax:

```
P(a) = 0.993
P(n) = 0.007
P(e) = 0.000
```

The model almost always picks `a`. Very repetitive, very "safe" output.

**temperature = 0.5** (default):

Divide logits by `0.5`: `[4.0, 3.0, 2.0, ...]`

After softmax:

```
P(a) = 0.665
P(n) = 0.245
P(e) = 0.090
```

The model usually picks `a` but sometimes picks `n`. Balanced variety.

**temperature = 1.0** (neutral):

Logits unchanged: `[2.0, 1.5, 1.0, ...]`

After softmax:

```
P(a) = 0.506
P(n) = 0.307
P(e) = 0.186
```

More spread out. More diversity in the output.

**temperature = 2.0** (very hot, very random):

Divide logits by `2.0`: `[1.0, 0.75, 0.5, ...]`

After softmax:

```
P(a) = 0.383
P(n) = 0.298
P(e) = 0.232
```

Probabilities are nearly uniform. The model picks almost randomly. Output is creative but often nonsensical.

### Summary table for temperature

| Temperature | Effect on probabilities | Output character |
|-------------|----------------------|-----------------|
| `0.1` | Extremely peaked (nearly deterministic) | Very repetitive, picks the top choice almost always |
| `0.5` | Moderately peaked | Good balance of quality and variety |
| `1.0` | Unchanged (raw model output) | More diverse, some odd choices |
| `2.0` | Nearly flat (close to uniform) | Very random, often nonsensical |

The key insight: temperature does not change what the model learned. It only changes how we sample from the model's predictions. Lower temperature means "trust the model's top choice." Higher temperature means "explore more options."

## All Experiments at a Glance

| Experiment | What to change | Line | Default | Effect |
|-----------|---------------|------|---------|--------|
| Embedding size | `n_embd` | 101 | `16` | Richer token representations. More params. |
| Layers | `n_layer` | 100 | `1` | More rounds of attention+MLP. Deeper reasoning. |
| Heads | `n_head` | 103 | `4` | More simultaneous attention patterns. |
| Training steps | `num_steps` | 186 | `1000` | More practice. Diminishing returns. |
| Training data | `input.txt` | 15-18 | Names | Model learns whatever patterns exist in the data. |
| Temperature | `temperature` | 220 | `0.5` | Controls randomness of generated output. |

## Running the Experiments

To run any experiment:

1. Open `microgpt.py`
2. Change the value(s) listed above
3. Run `python microgpt.py`
4. Watch the loss during training -- lower is better
5. Look at the generated names at the end

Try changing one thing at a time so you can see its individual effect. If you change everything at once, it is hard to tell what caused the difference.

The best way to build intuition for neural networks is to experiment. You now have enough understanding to predict what should happen before you run it, then check whether reality matches your prediction.

Next: [Lesson 18](./18-from-microgpt-to-gpt4.md)
