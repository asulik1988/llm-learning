# Understanding LLMs From Scratch

**A hands-on course using [microgpt](../microgpt/microgpt.py) — a complete GPT in ~200 lines of pure Python.**

Every concept is grounded in real, runnable code. No prerequisites — we start from zero.

---

## Part 1: The Math You Need

| # | Lesson | What You'll Learn |
|---|--------|-------------------|
| 1 | [Numbers as Dials](./01-numbers-as-dials.md) | Functions, parameters, and why a model is just a function with 4,192 dials |
| 2 | [Vectors](./02-vectors.md) | Why one number isn't enough — representing tokens as lists of numbers |
| 3 | [The Dot Product](./03-dot-product.md) | Measuring similarity between vectors — the foundation of attention |
| 4 | [Probability & Softmax](./04-probability-and-softmax.md) | Turning raw scores into probabilities that sum to 1 |
| 5 | [The Loss](./05-loss.md) | The model's report card — the only signal that drives all learning |
| 6 | [Derivatives](./06-derivatives.md) | Slopes — how each dial affects the output |

## Part 2: The Neural Network

| # | Lesson | What You'll Learn |
|---|--------|-------------------|
| 7 | [The Chain Rule](./07-chain-rule.md) | Tracing blame backward through a chain of operations (backpropagation) |
| 8 | [A Single Neuron](./08-neuron.md) | Weighted sums — the building block of every neural network |
| 9 | [ReLU](./09-relu.md) | Why max(0, x) is the key to learning complex patterns |
| 10 | [Gradient Descent](./10-gradient-descent.md) | Walking downhill — the algorithm that trains every neural network |
| 11 | [Adam Optimizer](./11-adam.md) | Momentum, adaptive learning rates, and smarter training |
| 12 | [Tokenization & Embeddings](./12-tokenization-and-embeddings.md) | From text to vectors — how words enter the model |

## Part 3: The Transformer

| # | Lesson | What You'll Learn |
|---|--------|-------------------|
| 13 | [Attention](./13-attention.md) | The Q/K/V mechanism — how the model decides what to focus on |
| 14 | [Multi-Head Attention](./14-multi-head-attention.md) | Multiple attention patterns running in parallel |
| 15 | [The Full Forward Pass](./15-full-forward-pass.md) | Token in, prediction out — the complete pipeline |
| 16 | [The Full Training Step](./16-full-training-step.md) | One complete cycle: predict, grade, blame, update |
| 17 | [Experiments](./17-experiments.md) | Change the knobs and see what happens |
| 18 | [From microgpt to GPT-4](./18-from-microgpt-to-gpt4.md) | Same architecture, different scale |

## Labs

Each lesson has a hands-on lab — a runnable Python script that modifies microgpt to drive the concept home. **Predict what will happen before you run it.**

| # | Lab | What You Break/Build |
|---|-----|---------------------|
| 1 | [Break Initialization](../labs/lab01_break_initialization.py) | Zero init, huge init — see why random matters |
| 2 | [See Embeddings](../labs/lab02_see_embeddings.py) | 2D embeddings, scatter plot, watch vowels cluster |
| 3 | [Dot Product Similarity](../labs/lab03_dot_product_similarity.py) | Compute dot products between letter embeddings — which are most similar? |
| 4 | [Temperature](../labs/lab04_temperature.py) | 0.01 to 3.0 — robotic to chaotic |
| 5 | [Watch the Loss](../labs/lab05_watch_the_loss.py) | Trace -log(prob) at every position for real names |
| 6 | [Verify Gradients](../labs/lab06_verify_gradients.py) | Numerical gradient checking — prove backprop is correct |
| 7 | [Add Tanh](../labs/lab07_add_tanh.py) | New Value operation with gradient verification |
| 8 | [Inspect a Neuron](../labs/lab08_inspect_a_neuron.py) | Look at weights, compute outputs by hand, see what a neuron responds to |
| 9 | [Remove ReLU](../labs/lab09_remove_relu.py) | See linearity collapse in action |
| 10 | [Learning Rate Explorer](../labs/lab10_learning_rate_explorer.py) | Plain SGD at 4 learning rates — too small, just right, too big |
| 11 | [Kill Momentum](../labs/lab11_kill_momentum.py) | Disable Adam's tricks one at a time |
| 12 | [Trace the Pipeline](../labs/lab12_trace_the_pipeline.py) | Walk "anna" through every step: text → tokens → embeddings → position → norm |
| 13 | [Remove Attention](../labs/lab13_remove_attention.py) | Skip attention entirely — how much worse? |
| 14 | [One vs Four Heads](../labs/lab14_one_vs_four_heads.py) | Single head vs multi-head attention |
| 15 | [Deeper Model](../labs/lab15_deeper_model.py) | 1 layer vs 2 vs 4 — does depth help? |
| 16 | [LR Warmup](../labs/lab16_lr_warmup.py) | Learning rate schedules: linear, cosine, warmup |

**Bonus labs** (not tied to a specific lesson):

| Lab | What It Does |
|-----|-------------|
| [Attention Scores](../labs/bonus_attention_scores.py) | Print what each attention head focuses on during generation |
| [Overfit on Purpose](../labs/bonus_overfit.py) | Train on 20 names, watch train/val loss diverge |
| [Freeze Layers](../labs/bonus_freeze_layers.py) | Freeze model components one at a time — which matters most? |
| [SGD vs Adam](../labs/bonus_sgd_vs_adam.py) | Compare plain SGD, SGD+momentum, and Adam |
| [Pokemon Names](../labs/bonus_pokemon_names.py) | Same model, different data — generate pokemon names |

```bash
cd labs
python3 lab01_break_initialization.py
```

## Appendices

| | Appendix | What It Covers |
|---|----------|----------------|
| A | [microgpt to PyTorch](./appendix-pytorch-mapping.md) | Mapping every microgpt concept to its PyTorch equivalent |
| B | [Glossary](./glossary.md) | Quick-reference definitions for every term |
| C | [Cheat Sheet](./cheat-sheet.md) | One-page reference card for the entire architecture |

---

## How to Use This Course

1. **Read the lessons in order** — each builds on the previous
2. **Keep `microgpt.py` open** — lessons reference specific line numbers
3. **Ask questions** — when something doesn't click, ask and we'll dig deeper
4. **Run the code** — `cd microgpt && python3 microgpt.py` trains in ~5 minutes

## The Code

```bash
cd microgpt
python3 microgpt.py
```

No dependencies required. Pure Python. Trains on 32,000 names and generates new ones.
