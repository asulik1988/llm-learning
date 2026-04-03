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
