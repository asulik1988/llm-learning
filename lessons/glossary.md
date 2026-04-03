# Glossary

Quick-reference glossary of every technical term introduced across the 18 lessons.

| Term | Definition | First Introduced |
|------|-----------|-----------------|
| **Parameter** | A learnable number (dial) in the model; microgpt has 4,192 of them. | [Lesson 1](./01-numbers-as-dials.md) |
| **Function** | A machine that takes an input and produces an output: `f(x) = w * x + b`. | [Lesson 1](./01-numbers-as-dials.md) |
| **Loss** | A single number measuring how wrong the model's prediction is; lower is better. | [Lesson 1](./01-numbers-as-dials.md) |
| **Vector** | A list of numbers, e.g. `[0.3, -0.1, 0.7]`; microgpt uses 16-dim vectors. | [Lesson 2](./02-vectors.md) |
| **Embedding** | A learned vector that represents a token or position; looked up from a table. | [Lesson 2](./02-vectors.md) |
| **Token** | The basic unit of text the model works with; in microgpt, a single character. | [Lesson 2](./02-vectors.md) |
| **Vocabulary** | The set of all possible tokens; microgpt has 27 (26 letters + BOS). | [Lesson 2](./02-vectors.md) |
| **BOS** | Beginning/end-of-sequence token (token ID 26); marks the start and end of a name. | [Lesson 2](./02-vectors.md) |
| **Positional Embedding** | A learned vector encoding a token's position in the sequence (`wpe`). | [Lesson 2](./02-vectors.md) |
| **Dot Product** | Multiply corresponding elements and sum; measures similarity between two vectors. | [Lesson 3](./03-dot-product.md) |
| **Cosine Similarity** | Dot product normalized by vector magnitudes; measures directional similarity from -1 to 1. | [Lesson 3](./03-dot-product.md) |
| **Probability** | A number between 0 and 1 representing how likely an outcome is; all probabilities sum to 1. | [Lesson 4](./04-probability-and-softmax.md) |
| **Logit** | A raw, unnormalized score output by the model before softmax converts it to a probability. | [Lesson 4](./04-probability-and-softmax.md) |
| **Softmax** | Converts a list of logits into probabilities: apply `exp()`, then divide by the sum. | [Lesson 4](./04-probability-and-softmax.md) |
| **Negative Log Likelihood** | The loss function `loss = -log(P(correct))`; high probability gives low loss. | [Lesson 5](./05-loss.md) |
| **Cross-Entropy** | Equivalent to negative log likelihood for one-hot targets; the loss used in microgpt. | [Lesson 5](./05-loss.md) |
| **Derivative** | The slope: how much an output changes when an input is nudged by a tiny amount. | [Lesson 6](./06-derivatives.md) |
| **Gradient** | The collection of derivatives of the loss with respect to every parameter; stored in `.grad`. | [Lesson 6](./06-derivatives.md) |
| **Slope** | The rate of change of a function at a point; synonymous with derivative in this context. | [Lesson 6](./06-derivatives.md) |
| **Chain Rule** | To find how A affects C through B, multiply the individual slopes along the chain. | [Lesson 7](./07-chain-rule.md) |
| **Backpropagation** | The chain rule applied automatically from the loss backward to every parameter. | [Lesson 7](./07-chain-rule.md) |
| **Computation Graph** | A record of every operation performed during the forward pass; used by backward(). | [Lesson 7](./07-chain-rule.md) |
| **Topological Sort** | An ordering where if A was used to compute B, A comes before B; used by `build_topo`. | [Lesson 7](./07-chain-rule.md) |
| **Neuron** | Computes a weighted sum of its inputs (a dot product); one row of a weight matrix. | [Lesson 8](./08-neuron.md) |
| **Weighted Sum** | `w1*x1 + w2*x2 + ...`; the core operation of a neuron. | [Lesson 8](./08-neuron.md) |
| **Linear Layer** | A group of neurons that share the same input; implemented by `linear(x, w)`. | [Lesson 8](./08-neuron.md) |
| **Weight Matrix** | A 2D grid of parameters where each row is one neuron's weights. | [Lesson 8](./08-neuron.md) |
| **ReLU** | Rectified Linear Unit: `relu(x) = max(0, x)`; the nonlinearity between MLP layers. | [Lesson 9](./09-relu.md) |
| **Nonlinearity** | An operation (like ReLU) that prevents stacked linear layers from collapsing into one. | [Lesson 9](./09-relu.md) |
| **Activation Function** | A function applied element-wise after a linear layer; ReLU is microgpt's activation function. | [Lesson 9](./09-relu.md) |
| **Gradient Descent** | Update rule: `p.data -= learning_rate * p.grad`; move each dial to reduce the loss. | [Lesson 10](./10-gradient-descent.md) |
| **Learning Rate** | Controls step size during parameter updates; microgpt uses 0.01. | [Lesson 10](./10-gradient-descent.md) |
| **Optimizer** | The algorithm that uses gradients to update parameters; microgpt uses Adam. | [Lesson 11](./11-adam.md) |
| **Adam** | Optimizer with momentum and adaptive per-parameter scaling; the standard for training LLMs. | [Lesson 11](./11-adam.md) |
| **Momentum** | A running average of past gradients that smooths noisy updates (`m` in Adam). | [Lesson 11](./11-adam.md) |
| **Bias Correction** | Early-step compensation in Adam for the fact that `m` and `v` start at zero. | [Lesson 11](./11-adam.md) |
| **Learning Rate Decay** | Linearly shrinking the learning rate during training; big steps early, small steps late. | [Lesson 11](./11-adam.md) |
| **Tokenizer** | The component that converts raw text into a sequence of token IDs. | [Lesson 12](./12-tokenization-and-embeddings.md) |
| **BPE** | Byte Pair Encoding; a subword tokenization method used by real LLMs (GPT-2 has 50,257 tokens). | [Lesson 12](./12-tokenization-and-embeddings.md) |
| **RMSNorm** | Root Mean Square Normalization; scales a vector so its RMS is approximately 1.0. | [Lesson 12](./12-tokenization-and-embeddings.md) |
| **Query (Q)** | A vector representing "what am I looking for?" in attention; produced by `attn_wq`. | [Lesson 13](./13-attention.md) |
| **Key (K)** | A vector representing "what do I contain?" in attention; produced by `attn_wk`. | [Lesson 13](./13-attention.md) |
| **Value (V)** | A vector carrying "my actual information" in attention; produced by `attn_wv`. | [Lesson 13](./13-attention.md) |
| **Attention** | Mechanism that lets each token selectively focus on relevant past tokens via Q, K, V. | [Lesson 13](./13-attention.md) |
| **Attention Score** | The raw dot product of a Query with a Key, before softmax; measures relevance. | [Lesson 13](./13-attention.md) |
| **Attention Weight** | The softmax-normalized attention score; sums to 1 across all past positions. | [Lesson 13](./13-attention.md) |
| **KV Cache** | The growing collection of stored Keys and Values that future tokens can attend to. | [Lesson 13](./13-attention.md) |
| **Causal Masking** | Preventing tokens from attending to future positions; implicit in microgpt's sequential processing. | [Lesson 13](./13-attention.md) |
| **Multi-Head Attention** | Running multiple attention mechanisms in parallel, each on a slice of the embedding dimensions. | [Lesson 14](./14-multi-head-attention.md) |
| **Head** | One independent attention mechanism within multi-head attention; microgpt has 4. | [Lesson 14](./14-multi-head-attention.md) |
| **head_dim** | The number of dimensions per head; `n_embd // n_head = 16 // 4 = 4` in microgpt. | [Lesson 14](./14-multi-head-attention.md) |
| **Residual Connection** | Adding the input of a block back to its output (`x = x + block(x)`); preserves the original signal. | [Lesson 14](./14-multi-head-attention.md) |
| **MLP** | Multi-Layer Perceptron; the feed-forward network: linear (expand) then ReLU then linear (compress). | [Lesson 15](./15-full-forward-pass.md) |
| **Feed-Forward Network** | Synonym for MLP in a transformer; processes each position independently after attention. | [Lesson 15](./15-full-forward-pass.md) |
| **Forward Pass** | Running input through the full model to produce a prediction; data flows left to right. | [Lesson 15](./15-full-forward-pass.md) |
| **Training Step** | One complete cycle: forward pass, loss, backward pass, parameter update. | [Lesson 16](./16-full-training-step.md) |
| **Inference** | Generating new text by repeatedly running the forward pass and sampling from the output. | [Lesson 16](./16-full-training-step.md) |
| **Temperature** | Divides logits before softmax during inference; low = confident, high = random. | [Lesson 17](./17-experiments.md) |
| **Overfitting** | When training loss drops but validation loss rises; the model memorizes instead of generalizing. | [Lesson 17](./17-experiments.md) |
| **Validation Loss** | Loss computed on held-out data the model never trains on; the honest measure of generalization. | [Lesson 17](./17-experiments.md) |
| **RLHF** | Reinforcement Learning from Human Feedback; fine-tuning a pre-trained model to be helpful using human preference rankings. | [Lesson 18](./18-from-microgpt-to-gpt4.md) |
| **Mixture of Experts** | An architecture where multiple expert MLPs exist per layer and a router selects which to activate per token. | [Lesson 18](./18-from-microgpt-to-gpt4.md) |
