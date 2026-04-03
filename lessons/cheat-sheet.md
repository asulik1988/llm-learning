# microgpt Cheat Sheet

## The Full Pipeline (One Line)

```
input → tokenize → tok_emb + pos_emb → rmsnorm → [attention → residual → MLP → residual] x N → lm_head → softmax → sample
```

## Each Step Expanded

| Step | Operation | Key Formula / Code | Line(s) |
|------|-----------|-------------------|---------|
| **Tokenize** | Characters to integer IDs | `tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]` | 190 |
| **Token embed** | Look up token vector | `tok_emb = state_dict['wte'][token_id]` | 139 |
| **Position embed** | Look up position vector | `pos_emb = state_dict['wpe'][pos_id]` | 140 |
| **Combine** | Add token + position | `x = [t + p for t, p in zip(tok_emb, pos_emb)]` | 141 |
| **RMSNorm** | Normalize magnitude | `scale = (mean(x_i^2) + 1e-5)^(-0.5)` ; `x_i *= scale` | 131-134 |
| **Q, K, V** | Three linear projections | `q = linear(x, attn_wq)` ; same for K, V | 149-151 |
| **Cache K, V** | Append for future tokens | `keys[li].append(k)` ; `values[li].append(v)` | 152-153 |
| **Attention scores** | Scaled dot product | `score_t = (Q . K_t) / sqrt(head_dim)` | 162 |
| **Attention weights** | Softmax over scores | `attn_weights = softmax(attn_logits)` | 163 |
| **Attention output** | Weighted sum of values | `head_out_j = sum(w_t * V_t_j)` | 164 |
| **Multi-head concat** | Stitch heads back together | `x_attn.extend(head_out)` for each head | 165 |
| **Output projection** | Mix across heads | `x = linear(x_attn, attn_wo)` | 167 |
| **Residual** | Add pre-attention input | `x = x + x_residual` | 168 |
| **MLP expand** | 16 to 64 dims | `x = linear(x, mlp_fc1)` | 173 |
| **ReLU** | `max(0, x)` | `x = [xi.relu() for xi in x]` | 174 |
| **MLP compress** | 64 to 16 dims | `x = linear(x, mlp_fc2)` | 175 |
| **Residual** | Add pre-MLP input | `x = x + x_residual` | 176 |
| **Logits** | Project to vocab size | `logits = linear(x, lm_head)` | 178 |
| **Softmax** | Logits to probabilities | `probs = softmax(logits)` | 200 |
| **Loss** | Score the prediction | `loss_t = -log(probs[target_id])` | 201 |

## Shapes -- How Data Transforms

```
token_id        1 int
    ↓ wte lookup
tok_emb         16 floats
    + pos_emb   16 floats
    ↓
x               16 floats
    ↓ rmsnorm
x               16 floats
    ↓ Q, K, V projections (each 16x16)
q, k, v         16 floats each
    ↓ split into 4 heads
q_h, k_h, v_h   4 floats each (head_dim = 4)
    ↓ dot product + softmax + weighted sum (per head)
head_out         4 floats
    ↓ concat 4 heads
x_attn          16 floats
    ↓ attn_wo (16x16) + residual
x               16 floats
    ↓ mlp_fc1 (64x16)
x               64 floats
    ↓ relu
x               64 floats
    ↓ mlp_fc2 (16x64) + residual
x               16 floats
    ↓ lm_head (27x16)
logits          27 floats
    ↓ softmax
probs           27 floats (sum to 1)
```

## The Training Loop (6 Lines)

```python
loss = (1/n) * sum(-probs[target].log() for each position)   # 204
loss.backward()                                                # 205
lr_t = learning_rate * (1 - step / num_steps)                  # 208
m[i] = beta1 * m[i] + (1 - beta1) * p.grad                   # 210
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2              # 211
p.data -= lr_t * (m[i] / (1 - beta1^t)) / (sqrt(v[i] / (1 - beta2^t)) + eps)  # 212-214
```

## The Adam Update Formula

```
m_t     = beta1 * m_{t-1}   + (1 - beta1) * g           momentum (smoothed gradient)
v_t     = beta2 * v_{t-1}   + (1 - beta2) * g^2         velocity (smoothed squared gradient)
m_hat   = m_t / (1 - beta1^t)                            bias-corrected momentum
v_hat   = v_t / (1 - beta2^t)                            bias-corrected velocity
p       = p - lr_t * m_hat / (sqrt(v_hat) + eps)         parameter update
```

## Key Hyperparameters

| Parameter | Value | Line | What It Controls |
|-----------|-------|------|-----------------|
| `n_embd` | `16` | 101 | Embedding dimensions (vector size) |
| `n_layer` | `1` | 100 | Number of transformer blocks |
| `n_head` | `4` | 103 | Number of attention heads |
| `head_dim` | `4` | 104 | Dimensions per head (`n_embd // n_head`) |
| `block_size` | `16` | 102 | Maximum context length (positions) |
| `vocab_size` | `27` | 27 | 26 letters + 1 BOS token |
| `learning_rate` | `0.01` | 182 | Initial learning rate |
| `beta1` | `0.85` | 182 | Adam momentum decay |
| `beta2` | `0.99` | 182 | Adam velocity decay |
| `eps_adam` | `1e-8` | 182 | Adam numerical stability constant |
| `num_steps` | `1000` | 186 | Training steps (one name per step) |
| `temperature` | `0.5` | 220 | Inference sampling temperature |
| `std` | `0.08` | 106 | Std dev for random parameter initialization |

## Parameter Count Breakdown

| Matrix | Shape | Count |
|--------|-------|-------|
| `wte` (token embeddings) | 27 x 16 | 432 |
| `wpe` (position embeddings) | 16 x 16 | 256 |
| `attn_wq` | 16 x 16 | 256 |
| `attn_wk` | 16 x 16 | 256 |
| `attn_wv` | 16 x 16 | 256 |
| `attn_wo` | 16 x 16 | 256 |
| `mlp_fc1` | 64 x 16 | 1,024 |
| `mlp_fc2` | 16 x 64 | 1,024 |
| `lm_head` (output projection) | 27 x 16 | 432 |
| **Total** | | **4,192** |

## Key Formulas

| Formula | Where |
|---------|-------|
| `softmax(x_i) = exp(x_i) / sum(exp(x_j))` | Lines 125-129 |
| `rmsnorm(x_i) = x_i / sqrt(mean(x^2) + eps)` | Lines 131-134 |
| `linear(x, w) = [dot(row, x) for row in w]` | Lines 122-123 |
| `relu(x) = max(0, x)` | Lines 57-58 |
| `loss = -log(P(correct))` | Line 201 |
| `attention = softmax(Q . K^T / sqrt(d_k)) . V` | Lines 162-164 |

## Derivative Rules (Value Class)

| Operation | Local Gradients | Line |
|-----------|----------------|------|
| `a + b` | `(1, 1)` | 42 |
| `a * b` | `(b, a)` | 46 |
| `a ** n` | `(n * a^(n-1),)` | 49 |
| `log(a)` | `(1/a,)` | 52 |
| `exp(a)` | `(exp(a),)` | 55 |
| `relu(a)` | `(1 if a > 0 else 0,)` | 58 |
