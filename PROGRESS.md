# AI Presentation — Progress

Single-file Reveal.js presentation (`index.html`, ~6750 lines) with D3.js interactive slides.  
Open directly in a browser — no server required.

---

## Sections & Slides

### 1. Title
- Title slide: "The Evolution of Artificial Intelligence"

### 2. Early ML & MNIST
- `early-ml` — Overview slide
- `mnist-preprocessing` — Preprocessing tricks (centering, normalisation)
- `mnist-nb-models` — Naive Bayes classifiers on MNIST

### 3. Neural Networks (MLP)
- `nn-intro` — Core idea, neuron diagram (SVG, animated)
- `mnist-to-vector` — D3: pixel grid → flattened input vector
- `perceptron` — D3: single perceptron with weights
- `matrix-equation` — Weights & biases as matrix multiplication
- `activation-functions-intro` — Why activation functions?
- `activation-functions` — Activation functions & role of bias
- `mlp-architecture` — MLP architecture overview
- `mlp-backprop` — D3: forward/backward pass animation
- `gradient-landscape` — Loss landscape & gradient descent (Canvas animation)
- `backprop-simple` — Backprop: one neuron, full calculation
- `bias-backprop` — Backprop through bias
- `backprop-network` — Multi-layer gradient flow
- `backprop-tensor` — Layer as matrix multiplication (tensor view)
- `batch-epochs` — Mini-batches & epochs
- `training-network` — D3: weight evolution during training
- `output-layer` — D3: output layer digit classification
- `training-hyperparameters` — Hyperparameter overview (SVG, animated)
- `optimization-algorithms` — SGD / Adam / RMSProp (Canvas landscape)
- `overfitting-epochs` — Overfitting: when the model memorises (animated chart)
- `training-tips` — Tips & tricks for setting up training

### 4. CNNs
- `cnns` / `conv-filters` — Convolutional filters: sliding masks
- `conv-details` — Sliding & summing: convolution and pooling
- `conv-full-digit` — Full digit: convolution & max pooling
- `cnn-architecture` — CNN architecture: 3D network view
- `cnn-params` — Parameters & learned filters
- `cnn-slabs-c1` — Conv2D (1) activations
- `cnn-slabs-p1` — MaxPool (1) activations
- `cnn-slabs-c2` — Conv2D (2) activations
- `cnn-slabs-p2` — MaxPool (2) activations
- `cnn-eval` — CNN digit classification
- `cnn-training` — Backprop through convolution (SVG, animated)
- `cnn-fcn-anim` — CNN → fully connected animation
- `enc-dec-concepts` — Encoder / decoder concepts
- `cnn-intro` — CNN intro (context slide)
- `webcam-demo` — Live webcam digit classification demo

### 5. Transition
- `no-more-mnist` — "No More MNIST Numbers!"

### 6. Embeddings & NLP
- `embeddings` — Word embeddings intro
- `vector-space-3d` — Similarity in vector space (3D interactive)
- `word-vector-demo` — GloVe word-vector arithmetic demo (interactive input)
- `llm-embeddings` — LLM embeddings deep-dive

### 7. Sequence Models (RNN / LSTM)
- `word-order-intro` — Word order: the missing piece
- `rnn-animation` — D3: RNN state flows through time
- `rnn-limits` — RNN limitations (vanishing gradient, SVG)
- `lstm-animation` — D3: LSTM cell state & gates
- `lstm-use-cases` — RNNs & LSTMs: applications & limitations

### 8. Transformers & Attention
- `transformers` — Section intro
- `allyouneed` — "Attention Is All You Need" paper overview
- `allyouneed_contd` — Paper continued
- `embed-context` — Token embeddings with context
- `qkv-attention` — Q / K / V attention concept
- `v-matrix` — The Value Matrix: what to emphasise in each token embedding
- `attn-head` — **One Attention Head — Full Computation** ← D3 diagram, redesigned 2025-05-13
- `qkv-generation` — QKV generation slide
- `llm-architectures` — LLM Architecture Building Blocks (D3: encoder-only / decoder-only / enc-dec)

### 9. Learning Paradigms
- `paradigms` — Learning paradigms overview

---

## D3 / Animated Slides

| Slide ID | Type | Notes |
|---|---|---|
| `mnist-to-vector` | D3 SVG | Pixel → vector unfold animation |
| `perceptron` | D3 SVG | Weighted sum + activation |
| `mlp-backprop` | D3 SVG | Forward + backward pass |
| `gradient-landscape` | Canvas | 3D loss surface, animated descent |
| `training-network` | D3 SVG | Weight heatmaps evolving per epoch |
| `output-layer` | D3 SVG | Softmax bar chart per digit |
| `optimization-algorithms` | Canvas | SGD / Adam paths on loss surface |
| `rnn-animation` | D3 SVG | Hidden state propagation |
| `lstm-animation` | D3 SVG | Gates and cell state |
| `attn-head` | D3 SVG | Single-head attention: X → Projection → Q/K/V → A → Out |
| `llm-architectures` | D3 SVG | Three architecture columns fade in |

---

## Assets

| File | Purpose |
|---|---|
| `styles.css` | Slide theme (dark, monospace) |
| `neural-network-d3.js` | Shared D3 helpers for MLP animations |
| `mnist-sample.json` | 50 real MNIST test images (IDX binary → JSON) |
| `mnist-sample.js` | Training snapshots for weight-evolution animation |
| `glove-vectors.js` | GloVe 50d subset for word-vector demo |
| `cnn-weights.js` | Trained CNN weights for webcam demo |
| `mnist_cnn.pth` | PyTorch CNN checkpoint |
| `train_mlp.py` | Trains MLP, generates `mnist-sample.js` |
| `train_cnn.py` | Trains CNN |
| `generate-glove.py` | Generates `glove-vectors.js` from GloVe text file |
| `presentation_images/` | Static images embedded in slides |
| `animations/` | Remotion video sub-projects |
| `MNIST_ORG/` | Raw IDX binaries (not committed, see README) |

---

## Pending / Next

- Multi-head attention slide (the `attn-head` slide explicitly sets up a subsequent slide showing how multiple single heads are run in parallel)
- Any remaining transformer section slides after `llm-architectures`
- Review of `paradigms` section completeness
