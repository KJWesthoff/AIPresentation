# The Evolution of Artificial Intelligence

A [Reveal.js](https://revealjs.com) presentation covering the full arc of modern AI — from early ML and MNIST through CNNs, RNNs, LSTMs, Transformers, and large language models. Features live D3.js neural network animations driven by real MNIST data, an in-browser webcam digit classifier, interactive word-vector explorer, and a 3D attention visualization.

---

## Quick view (no setup)

The committed files are self-contained. Just open `index.html` in a browser:

```bash
git clone https://github.com/<your-handle>/AIPresentation.git
cd AIPresentation
open index.html          # macOS
xdg-open index.html      # Linux
```

> **Note:** The webcam demo requires a proper HTTP origin (browsers block `getUserMedia` on `file://`). Use the dev server below if you need that slide.

---

## Dev server

Run a local HTTP server so all features work (webcam, fetch calls):

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

No build step is required — everything runs in the browser.

---

## Regenerating large assets

Several files are excluded from the repository because of size. Follow the steps below in order to restore them.

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9 + | Training scripts, GloVe generator |
| NumPy | any recent | MLP & CNN training |
| PyTorch | 2.x | CNN training (`train_cnn.py`) |
| Node.js | 18 + | Remotion animation renderer |

```bash
pip install numpy torch torchvision
```

---

### 1. MNIST dataset (~53 MB)

Required by the training scripts. Place the raw IDX files in `MNIST_ORG/`:

```
MNIST_ORG/
  train-images.idx3-ubyte
  train-labels.idx1-ubyte
  t10k-images.idx3-ubyte
  t10k-labels.idx1-ubyte
```

```bash
mkdir -p MNIST_ORG && cd MNIST_ORG
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
mv train-images-idx3-ubyte train-images.idx3-ubyte
mv train-labels-idx1-ubyte train-labels.idx1-ubyte
mv t10k-images-idx3-ubyte  t10k-images.idx3-ubyte
mv t10k-labels-idx1-ubyte  t10k-labels.idx1-ubyte
cd ..
```

> Mirror if the LeCun server is down: `https://storage.googleapis.com/cvdf-datasets/mnist/`

---

### 2. MLP animation data (`mnist-sample.js`, ~1.7 MB)

Trains a 784 → 64 → 32 → 10 MLP on MNIST and serialises 50 sample images plus per-epoch weight snapshots. Required for the D3 neural network animation slides.

```bash
python3 train_mlp.py
```

Takes ~2 minutes on CPU. Produces `mnist-sample.js` in the project root.
Expected final test accuracy: ~92.6 %.

---

### 3. CNN weights (`cnn-weights.js`)

Trains a small convolutional network on MNIST and exports weights for the in-browser webcam classifier.

```bash
python3 train_cnn.py
```

Takes ~5 minutes on CPU (faster with a GPU). Produces `cnn-weights.js`.
Expected final test accuracy: ~99 %.

---

### 4. GloVe word vectors (`glove-vectors.js`, ~4–5 MB)

The interactive word-vector demo uses GloVe 6B 50-dimensional embeddings. The generator downloads the source file automatically (~822 MB zip) if it is not already present, or you can place `glove.6B.50d.txt` in the project root manually.

```bash
python3 generate-glove.py
```

Outputs `glove-vectors.js` (top 10 000 words + curated places, food, and people, L2-normalised).

---

### 5. Remotion animation dependencies (~624 MB)

The `animations/mlp/` sub-project uses [Remotion](https://www.remotion.dev/) to render the MLP backpropagation video.

```bash
cd animations/mlp
npm install
```

To re-render the video:

```bash
npx remotion render
```

Output `.mp4` files are excluded from the repo. Pre-rendered copies can be shared separately.

---

## Project structure

```
index.html                  # Main Reveal.js presentation (entry point)
styles.css                  # Slide styles and theme
neural-network-d3.js        # D3 / Canvas animation code for neural network slides
train_mlp.py                # Trains MLP → generates mnist-sample.js
train_cnn.py                # Trains CNN → generates cnn-weights.js
generate-glove.py           # Generates glove-vectors.js from GloVe source
mnist-sample.js             # Serialised MNIST samples + MLP weight snapshots
cnn-weights.js              # Exported CNN weights for webcam classifier
glove-vectors.js            # Pre-processed GloVe embeddings for word demo
presentation_images/        # Static images embedded in slides
cnn_visuals/                # CNN layer visualisation images
animations/                 # Remotion video sub-projects (mlp, cnn, lstm, transformer)
MNIST_ORG/                  # Raw MNIST IDX files (not committed)
```

---

## Slide topics

- Early ML & MNIST preprocessing
- Naive Bayes classifiers
- Neural network fundamentals & backpropagation
- Activation functions (ReLU, GELU, SiLU, SwiGLU)
- CNNs — architecture, training, webcam live demo
- Word embeddings — Word2Vec, GloVe, interactive 3D vector space
- RNNs & LSTMs
- Transformers — attention, Q/K/V, positional encoding, multi-head attention
- LLM architecture families (encoder-only, decoder-only, encoder-decoder)
- Fine-tuning & RLHF
- Mixture of Experts (MoE)
- LLM model landscape & scale
- AI Agents
