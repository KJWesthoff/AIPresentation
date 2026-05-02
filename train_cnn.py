#!/usr/bin/env python3
"""
CNN on MNIST — train, inspect learned kernels, feature maps, and run inference.

Architecture matches the CNN Architecture slide:
  Input(28×28×1)
    → Conv2D(16, k=5, same-pad, ReLU) → MaxPool(2)
    → Conv2D(16, k=3, same-pad, ReLU) → MaxPool(2)
    → Flatten → FC(128, ReLU) → FC(10)

Install:  pip install torch matplotlib
Run:      python train_cnn.py
          python train_cnn.py --digit 7   # visualize feature maps for digit 7
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Config ─────────────────────────────────────────────────────────────────────
MNIST_DIR  = Path("MNIST_ORG")
MODEL_PATH = Path("mnist_cnn.pth")
OUT_DIR    = Path("cnn_visuals")
DEVICE = torch.device("cpu")   # Quadro P520 (CC 6.1) not supported by this PyTorch build
EPOCHS     = 5
BATCH_SIZE = 128
LR         = 1e-3

# Presentation theme colours
BG   = "#1a1a2e"
TEXT = "#f8f9fa"
CYAN = "#00d9ff"
PURP = "#9d4edd"

OUT_DIR.mkdir(exist_ok=True)


# ── MNIST ──────────────────────────────────────────────────────────────────────
def _load_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0

def _load_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

def get_loaders():
    X_tr = _load_images(MNIST_DIR / "train-images.idx3-ubyte")
    y_tr = _load_labels(MNIST_DIR / "train-labels.idx1-ubyte")
    X_te = _load_images(MNIST_DIR / "t10k-images.idx3-ubyte")
    y_te = _load_labels(MNIST_DIR / "t10k-labels.idx1-ubyte")

    tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    te = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                    batch_size=BATCH_SIZE)
    return tr, te


# ── Model ──────────────────────────────────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=5, padding=2)   # 28×28 → 28×28×16
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)   # 14×14 → 14×14×16
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(16 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))    # → 14×14×16
        x = self.pool(self.relu(self.conv2(x)))    # →  7×7×16
        x = self.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)

    @torch.no_grad()
    def activations(self, x):
        """Return intermediate tensors for a single image (shape 1×1×28×28)."""
        a1 = self.relu(self.conv1(x));  p1 = self.pool(a1)
        a2 = self.relu(self.conv2(p1)); p2 = self.pool(a2)
        return {"conv1": a1, "pool1": p1, "conv2": a2, "pool2": p2}


# ── Training ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train_or_load(model, train_ld, test_ld):
    criterion = nn.CrossEntropyLoss()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
        print(f"Training {EPOCHS} epochs on {DEVICE} …")
        for ep in range(1, EPOCHS + 1):
            tl, ta = run_epoch(model, train_ld, criterion, optimizer)
            vl, va = run_epoch(model, test_ld,  criterion)
            print(f"  [{ep}/{EPOCHS}]  train  loss={tl:.4f}  acc={ta*100:.1f}%"
                  f"    val  loss={vl:.4f}  acc={va*100:.1f}%")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved → {MODEL_PATH}")

    _, acc = run_epoch(model, test_ld, criterion)
    print(f"Test accuracy: {acc*100:.2f}%\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fig(w, h, title=""):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    return fig

def _off(ax, title=""):
    ax.axis("off")
    ax.set_facecolor(BG)
    if title:
        ax.set_title(title, color=TEXT, fontsize=7, pad=2)

def _save(name):
    out = OUT_DIR / name
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  → {out}")
    plt.show()


# ── 1. Kernels ─────────────────────────────────────────────────────────────────
def plot_kernels(model: MnistCNN):
    """
    Row 1 — 16 conv1 kernels (5×5, single input channel).
    Row 2 — 16 conv2 kernels (3×3, averaged over 16 input channels).
    Blue = negative weight, red = positive.
    """
    w1 = model.conv1.weight.detach().cpu().numpy()   # (16,  1, 5, 5)
    w2 = model.conv2.weight.detach().cpu().numpy()   # (16, 16, 3, 3)

    fig, axes = plt.subplots(2, 16, figsize=(20, 4), facecolor=BG)
    fig.suptitle("Learned Convolutional Kernels", color=TEXT, fontsize=12,
                 fontweight="bold", y=1.02)

    for i in range(16):
        k1 = w1[i, 0]
        v1 = max(abs(k1.min()), abs(k1.max())) or 1e-9
        axes[0, i].imshow(k1, cmap="RdBu_r", vmin=-v1, vmax=v1, interpolation="nearest")
        _off(axes[0, i], f"F{i+1}")

        k2 = w2[i].mean(axis=0)
        v2 = max(abs(k2.min()), abs(k2.max())) or 1e-9
        axes[1, i].imshow(k2, cmap="RdBu_r", vmin=-v2, vmax=v2, interpolation="nearest")
        _off(axes[1, i])

    axes[0, 0].set_ylabel("Conv1 (5×5)\n1 input ch", color=TEXT, fontsize=8)
    axes[1, 0].set_ylabel("Conv2 (3×3)\nmean 16 ch", color=TEXT, fontsize=8)

    plt.tight_layout(pad=0.4)
    _save("1_kernels.png")


# ── 2. Conv2 kernels per input channel ────────────────────────────────────────
def plot_kernels_conv2_full(model: MnistCNN):
    """
    16×16 grid: each row = one output filter, each column = its kernel
    for that input channel.  Shows which prior feature maps each filter reads from.
    """
    w2 = model.conv2.weight.detach().cpu().numpy()   # (16, 16, 3, 3)

    fig, axes = plt.subplots(16, 16, figsize=(16, 16), facecolor=BG)
    fig.suptitle("Conv2 Kernels — 16 output filters × 16 input channels (3×3 each)",
                 color=TEXT, fontsize=11, fontweight="bold", y=1.005)

    for out_f in range(16):
        for in_ch in range(16):
            k = w2[out_f, in_ch]
            v = max(abs(k.min()), abs(k.max())) or 1e-9
            axes[out_f, in_ch].imshow(k, cmap="RdBu_r", vmin=-v, vmax=v,
                                       interpolation="nearest")
            axes[out_f, in_ch].axis("off")
        axes[out_f, 0].set_ylabel(f"F{out_f+1}", color=TEXT, fontsize=6, rotation=0,
                                   labelpad=12, va="center")

    for in_ch in range(16):
        axes[0, in_ch].set_title(f"ch{in_ch+1}", color=TEXT, fontsize=6)

    plt.tight_layout(pad=0.2)
    _save("2_kernels_conv2_full.png")


# ── 3. Feature maps for one image ─────────────────────────────────────────────
def plot_feature_maps(model: MnistCNN, img: torch.Tensor, label: int):
    """
    Input + 4 stages (conv1, pool1, conv2, pool2), each showing all 16 channels.
    """
    acts = model.activations(img.unsqueeze(0).to(DEVICE))
    stages = [
        ("conv1", f"28×28  Conv1+ReLU"),
        ("pool1",  "14×14  MaxPool"),
        ("conv2",  "14×14  Conv2+ReLU"),
        ("pool2",  " 7×7   MaxPool"),
    ]

    fig = _fig(22, 9, f"Feature Maps — label: {label}")
    gs  = gridspec.GridSpec(5, 17, figure=fig, hspace=0.38, wspace=0.1,
                            left=0.01, right=0.99, top=0.93, bottom=0.01)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    _off(ax0, "Input")

    for si, (key, title) in enumerate(stages):
        fm  = acts[key][0].cpu().numpy()   # (16, H, W)
        row = si + 1
        lax = fig.add_subplot(gs[row, 0])
        lax.text(0.5, 0.5, title, ha="center", va="center",
                 color=TEXT, fontsize=7, transform=lax.transAxes)
        lax.axis("off")
        for ch in range(16):
            ax = fig.add_subplot(gs[row, ch + 1])
            ax.imshow(fm[ch], cmap="viridis", interpolation="nearest")
            ax.axis("off")
            if si == 0:
                ax.set_title(f"{ch+1}", color=TEXT, fontsize=6)

    _save("3_feature_maps.png")


# ── 4. Conv1 filter responses side-by-side ────────────────────────────────────
def plot_filter_responses(model: MnistCNN, img: torch.Tensor, label: int):
    """
    Input image next to each of the 16 conv1 filter responses.
    Makes it easy to see what each filter detects.
    """
    acts = model.activations(img.unsqueeze(0).to(DEVICE))
    fm   = acts["conv1"][0].cpu().numpy()    # (16, 28, 28)
    w1   = model.conv1.weight.detach().cpu().numpy()

    fig = _fig(20, 8, f"Input vs Conv1 Filter Responses  (label: {label})")
    # 2 rows: top = kernels, bottom = feature maps; plus a column for the input
    gs = gridspec.GridSpec(3, 17, figure=fig, hspace=0.1, wspace=0.08,
                           left=0.01, right=0.99, top=0.91, bottom=0.01)

    # Input image (spans rows 0+1 in col 0)
    ax_in = fig.add_subplot(gs[0:2, 0])
    ax_in.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    _off(ax_in, "Input")

    # Row labels
    for row, lbl in enumerate(["Kernel", "Feature map"]):
        ax_l = fig.add_subplot(gs[row, 0])
        ax_l.axis("off")

    # Row 0: kernels (5×5)
    for i in range(16):
        k = w1[i, 0]
        v = max(abs(k.min()), abs(k.max())) or 1e-9
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(k, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        _off(ax, f"F{i+1}")

    # Row 1: feature maps
    for i in range(16):
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(fm[i], cmap="inferno", interpolation="nearest")
        ax.axis("off")

    # Row 2 label
    ax_k = fig.add_subplot(gs[0, 0]); ax_k.axis("off")
    ax_k.text(0.5, 0.5, "Kernel\n(5×5)", ha="center", va="center",
              color=TEXT, fontsize=7, transform=ax_k.transAxes)
    ax_f = fig.add_subplot(gs[1, 0]); ax_f.axis("off")
    ax_f.text(0.5, 0.5, "Feature\nmap", ha="center", va="center",
              color=TEXT, fontsize=7, transform=ax_f.transAxes)

    _save("4_filter_responses.png")


# ── 5. Inference grid ──────────────────────────────────────────────────────────
def plot_inference(model: MnistCNN, test_ld: DataLoader, n: int = 25):
    """
    Grid of test images annotated with predicted digit and confidence.
    Green border = correct, red = wrong.
    """
    model.eval()
    all_imgs, all_labels = [], []
    for imgs, labels in test_ld:
        all_imgs.append(imgs); all_labels.append(labels)
        if sum(len(x) for x in all_imgs) >= n:
            break
    imgs   = torch.cat(all_imgs)[:n]
    labels = torch.cat(all_labels)[:n].numpy()

    with torch.no_grad():
        logits = model(imgs.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6), facecolor=BG)
    fig.suptitle("Inference Examples   (green = correct · red = wrong)",
                 color=TEXT, fontsize=12, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); ax.set_facecolor(BG); continue
        correct = preds[i] == labels[i]
        color   = "#22c55e" if correct else "#ef4444"
        conf    = probs[i, preds[i]] * 100

        ax.imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"pred {preds[i]}   true {labels[i]}\n{conf:.1f}%",
                     color=color, fontsize=8, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.5)

    plt.tight_layout(pad=0.7)
    _save("5_inference.png")


# ── Export weights + feature maps to JS ───────────────────────────────────────
def export_js_weights(model: MnistCNN, test_imgs: torch.Tensor, test_labels: torch.Tensor):
    """Save trained conv1 kernels + one correctly-classified sample per digit to cnn-weights.js."""
    import json
    model.eval()

    def to_native_norm(fm_tensor):
        """Normalize each channel to [0,1] and flatten — keeps native spatial resolution."""
        fm = fm_tensor[0].cpu().numpy()   # (16, H, W)
        result = []
        for ch in fm:
            mn, mx = float(ch.min()), float(ch.max())
            rng = mx - mn if mx > mn else 1.0
            result.append([round(float(v), 4) for v in ((ch - mn) / rng).flatten()])
        return result

    # Collect one correctly-classified sample per digit 0-9
    samples = []
    for digit in range(10):
        idx_arr = (test_labels == digit).nonzero(as_tuple=True)[0]
        chosen = None
        for i in idx_arr:
            cand = test_imgs[i]
            with torch.no_grad():
                logits = model(cand.unsqueeze(0).to(DEVICE))
                p  = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pr = int(logits.argmax(1).item())
            if pr == digit:
                acts = model.activations(cand.unsqueeze(0).to(DEVICE))
                chosen = {
                    "pixels":    [round(float(v), 4) for v in cand.squeeze().numpy().flatten()],
                    "label":     digit,
                    "predicted": pr,
                    "probs":     [round(float(v), 6) for v in p],
                    "fm_c1":     to_native_norm(acts["conv1"]),  # 16 × 784 (28×28)
                    "fm_p1":     to_native_norm(acts["pool1"]),  # 16 × 196 (14×14)
                    "fm_c2":     to_native_norm(acts["conv2"]),  # 16 × 196 (14×14)
                    "fm_p2":     to_native_norm(acts["pool2"]),  # 16 × 49  (7×7)
                }
                break
        if chosen:
            samples.append(chosen)

    w1 = model.conv1.weight.detach().cpu().numpy()   # (16, 1, 5, 5)
    data = {
        "conv1":   [[[round(float(v), 5) for v in row] for row in w1[i, 0]] for i in range(16)],
        "samples": samples,
    }

    out_path = Path("cnn-weights.js")
    with open(out_path, "w") as f:
        f.write("// Auto-generated by train_cnn.py — CNN weights and feature maps\n")
        f.write("window.CNN_WEIGHTS = ")
        f.write(json.dumps(data, separators=(',', ':')))
        f.write(";\n")
        f.write("window.CNN_SAMPLE = window.CNN_WEIGHTS.samples["
                "Math.floor(Math.random()*window.CNN_WEIGHTS.samples.length)];\n")

    size_kb = out_path.stat().st_size // 1024
    print(f"Exported CNN weights → {out_path}  ({size_kb} KB, {len(samples)} samples)")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=None,
                        help="Digit class (0-9) to use for feature-map / filter plots")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training even if a saved model exists")
    parser.add_argument("--plots", action="store_true",
                        help="Generate matplotlib visualizations (off by default)")
    args = parser.parse_args()

    if args.retrain and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    print(f"Device: {DEVICE}\n")
    train_ld, test_ld = get_loaders()

    model = MnistCNN().to(DEVICE)
    train_or_load(model, train_ld, test_ld)

    # Always collect all test images (needed for JS export)
    all_imgs, all_labels = [], []
    for imgs, labels in test_ld:
        all_imgs.append(imgs); all_labels.append(labels)
    all_imgs   = torch.cat(all_imgs)
    all_labels = torch.cat(all_labels)

    if args.plots:
        target = args.digit if args.digit is not None else 0
        idxs   = (all_labels == target).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            raise ValueError(f"No test samples found for digit {target}")
        sample_img   = all_imgs[idxs[0]]
        sample_label = target

        print(f"Generating visualizations for digit {sample_label} …\n")
        plot_kernels(model)
        plot_kernels_conv2_full(model)
        plot_feature_maps(model, sample_img, sample_label)
        plot_filter_responses(model, sample_img, sample_label)
        plot_inference(model, test_ld, n=25)
        print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")

    export_js_weights(model, all_imgs, all_labels)


if __name__ == "__main__":
    main()
