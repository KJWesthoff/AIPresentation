"""
Train 784->64->32->10 MLP on MNIST, save snapshots + samples to mnist-sample.js
"""
import struct, random, math, json, numpy as np

# ── Load IDX binary ───────────────────────────────────────────────────────────
def load_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0

def load_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)

print("Loading MNIST...")
X_train = load_images('MNIST_ORG/train-images.idx3-ubyte')
y_train = load_labels('MNIST_ORG/train-labels.idx1-ubyte')
X_test  = load_images('MNIST_ORG/t10k-images.idx3-ubyte')
y_test  = load_labels('MNIST_ORG/t10k-labels.idx1-ubyte')
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Architecture ──────────────────────────────────────────────────────────────
NI, NH1, NH2, NO = 784, 64, 32, 10

# He init for W1, small init for W2/W3 so growth is visible
np.random.seed(42)
W1 = np.random.randn(NI,  NH1).astype(np.float32) * np.sqrt(2.0/NI)
b1 = np.zeros(NH1, dtype=np.float32)
W2 = np.random.randn(NH1, NH2).astype(np.float32) * 0.05
b2 = np.zeros(NH2, dtype=np.float32)
W3 = np.random.randn(NH2, NO ).astype(np.float32) * 0.05
b3 = np.zeros(NO,  dtype=np.float32)

def relu(x):    return np.maximum(0, x)
def relu_d(x):  return (x > 0).astype(np.float32)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(X):
    z1 = X @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2; a2 = relu(z2)
    z3 = a2 @ W3 + b3; a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def accuracy(X, y):
    *_, a3 = forward(X)
    return (a3.argmax(axis=1) == y).mean()

def cross_entropy(a3, y):
    n = len(y)
    return -np.log(a3[np.arange(n), y] + 1e-9).mean()

# ── Training ──────────────────────────────────────────────────────────────────
LR = 0.005
BS = 256
EPOCHS = 80
N = len(X_train)

snapshots = []

def make_snapshot(epoch, loss, acc):
    # W1: store center column of 28 pixels × NH1  (28*NH1 values)
    mid = 14  # center column index in 28-wide input
    w1_slice = []
    for row in range(28):
        pixel_idx = row * 28 + mid
        w1_slice.extend(W1[pixel_idx].tolist())
    snap = {
        "epoch": epoch,
        "loss":  round(float(loss), 4),
        "acc":   round(float(acc),  4),
        "W1":    [round(float(v), 5) for v in w1_slice],   # 28*NH1 values
        "b1":    [round(float(v), 5) for v in b1],
        "W2":    [round(float(v), 5) for v in W2.flatten()],   # NH1*NH2
        "b2":    [round(float(v), 5) for v in b2],
        "W3":    [round(float(v), 5) for v in W3.flatten()],   # NH2*NO
        "b3":    [round(float(v), 5) for v in b3],
    }
    return snap

# epoch 0 snapshot (before any training)
*_, a3 = forward(X_test[:1000])
init_loss = cross_entropy(a3, y_test[:1000])
init_acc  = accuracy(X_test[:1000], y_test[:1000])
snapshots.append(make_snapshot(0, init_loss, init_acc))
print(f"Epoch   0 | loss={init_loss:.4f} | acc={init_acc:.4f}")

for epoch in range(1, EPOCHS + 1):
    idx = np.random.permutation(N)
    for start in range(0, N, BS):
        batch = idx[start:start+BS]
        X_b, y_b = X_train[batch], y_train[batch]
        z1, a1, z2, a2, z3, a3 = forward(X_b)
        n_b = len(y_b)

        dz3 = a3.copy(); dz3[np.arange(n_b), y_b] -= 1; dz3 /= n_b
        dW3 = a2.T @ dz3; db3 = dz3.sum(axis=0)
        da2 = dz3 @ W3.T
        dz2 = da2 * relu_d(z2)
        dW2 = a1.T @ dz2; db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_d(z1)
        dW1 = X_b.T @ dz1; db1 = dz1.sum(axis=0)

        W1 -= LR * dW1; b1 -= LR * db1
        W2 -= LR * dW2; b2 -= LR * db2
        W3 -= LR * dW3; b3 -= LR * db3

    if epoch % 2 == 0 or epoch <= 5 or epoch == EPOCHS:
        *_, a3_t = forward(X_test[:2000])
        loss = cross_entropy(a3_t, y_test[:2000])
        acc  = (a3_t.argmax(axis=1) == y_test[:2000]).mean()
        snapshots.append(make_snapshot(epoch, loss, acc))
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | acc={acc:.4f}")

final_acc = accuracy(X_test, y_test)
print(f"\nFinal test accuracy: {final_acc:.4f}")

# ── Sample 50 random test images ──────────────────────────────────────────────
random.seed(99)
indices = random.sample(range(len(X_test)), 50)
samples = []
for i in indices:
    x = X_test[i:i+1]
    _, a1, _, a2, _, a3 = forward(x)
    probs = a3[0].tolist()
    pred  = int(np.argmax(probs))
    samples.append({
        "index":     i,
        "label":     int(y_test[i]),
        "predicted": pred,
        "correct":   pred == int(y_test[i]),
        "probs":     [round(float(p), 6) for p in probs],
        "a1":        [round(float(v), 4) for v in a1[0]],   # 64 ReLU activations
        "a2":        [round(float(v), 4) for v in a2[0]],   # 32 ReLU activations
        "pixels":    [round(float(v), 4) for v in X_test[i]],
    })

# ── Write mnist-sample.js ─────────────────────────────────────────────────────
print("Writing mnist-sample.js ...")
with open('mnist-sample.js', 'w') as f:
    f.write('// Auto-generated by train_mlp.py — 784->64->32->10 MLP\n')
    f.write('window.MNIST_SAMPLES = ')
    f.write(json.dumps(samples, separators=(',', ':')))
    f.write(';\n')
    f.write('window.MNIST_SAMPLE = window.MNIST_SAMPLES[Math.floor(Math.random() * window.MNIST_SAMPLES.length)];\n')
    f.write('window.TRAINING_SNAPSHOTS = ')
    f.write(json.dumps(snapshots, separators=(',', ':')))
    f.write(';\n')

import os
size_kb = os.path.getsize('mnist-sample.js') // 1024
print(f"Done. {len(snapshots)} snapshots, file size: {size_kb} KB, final acc: {final_acc:.4f}")
