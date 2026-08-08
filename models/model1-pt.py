"""
The same multi-neuron network of Chapter 3, this time in PyTorch.

Same architecture as model1.py:

    x^i_nu --[ w^(1), b^(1) ]--> u^i_mu --[ ReLU ]--> z^i_mu
           --[ w^(2), b^(2) ]--> v^i_eta --[ softmax ]--> yhat^i_eta

The reason to write it out by hand: the training loop below IS the two-sweep
procedure of Chapter 2, line by line.

    model(x)        forward sweep   -- computes u, z, v
    loss.backward() backward sweep  -- backpropagation, fills every .grad
    opt.step()      gradient descent update, eq. (mse4-partials)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)

N_FEATURES = 20   # N, the input dimension  (index nu)
N_HIDDEN = 32     # M, the hidden layer size (index mu)
N_CLASSES = 5     # the output dimension     (index eta)
LR = 0.01         # the book's alpha
BATCH = 32
EPOCHS = 300

# ----- data (identical to model1.py, for a fair comparison) -----
X, y = make_classification(
    n_samples=2000,
    n_features=N_FEATURES,
    n_informative=10,
    n_classes=N_CLASSES,
    random_state=0,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

scaler = StandardScaler().fit(X_tr)          # statistics from the training set only
Xtr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
Xte = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.long)
yte = torch.tensor(y_te, dtype=torch.long)

# ----- model -----
# The last layer outputs the pre-activation v^i_eta (the "logits").
# Do NOT append a softmax here: CrossEntropyLoss applies log-softmax itself,
# and doing it twice is the single most common beginner's bug.
model = nn.Sequential(
    nn.Linear(N_FEATURES, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_CLASSES),
)

loss_fn = nn.CrossEntropyLoss()                     # softmax + categorical cross-entropy
opt = torch.optim.SGD(model.parameters(), lr=LR)

# ----- training loop: the two sweeps, explicitly -----
loss_curve = []
n = len(Xtr)
for epoch in range(EPOCHS):
    perm = torch.randperm(n)                        # reshuffle the mini-batches
    running = 0.0
    for k in range(0, n, BATCH):
        idx = perm[k : k + BATCH]

        v = model(Xtr[idx])                         # forward sweep
        loss = loss_fn(v, ytr[idx])

        opt.zero_grad()                             # gradients accumulate; clear them
        loss.backward()                             # backward sweep = backpropagation
        opt.step()                                  # parameters <- parameters - alpha * grad

        running += loss.item() * len(idx)
    loss_curve.append(running / n)

# ----- evaluation -----
model.eval()
with torch.no_grad():
    v_te = model(Xte)
    acc = (v_te.argmax(dim=1) == yte).float().mean().item()
    proba = torch.softmax(v_te, dim=1)              # here we DO want the softmax

print(f"test accuracy: {acc:.3f}")

# ----- the learned parameters, in the book's terms -----
W1, b1 = model[0].weight, model[0].bias
W2, b2 = model[2].weight, model[2].bias

# Note: PyTorch stores weights as (out_features, in_features), i.e. exactly
# w_{mu nu} with mu first -- unlike scikit-learn, which stores the transpose.
print(f"w^(1): {tuple(W1.shape)}   b^(1): {tuple(b1.shape)}")   # (32, 20)  (32,)
print(f"w^(2): {tuple(W2.shape)}   b^(2): {tuple(b2.shape)}")   # (5, 32)   (5,)
print(f"trainable parameters: {sum(p.numel() for p in model.parameters())}")

print(f"row sums (should be 1): {torch.allclose(proba.sum(dim=1), torch.ones(len(proba)))}")
print(f"loss: {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f} over {EPOCHS} epochs")

# Uncomment for the descent figure described in the Chapter 2 footnote:
# import matplotlib.pyplot as plt
# plt.plot(loss_curve)
# plt.xlabel("epoch"); plt.ylabel(r"$L$")
# plt.savefig("figures/loss_curve_pt.pdf")