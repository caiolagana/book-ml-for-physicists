"""
Multi-neuron network of Chapter 3, implemented with scikit-learn.

Architecture (matching the book's notation):

    x^i_nu  --[ w^(1)_{mu nu} , b^(1)_mu ]--> u^i_mu --[ ReLU ]--> z^i_mu
            --[ w^(2)_{eta mu} , b^(2)_eta ]--> v^i_eta --[ softmax ]--> yhat^i_eta

MLPClassifier infers the softmax output and the categorical cross-entropy loss
from the labels: multiclass integer y => softmax + cross-entropy, automatically.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

N_FEATURES = 20   # N, the input dimension  (index nu)
N_HIDDEN = 32     # M, the hidden layer size (index mu)
N_CLASSES = 5     # the output dimension     (index eta)

# ----- data -----
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

# ----- model -----
clf = make_pipeline(
    StandardScaler(),                # gradient descent needs comparable scales
    MLPClassifier(
        hidden_layer_sizes=(N_HIDDEN,),
        activation="relu",           # sigma for the hidden layer
        solver="sgd",                # the update rule derived in Chapter 2
        learning_rate_init=0.01,     # this is the book's alpha
        batch_size=32,
        max_iter=300,
        alpha=0.0,                   # L2 penalty -- NOT the learning rate
        random_state=0,
    ),
)

clf.fit(X_tr, y_tr)
print(f"test accuracy: {clf.score(X_te, y_te):.3f}")

# ----- the learned parameters, in the book's terms -----
mlp = clf[-1]
W1, W2 = mlp.coefs_          # note: stored transposed w.r.t. w_{mu nu}
b1, b2 = mlp.intercepts_

print(f"w^(1): {W1.shape}   b^(1): {b1.shape}")   # (20, 32)  (32,)
print(f"w^(2): {W2.shape}   b^(2): {b2.shape}")   # (32, 5)   (5,)
print(f"trainable parameters: {W1.size + b1.size + W2.size + b2.size}")

# ----- the softmax outputs -----
proba = clf.predict_proba(X_te)
print(f"row sums (should be 1): {np.allclose(proba.sum(axis=1), 1.0)}")

# ----- the loss descending the landscape -----
print(f"loss: {mlp.loss_curve_[0]:.4f} -> {mlp.loss_curve_[-1]:.4f} "
      f"over {len(mlp.loss_curve_)} iterations")

# Uncomment for the descent figure described in the Chapter 2 footnote:
# import matplotlib.pyplot as plt
# plt.plot(mlp.loss_curve_)
# plt.xlabel("iteration"); plt.ylabel(r"$L$")
# plt.savefig("figures/loss_curve.pdf")