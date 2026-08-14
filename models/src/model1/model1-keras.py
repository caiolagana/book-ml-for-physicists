"""
The same multi-neuron network of Chapter 3, in Keras 3 with the PyTorch backend.

Same architecture as model1.py and model1-pt.py:

    x^i_nu --[ w^(1), b^(1) ]--> u^i_mu --[ ReLU ]--> z^i_mu
           --[ w^(2), b^(2) ]--> v^i_eta --[ softmax ]--> yhat^i_eta

Keras sits one level above PyTorch: the two sweeps that model1-pt.py spells out
by hand (forward, loss.backward(), opt.step()) are folded into model.fit().
The engine underneath is still torch -- KERAS_BACKEND selects it.
"""

import os

# Must be set BEFORE importing keras, otherwise the backend is already fixed.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import layers
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

keras.utils.set_random_seed(0)

N_FEATURES = 8   # N, the input dimension  (index nu)
N_HIDDEN = 12     # M, the hidden layer size (index mu)
N_CLASSES = 5     # the output dimension     (index eta)
LR = 0.01         # the book's alpha
BATCH = 32
EPOCHS = 400

# ----- data (identical to model1.py, for a fair comparison) -----
X, y = make_classification(
    n_samples=3000,
    n_features=N_FEATURES,
    n_informative=N_FEATURES-2,
    n_classes=N_CLASSES,
    random_state=0,
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

scaler = StandardScaler().fit(X_tr)          # statistics from the training set only
# X_tr = scaler.transform(X_tr).astype("float32")
# X_te = scaler.transform(X_te).astype("float32")

# ----- model -----
# The softmax is written explicitly here, mirroring eq. (network output).
# In production one usually leaves the last layer linear and passes
# from_logits=True to the loss, which is numerically stabler -- that is exactly
# what model1-pt.py does with CrossEntropyLoss.
model = keras.Sequential(
    [
        keras.Input(shape=(N_FEATURES,)),
        layers.Dense(N_HIDDEN, activation="relu"),   # u^i_mu then z^i_mu
        layers.Dense(N_CLASSES, activation="softmax"),  # v^i_eta then yhat^i_eta
    ]
)

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),  # integer labels
    metrics=["accuracy"],
)

model.summary()

# ----- training: the two sweeps, now hidden inside fit() -----
history = model.fit(
    X_tr,
    y_tr,
    batch_size=BATCH,
    epochs=EPOCHS,
    verbose=0,
)

# ----- evaluation -----
loss_te, acc_te = model.evaluate(X_te, y_te, verbose=0)
print(f"test accuracy: {acc_te:.3f}")

# ----- the learned parameters, in the book's terms -----
W1, b1 = model.layers[0].get_weights()
W2, b2 = model.layers[1].get_weights()

# Note: Keras stores weights as (in_features, out_features) -- the transpose of
# w_{mu nu}, same convention as scikit-learn, opposite to raw PyTorch.
print(f"w^(1): {W1.shape}   b^(1): {b1.shape}")   # (20, 32)  (32,)
print(f"w^(2): {W2.shape}   b^(2): {b2.shape}")   # (32, 5)   (5,)
print(f"trainable parameters: {model.count_params()}")

# ----- the softmax outputs -----
proba = model.predict(X_te, verbose=0)
print(f"row sums (should be 1): {np.allclose(proba.sum(axis=1), 1.0)}")

# ----- the loss descending the landscape -----
loss_curve = history.history["loss"]
print(f"loss: {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f} over {EPOCHS} epochs")

# Uncomment for the descent figure described in the Chapter 2 footnote:
# import matplotlib.pyplot as plt
# plt.plot(loss_curve)
# plt.xlabel("epoch"); plt.ylabel(r"$L$")
# plt.savefig("figures/loss_curve_keras.pdf")