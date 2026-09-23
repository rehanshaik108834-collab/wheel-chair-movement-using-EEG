"""Training / evaluation utilities shared by all experiments."""
import copy
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, fast: bool = False):
    """Seed Python, NumPy and PyTorch.

    fast=False: cuDNN deterministic mode (bit-for-bit reproducible, slower).
    fast=True : cuDNN benchmark mode (~1.8x faster for EEGNet); the seed still fixes weight
                initialisation and batch order, but results can differ in the last decimals.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.benchmark = fast


class ChannelScaler:
    """Per-channel z-scoring. Fit on training trials only, then apply to val/test (no leakage)."""

    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)


def to_device(X, y):
    """Move a whole dataset to the GPU once (EEG trials are small), instead of per batch."""
    return torch.as_tensor(X, device=DEVICE), torch.as_tensor(y, device=DEVICE)


def _batches(X, y, batch_size, shuffle, rng=None):
    """Mini-batches from NumPy arrays (copied per batch) or from tensors already on the device."""
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    on_device = isinstance(X, torch.Tensor)
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        if on_device:
            b = torch.as_tensor(b, device=X.device)
            yield X[b], y[b]
        else:
            yield torch.from_numpy(X[b]).to(DEVICE), torch.from_numpy(y[b]).to(DEVICE)


@torch.no_grad()
def evaluate_loss(model, X, y, batch_size=256):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    for xb, yb in _batches(X, y, batch_size, shuffle=False):
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def train_model(model, X_train, y_train, X_val=None, y_val=None, *, lr=1e-3, batch_size=64,
                max_epochs=300, patience=50, seed=0, full_train_eval=True):
    """Adam + cross-entropy.

    With a validation set: early stopping on validation loss; the weights of the best
    validation epoch are restored. Without one (X_val=None): trains for exactly max_epochs
    and keeps the final weights.

    full_train_eval=True re-evaluates the whole training set in eval mode after each epoch;
    False records the running average over the epoch's mini-batches (train mode, faster).
    If the model defines apply_constraints() (our EEGNet), it is called after every step.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    constrain = getattr(model, "apply_constraints", None)
    use_val = X_val is not None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, waited = np.inf, None, max_epochs - 1, 0
    for epoch in range(max_epochs):
        model.train()
        run_loss, run_correct = 0.0, 0
        for xb, yb in _batches(X_train, y_train, batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            if constrain is not None:
                constrain()
            run_loss += loss.item() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum().item()

        if full_train_eval:
            tr_loss, tr_acc = evaluate_loss(model, X_train, y_train)
        else:
            tr_loss, tr_acc = run_loss / len(X_train), run_correct / len(X_train)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if use_val:
            va_loss, va_acc = evaluate_loss(model, X_val, y_val)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            if va_loss < best_loss:
                best_loss, best_epoch, waited = va_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= patience:
                    break

    if use_val:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history


@torch.no_grad()
def predict_logits(model, X, batch_size=256):
    """Raw logits (before softmax). Kept separate so temperature scaling can be added later."""
    model.eval()
    out = [model(xb).cpu() for xb, _ in _batches(X, np.zeros(len(X), dtype=np.int64), batch_size, False)]
    return torch.cat(out).numpy()


def classification_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}
