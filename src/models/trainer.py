"""Vòng lặp huấn luyện với early stopping, lưu history loss/accuracy."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.long), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 64, device="cpu") -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in _loader(X, np.zeros(len(X)), batch_size, False):
        out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def fit(model, X_train, y_train, X_val, y_val, tcfg: dict, device="cpu", verbose=True):
    model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0.0))
    loss_fn = nn.BCEWithLogitsLoss()
    train_dl = _loader(X_train, y_train, tcfg["batch_size"], True)
    val_dl = _loader(X_val, y_val, tcfg["batch_size"], False)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_state, best_val, wait = None, float("inf"), 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(yb)
            tc += ((logits > 0).float() == yb).sum().item()
            n += len(yb)

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vl += loss_fn(logits, yb).item() * len(yb)
                vc += ((logits > 0).float() == yb).sum().item()
                vn += len(yb)

        history["loss"].append(tl / n)
        history["accuracy"].append(tc / n)
        history["val_loss"].append(vl / vn)
        history["val_accuracy"].append(vc / vn)
        if verbose:
            print(f"epoch {epoch:3d} | loss {tl/n:.4f} acc {tc/n:.4f} | val_loss {vl/vn:.4f} val_acc {vc/vn:.4f}")

        if vl / vn < best_val - 1e-4:
            best_val, wait = vl / vn, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= tcfg["early_stopping_patience"]:
                if verbose:
                    print(f"Early stopping tại epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
