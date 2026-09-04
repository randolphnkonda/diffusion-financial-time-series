"""Classifier two-sample test (C2ST; Lopez-Paz & Oquab, 2017).

A GRU discriminator is trained to distinguish real from generated windows;
its accuracy on held-out samples measures distributional similarity
(accuracy near 0.5 = indistinguishable).

Procedure (per comparison and per seed):
    - classes balanced by subsampling the larger side; stratified
      train / validation / test split of 64 / 16 / 20 percent;
    - binary cross-entropy, Adam, early stopping on validation accuracy;
    - the best-on-validation checkpoint (not the final epoch) is evaluated
      once on the untouched test set;
    - results reported as mean +/- std over `n_seeds` seeds.

`c2st_battery` runs a separate, identically configured test for each row:
    real vs real        (floor: expected accuracy ~ 0.5),
    real vs Gaussian    (power check: expected accuracy high),
    real vs generated   (the model under evaluation).
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .discriminator import TimeSeriesDiscriminator


# ===========================================================================
# Data prep
# ===========================================================================
def _to_np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x,
                      dtype=np.float32)


def prepare_discriminative_data(real_data, generated_data, rng):
    """(n, L) x2 -> X (n_total, L, 1), y (n_total, 1). Balances counts by
    subsampling the larger side."""
    r, g = _to_np(real_data), _to_np(generated_data)

    if r.ndim != g.ndim or r.shape[-1] != g.shape[-1]:
        raise ValueError(
            f"Window length / ndim mismatch: real {r.shape} vs "
            f"generated {g.shape}."
        )

    n = min(len(r), len(g))
    if len(r) > n:
        r = r[rng.choice(len(r), size=n, replace=False)]
    if len(g) > n:
        g = g[rng.choice(len(g), size=n, replace=False)]

    if r.ndim == 2:                     # (n, L) -> (n, L, 1)
        r, g = r[:, :, None], g[:, :, None]

    X = np.concatenate([r, g], axis=0)
    y = np.concatenate([np.ones((n, 1), np.float32),
                        np.zeros((n, 1), np.float32)], axis=0)
    return X, y


# ===========================================================================
# Single C2ST run (one seed)
# ===========================================================================
def _accuracy(model, X, y):
    with torch.no_grad():
        pred = (model(X) >= 0.5).float()
    return (pred == y).float().mean().item()


def c2st_single(real_data, generated_data, seed=42, hidden_dim=64,
                num_layers=1, batch_size=128, max_epochs=1000, lr=1e-3,
                test_size=0.2, val_size=0.2, patience=20, device="cpu",
                verbose=False, log_every=10, save_path=None):
    """One C2ST run. Returns dict with test accuracy of the best-on-val model."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X, y = prepare_discriminative_data(real_data, generated_data, rng)

    # test split (untouched until the end), then val carved from train
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed)

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_tr, y_tr = t(X_tr), t(y_tr)
    X_va, y_va = t(X_va), t(y_va)
    X_te, y_te = t(X_te), t(y_te)

    disc = TimeSeriesDiscriminator(
        input_dim=X_tr.shape[-1], hidden_dim=hidden_dim,
        num_layers=num_layers).to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val, best_state, bad = 0.0, None, 0
    n = len(X_tr)
    train_losses, val_accs = [], []

    for epoch in range(max_epochs):
        disc.train()
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = criterion(disc(X_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        epoch_loss /= n
        train_losses.append(epoch_loss)

        disc.eval()
        val_acc = _accuracy(disc, X_va, y_va)
        val_accs.append(val_acc)

        if verbose and (epoch == 0 or (epoch + 1) % log_every == 0):
            print(f"  epoch {epoch + 1:4d} | loss {epoch_loss:.6f} | "
                  f"val acc {val_acc:.4f} | best {max(best_val, val_acc):.4f}")

        if val_acc > best_val:
            best_val, best_state, bad = val_acc, deepcopy(disc.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} "
                          f"(best val acc {best_val:.4f})")
                break

    disc.load_state_dict(best_state)      # best-on-validation model
    disc.eval()
    test_acc = _accuracy(disc, X_te, y_te)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "input_dim": int(X_tr.shape[-1]),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "model_state_dict": disc.state_dict(),
            "seed": seed,
            "val_accuracy": best_val,
            "test_accuracy": test_acc,
            "stopped_epoch": epoch + 1,
        }, save_path)
        if verbose:
            print(f"  saved classifier to {save_path}")

    return {
        "accuracy": test_acc,
        "discriminative_score": abs(test_acc - 0.5),
        "val_accuracy": best_val,
        "stopped_epoch": epoch + 1,
        "train_losses": train_losses,
        "val_accuracies": val_accs,
        "model": disc,
    }


# ===========================================================================
# Multi-seed wrapper
# ===========================================================================
def c2st_score(real_data, generated_data, n_seeds=3, seed0=42,
               save_path=None, **kw):
    """Repeat c2st_single over seeds; report mean +/- std.

    If save_path is given, the classifier from the BEST seed (highest test
    accuracy, i.e. the strongest distinguisher found) is saved there.
    """
    accs, runs = [], []
    for s in range(n_seeds):
        r = c2st_single(real_data, generated_data, seed=seed0 + s, **kw)
        accs.append(r["accuracy"])
        runs.append(r)
    accs = np.array(accs)

    if save_path is not None:
        best = runs[int(np.argmax(accs))]
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "input_dim": 1,   # univariate windows: prepare_discriminative_data
                              # reshapes (n, L) -> (n, L, 1)
            "hidden_dim": kw.get("hidden_dim", 64),
            "num_layers": kw.get("num_layers", 1),
            "model_state_dict": best["model"].state_dict(),
            "seed": seed0 + int(np.argmax(accs)),
            "test_accuracy": float(accs.max()),
            "accuracies_all_seeds": accs.tolist(),
        }, save_path)
        print(f"[c2st] saved best-seed classifier "
              f"(acc {accs.max():.4f}) to {save_path}")

    return {
        "accuracy_mean": accs.mean(),
        "accuracy_std": accs.std(ddof=1) if n_seeds > 1 else 0.0,
        "score_mean": np.abs(accs - 0.5).mean(),
        "accuracies": accs,
    }


# ===========================================================================
# Battery: floor + power check + model
# ===========================================================================
def c2st_battery(chunks, generated, model_name="model", device="cpu",
                 n_seeds=3, seed0=42, checkpoint_dir="./checkpoints", **kw):
    """Floor (real vs real), power check (real vs matched Gaussian), and
    model (real vs generated), all through the identical pipeline.

    The classifier from the model-vs-real row (best seed) is saved to
    {checkpoint_dir}/{model_name}_classifier.pt. Floor/Gaussian classifiers
    are throwaway controls and are not saved.

    chunks: the shared disjoint real chunks. Uses chunks[0]/chunks[1] for
    the floor and chunks[2] (or chunks[0] if only 2) as the real side for
    the Gaussian and model tests.
    """
    rng = np.random.default_rng(seed0)
    real_for_tests = chunks[2] if len(chunks) > 2 else chunks[0]
    ref = _to_np(real_for_tests)
    gauss = rng.normal(ref.mean(), ref.std(),
                       size=ref.shape).astype(np.float32)

    ckpt = str(Path(checkpoint_dir) / f"{model_name}_classifier.pt")

    rows = {}
    print(f"{'comparison':<28s} {'acc mean':>9s} {'acc std':>8s} {'|acc-0.5|':>9s}")
    print("-" * 58)
    for label, a, b, sp in [
        ("floor: real vs real", chunks[0], chunks[1], None),
        ("power: real vs Gaussian", real_for_tests, gauss, None),
        (f"model: real vs {model_name}", real_for_tests, generated, ckpt),
    ]:
        res = c2st_score(a, b, n_seeds=n_seeds, seed0=seed0,
                         device=device, save_path=sp, **kw)
        rows[label] = res
        print(f"{label:<28s} {res['accuracy_mean']:>9.4f} "
              f"{res['accuracy_std']:>8.4f} {res['score_mean']:>9.4f}")

    return rows