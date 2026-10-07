from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data" / "run_inputs" / "known_mining_neutral_ratio_1_10_supervised_all_features_v1"
OUTPUT_DIR = ROOT / "outputs" / "rebuttal_experiments" / "lin2025_idann_tabular"
MODES = ("source_only", "dan_mmd", "dann", "idann")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class TabularDomainNetwork(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.label_head = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(32, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.25), nn.Linear(32, 1),
        )

    def predict_logits(self, values: torch.Tensor) -> torch.Tensor:
        return self.label_head(self.extractor(values)).squeeze(1)


def mmd_rbf(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    joined = torch.cat([source, target], dim=0)
    squared = torch.cdist(joined, joined).square()
    positive = squared.detach()[squared.detach() > 0]
    bandwidth = positive.median().clamp_min(1e-6) if positive.numel() else squared.new_tensor(1.0)
    kernels = [torch.exp(-squared / (scale * bandwidth)) for scale in (0.5, 1.0, 2.0)]
    n_source = len(source)
    return sum(
        kernel[:n_source, :n_source].mean()
        + kernel[n_source:, n_source:].mean()
        - 2 * kernel[:n_source, n_source:].mean()
        for kernel in kernels
    ) / len(kernels)


def source_smote_style(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = x[y == 1]
    negative_count = int((y == 0).sum())
    needed = negative_count - len(positive)
    if needed <= 0 or len(positive) < 2:
        return x.astype(np.float32), y.astype(np.int64)
    rng = np.random.default_rng(seed)
    neighbours = NearestNeighbors(n_neighbors=min(6, len(positive))).fit(positive)
    indices = neighbours.kneighbors(positive, return_distance=False)
    origin = rng.integers(len(positive), size=needed)
    choice = rng.integers(1, indices.shape[1], size=needed)
    partner = indices[origin, choice]
    interpolation = rng.random((needed, 1))
    synthetic = positive[origin] + interpolation * (positive[partner] - positive[origin])
    return (
        np.concatenate([x, synthetic], axis=0).astype(np.float32),
        np.concatenate([y, np.ones(needed, dtype=np.int64)]),
    )


def top_k(y_true: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    count = max(1, int(math.ceil(len(y_true) * fraction)))
    selected = y_true[np.argsort(-score, kind="stable")[:count]]
    precision = float(selected.mean())
    recall = float(selected.sum() / max(int(y_true.sum()), 1))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def train_one(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[TabularDomainNetwork, list[dict]]:
    set_seed(seed)
    model = TabularDomainNetwork(x_source.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    source_x = torch.from_numpy(x_source)
    source_y = torch.from_numpy(y_source.astype(np.float32))
    target_x = torch.from_numpy(x_target)
    rng = np.random.default_rng(seed)
    steps = math.ceil(len(source_x) / batch_size)
    logs: list[dict] = []

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(source_x))
        totals = np.zeros(3, dtype=float)
        for step in range(steps):
            source_indices = order[step * batch_size : (step + 1) * batch_size]
            if len(source_indices) < 2:
                continue
            xs = source_x[source_indices]
            ys = source_y[source_indices]
            xt = target_x[rng.integers(len(target_x), size=len(source_indices))]
            progress = (epoch * steps + step + 1) / (epochs * steps)
            domain_weight = 2 / (1 + math.exp(-10 * progress)) - 1
            mmd_weight = 2 / (1 + math.exp(10 * progress))

            source_features = model.extractor(xs)
            label_loss = nn.functional.binary_cross_entropy_with_logits(
                model.label_head(source_features).squeeze(1), ys
            )
            loss = label_loss
            mmd_loss = torch.tensor(0.0)
            domain_loss = torch.tensor(0.0)
            if mode != "source_only":
                target_features = model.extractor(xt)
                if mode in {"dan_mmd", "idann"}:
                    mmd_loss = mmd_rbf(source_features, target_features)
                    loss = loss + mmd_weight * mmd_loss
                if mode in {"dann", "idann"}:
                    domain_features = ReverseGradient.apply(
                        torch.cat([source_features, target_features]), domain_weight
                    )
                    domain_labels = torch.cat([torch.zeros(len(xs)), torch.ones(len(xt))])
                    domain_loss = nn.functional.binary_cross_entropy_with_logits(
                        model.domain_head(domain_features).squeeze(1), domain_labels
                    )
                    loss = loss + domain_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += (float(label_loss.detach()), float(mmd_loss.detach()), float(domain_loss.detach()))
        logs.append({
            "epoch": epoch + 1,
            "label_loss": totals[0] / steps,
            "mmd_loss": totals[1] / steps,
            "domain_loss": totals[2] / steps,
        })
    return model, logs


def predict(model: TabularDomainNetwork, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            torch.sigmoid(model.predict_logits(torch.from_numpy(x[start : start + 256]))).numpy()
            for start in range(0, len(x), 256)
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Lin and Zuo (2025) IDANN-inspired tabular adaptation pilot")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--max-folds", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_folds not in range(1, 6):
        parser.error("epochs must be positive and max-folds must be 1..5")
    torch.set_num_threads(min(4, torch.get_num_threads()))

    manifest = json.loads((args.run_dir / "standard_workflow_manifest.json").read_text(encoding="utf-8"))
    input_path = Path(manifest["input_path"])
    df = pd.read_parquet(input_path).reset_index(drop=True)
    df = df[df.Y_label.isin([0, 1])].reset_index(drop=True)
    role_path = args.run_dir / "00_dataset_profile" / "feature_roles.csv"
    roles = pd.read_csv(role_path)
    columns = roles.loc[roles["is_numeric"] & roles["used_in_m2_no_climate"], "column"].tolist()
    if len(columns) != 385 or not set(columns).issubset(df.columns):
        raise ValueError(f"Unexpected M2 feature set: {len(columns)} columns")
    y = df.Y_label.to_numpy(dtype=np.int64)
    groups = df.state.fillna("unknown").astype(str)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    predictions: list[dict] = []
    training_logs: list[dict] = []

    for fold, (source_indices, target_indices) in enumerate(GroupKFold(n_splits=5).split(df, y, groups), start=1):
        if fold > args.max_folds:
            break
        source_values = df.loc[source_indices, columns].to_numpy(dtype=np.float64)
        target_values = df.loc[target_indices, columns].to_numpy(dtype=np.float64)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(source_values)
        source_imputed = imputer.transform(source_values)
        target_imputed = imputer.transform(target_values)
        scaler = StandardScaler().fit(source_imputed)
        source_scaled = np.clip(scaler.transform(source_imputed), -8, 8).astype(np.float32)
        target_scaled = np.clip(scaler.transform(target_imputed), -8, 8).astype(np.float32)
        if not (np.isfinite(source_scaled).all() and np.isfinite(target_scaled).all()):
            raise ValueError(f"Non-finite features in fold {fold}")
        target_states = ";".join(sorted(groups.iloc[target_indices].unique()))
        for seed in args.seeds:
            source_aug, label_aug = source_smote_style(source_scaled, y[source_indices], seed + fold)
            for mode in MODES:
                model, logs = train_one(
                    source_aug, label_aug, target_scaled, mode,
                    args.epochs, args.batch_size, args.learning_rate, seed + fold,
                )
                score = predict(model, target_scaled)
                truth = y[target_indices]
                at5 = top_k(truth, score, 0.05)
                at10 = top_k(truth, score, 0.10)
                metrics.append({
                    "fold": fold, "seed": seed, "test_group": target_states, "model": mode,
                    "n_train": len(source_indices), "n_test": len(target_indices),
                    "positive_test": int(truth.sum()), "n_features": len(columns),
                    "n_source_augmented": len(source_aug), "n_target_unlabeled": len(target_scaled),
                    "roc_auc": float(roc_auc_score(truth, score)),
                    "average_precision": float(average_precision_score(truth, score)),
                    "top05_precision": at5["precision"], "top05_recall": at5["recall"],
                    "top05_f1": at5["f1"], "top10_precision": at10["precision"],
                    "top10_recall": at10["recall"], "top10_f1": at10["f1"],
                })
                for idx, probability in zip(target_indices, score):
                    predictions.append({
                        "row_index": int(idx), "sample_id": str(df.iloc[idx].sample_id),
                        "state": str(groups.iloc[idx]), "fold": fold, "seed": seed,
                        "model": mode, "Y_label": int(y[idx]), "y_prob": float(probability),
                    })
                training_logs.extend({"fold": fold, "seed": seed, "model": mode, **entry} for entry in logs)
                print(
                    f"fold={fold} seed={seed} mode={mode} AP={metrics[-1]['average_precision']:.4f} "
                    f"F1@5={metrics[-1]['top05_f1']:.4f}", flush=True
                )

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(training_logs).to_csv(args.output_dir / "training_log.csv", index=False)
    summary = metric_frame.groupby("model", as_index=False)[
        ["roc_auc", "average_precision", "top05_precision", "top05_recall", "top05_f1", "top10_f1"]
    ].agg(["mean", "std"])
    summary.columns = ["model" if column[0] == "model" else "_".join(column) for column in summary.columns]
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    run_manifest = {
        "reference": "Lin and Zuo (2025), DOI 10.1007/s11004-024-10164-3",
        "status": "IDANN-inspired tabular adaptation; not an exact paper reproduction",
        "input_path": str(input_path), "input_sha256": sha256(input_path),
        "roles_path": str(role_path), "roles_sha256": sha256(role_path),
        "reference_note": "The article PDF is not redistributed; use the DOI above.",
        "n_samples": len(df), "n_positives": int(y.sum()), "n_features": len(columns),
        "cv": "GroupKFold(n_splits=5, groups=state)", "modes": list(MODES),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seeds": args.seeds, "max_folds": args.max_folds,
        "architecture": "MLP 385-128-64-32; label head 32-64-32-1; domain head 32-32-1",
        "adaptations": [
            "Tabular MLP replaces original 7x7x42 patch CNN because patch rasters are unavailable.",
            "Train-fold-only k-nearest-neighbor linear interpolation replaces DeepSMOTE.",
            "Target samples are used as unlabeled transductive inputs only; no target labels enter training.",
            "Source-fitted median imputation, standardization, and clipping to [-8,8].",
            "Fixed epochs and hyperparameters; no held-out labels used for tuning.",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
