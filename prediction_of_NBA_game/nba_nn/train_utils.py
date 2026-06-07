from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    # Total loss equals margin regression loss in the single-task version.
    loss: float
    # Kept for log compatibility with the previous multitask format.
    win_loss: float
    # Margin regression loss.
    margin_loss: float


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    mse_loss = nn.MSELoss()

    total_loss = 0.0
    total_margin_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        categorical_features = batch["categorical_features"].to(device)
        margin_target = batch["margin_target"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(features, categorical_features)
            margin_loss = mse_loss(outputs["margin_pred"], margin_target)
            loss = margin_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_margin_loss += margin_loss.item() * batch_size

    return EpochResult(
        loss=total_loss / total_examples,
        win_loss=0.0,
        margin_loss=total_margin_loss / total_examples,
    )


def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    margin_preds = []
    win_targets = []
    margin_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            categorical_features = batch["categorical_features"].to(device)
            outputs = model(features, categorical_features)

            margin_preds.append(outputs["margin_pred"].cpu().numpy())
            win_targets.append(batch["win_target"].cpu().numpy())
            margin_targets.append(batch["margin_target"].cpu().numpy())

    return {
        "margin_preds": np.vstack(margin_preds).reshape(-1),
        "win_targets": np.vstack(win_targets).reshape(-1),
        "margin_targets": np.vstack(margin_targets).reshape(-1),
    }


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    margin_targets_original: np.ndarray,
    margin_preds_original: np.ndarray,
) -> Dict[str, float]:
    # Derive win/loss directly from the predicted sign of the point margin.
    win_targets = predictions["win_targets"]
    win_scores = margin_preds_original
    win_labels = (margin_preds_original > 0).astype(int)

    metrics = {
        "accuracy": accuracy_score(win_targets, win_labels),
        "mae": mean_absolute_error(margin_targets_original, margin_preds_original),
        "rmse": float(np.sqrt(mean_squared_error(margin_targets_original, margin_preds_original))),
    }

    if len(np.unique(win_targets)) > 1:
        metrics["roc_auc"] = roc_auc_score(win_targets, win_scores)

    return metrics
