from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class ClassificationEpochResult:
    loss: float


def run_classification_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> ClassificationEpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        categorical_features = batch["categorical_features"].to(device)
        win_target = batch["win_target"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(features, categorical_features)
            loss = bce_loss(outputs["win_logits"], win_target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size

    return ClassificationEpochResult(loss=total_loss / total_examples)


def predict_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    win_probs = []
    win_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            categorical_features = batch["categorical_features"].to(device)
            outputs = model(features, categorical_features)

            win_probs.append(torch.sigmoid(outputs["win_logits"]).cpu().numpy())
            win_targets.append(batch["win_target"].cpu().numpy())

    return {
        "win_probs": np.vstack(win_probs).reshape(-1),
        "win_targets": np.vstack(win_targets).reshape(-1),
    }


def compute_classification_metrics(predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    return compute_classification_metrics_with_threshold(predictions, threshold=0.5)


def find_best_accuracy_threshold(predictions: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Choose the validation threshold that maximizes accuracy."""
    win_targets = predictions["win_targets"].astype(int)
    win_probs = predictions["win_probs"]

    candidate_thresholds = np.unique(np.round(win_probs, 4))
    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in candidate_thresholds:
        win_labels = (win_probs >= threshold).astype(int)
        accuracy = accuracy_score(win_targets, win_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, float(best_accuracy)


def compute_classification_metrics_with_threshold(
    predictions: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    win_targets = predictions["win_targets"]
    win_probs = predictions["win_probs"]
    win_labels = (win_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(win_targets, win_labels),
        "log_loss": log_loss(win_targets, win_probs, labels=[0, 1]),
        "threshold": float(threshold),
    }

    if len(np.unique(win_targets)) > 1:
        metrics["roc_auc"] = roc_auc_score(win_targets, win_probs)

    return metrics
