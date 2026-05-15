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
    # 分类损失和回归损失加权后的总损失
    loss: float
    # 胜负分类损失
    win_loss: float
    # 净胜分回归损失
    margin_loss: float


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    classification_weight: float,
    regression_weight: float,
) -> EpochResult:
    # optimizer 为 None 时表示验证阶段，只前向不更新参数
    is_train = optimizer is not None
    model.train(is_train)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    total_win_loss = 0.0
    total_margin_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        win_target = batch["win_target"].to(device)
        margin_target = batch["margin_target"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(features)
            # 分类头直接接 logits，所以这里用 BCEWithLogitsLoss
            win_loss = bce_loss(outputs["win_logits"], win_target)
            margin_loss = mse_loss(outputs["margin_pred"], margin_target)
            loss = classification_weight * win_loss + regression_weight * margin_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_win_loss += win_loss.item() * batch_size
        total_margin_loss += margin_loss.item() * batch_size

    return EpochResult(
        loss=total_loss / total_examples,
        win_loss=total_win_loss / total_examples,
        margin_loss=total_margin_loss / total_examples,
    )


def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    # 收集整批预测结果，供验证和测试指标计算
    model.eval()
    win_probs = []
    margin_preds = []
    win_targets = []
    margin_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            outputs = model(features)

            win_probs.append(torch.sigmoid(outputs["win_logits"]).cpu().numpy())
            margin_preds.append(outputs["margin_pred"].cpu().numpy())
            win_targets.append(batch["win_target"].cpu().numpy())
            margin_targets.append(batch["margin_target"].cpu().numpy())

    return {
        "win_probs": np.vstack(win_probs).reshape(-1),
        "margin_preds": np.vstack(margin_preds).reshape(-1),
        "win_targets": np.vstack(win_targets).reshape(-1),
        "margin_targets": np.vstack(margin_targets).reshape(-1),
    }


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    margin_targets_original: np.ndarray,
    margin_preds_original: np.ndarray,
) -> Dict[str, float]:
    # 以 0.5 为阈值将胜率转成二分类预测
    win_targets = predictions["win_targets"]
    win_probs = predictions["win_probs"]
    win_labels = (win_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(win_targets, win_labels),
        "mae": mean_absolute_error(margin_targets_original, margin_preds_original),
        "rmse": float(np.sqrt(mean_squared_error(margin_targets_original, margin_preds_original))),
    }

    # AUC 要求当前数据切片里同时存在正负样本
    if len(np.unique(win_targets)) > 1:
        metrics["roc_auc"] = roc_auc_score(win_targets, win_probs)

    return metrics
