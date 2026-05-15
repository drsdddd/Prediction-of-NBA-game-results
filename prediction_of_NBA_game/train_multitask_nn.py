from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nba_nn.data import (
    NBAGameDataset,
    build_feature_frame,
    chronological_split,
    fit_scalers,
    load_processed_games,
    save_preprocessors,
    transform_split,
)
from nba_nn.model import MultitaskNBAPredictor
from nba_nn.train_utils import compute_metrics, predict, run_epoch


DATA_PATH = Path("data_preparation/data/processed/features_2023_24.csv")
ARTIFACT_DIR = Path("artifacts/multitask_nn")


def main() -> None:
    # 训练入口：读取数据、切分、训练、评估、保存产物
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = load_processed_games(DATA_PATH)
    df, feature_cols = build_feature_frame(df)
    # 必须按时间切分，不能随机打乱
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.7, val_ratio=0.15)

    feature_scaler, margin_scaler = fit_scalers(train_df, feature_cols)
    train_split = transform_split(train_df, feature_cols, feature_scaler, margin_scaler)
    val_split = transform_split(val_df, feature_cols, feature_scaler, margin_scaler)
    test_split = transform_split(test_df, feature_cols, feature_scaler, margin_scaler)

    train_loader = DataLoader(NBAGameDataset(train_split), batch_size=32, shuffle=True)
    val_loader = DataLoader(NBAGameDataset(val_split), batch_size=64, shuffle=False)
    test_loader = DataLoader(NBAGameDataset(test_split), batch_size=64, shuffle=False)

    # 有 GPU 就优先用 GPU，没有就自动退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskNBAPredictor(input_dim=len(feature_cols), hidden_dim=64, dropout=0.2).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # 两个任务共享训练，但分类任务权重略高一些
    classification_weight = 1.0
    regression_weight = 0.5
    max_epochs = 100
    patience = 12
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    # 净胜分在标准化空间里训练，回归头会更稳定；输出时再还原成真实分差
    for epoch in range(1, max_epochs + 1):
        train_result = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            classification_weight=classification_weight,
            regression_weight=regression_weight,
        )
        val_result = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            device=device,
            classification_weight=classification_weight,
            regression_weight=regression_weight,
        )

        scheduler.step(val_result.loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_result.loss:.4f} "
            f"val_loss={val_result.loss:.4f} "
            f"train_win_loss={train_result.win_loss:.4f} "
            f"val_win_loss={val_result.win_loss:.4f} "
            f"train_margin_loss={train_result.margin_loss:.4f} "
            f"val_margin_loss={val_result.margin_loss:.4f}"
        )

        if val_result.loss < best_val_loss:
            # 只保留验证集表现最好的那一版参数
            best_val_loss = val_result.loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存模型结构参数和权重，后续推理时直接加载
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_cols),
            "hidden_dim": 64,
            "dropout": 0.2,
        },
        ARTIFACT_DIR / "multitask_model.pt",
    )
    save_preprocessors(ARTIFACT_DIR, feature_scaler, margin_scaler, feature_cols)

    val_predictions = predict(model, val_loader, device)
    test_predictions = predict(model, test_loader, device)

    # 将标准化后的净胜分预测还原回原始比赛分差
    val_margin_pred = margin_scaler.inverse_transform(val_predictions["margin_preds"].reshape(-1, 1)).reshape(-1)
    val_margin_true = val_df["TARGET_PLUS_MINUS_HOME"].values
    test_margin_pred = margin_scaler.inverse_transform(test_predictions["margin_preds"].reshape(-1, 1)).reshape(-1)
    test_margin_true = test_df["TARGET_PLUS_MINUS_HOME"].values

    val_metrics = compute_metrics(val_predictions, val_margin_true, val_margin_pred)
    test_metrics = compute_metrics(test_predictions, test_margin_true, test_margin_pred)

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "num_features": len(feature_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 保存一份可直接查看的测试集预测结果
    test_preview = test_df[
        [
            "GAME_DATE",
            "TEAM_ABBREVIATION_HOME",
            "TEAM_ABBREVIATION_AWAY",
            "TARGET_WIN_HOME",
            "TARGET_PLUS_MINUS_HOME",
        ]
    ].copy()
    test_preview["PRED_WIN_PROB_HOME"] = test_predictions["win_probs"]
    test_preview["PRED_PLUS_MINUS_HOME"] = test_margin_pred
    test_preview.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
