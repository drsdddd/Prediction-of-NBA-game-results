from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nba_nn.classification_utils import (
    compute_classification_metrics,
    compute_classification_metrics_with_threshold,
    find_best_accuracy_threshold,
    predict_classification,
    run_classification_epoch,
)
from nba_nn.data import (
    NBAGameDataset,
    build_category_maps,
    build_feature_frame,
    fit_scalers,
    load_processed_games,
    save_preprocessors,
    seasonwise_time_split,
    transform_split,
)
from nba_nn.model import WinNBAPredictor


DATA_PATH = Path("data_preparation/data/processed/features_multi_seasons.csv")
ARTIFACT_DIR = Path("artifacts/win_classifier")

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = load_processed_games(DATA_PATH)
    df, feature_cols = build_feature_frame(df)
    train_df, val_df, test_df = seasonwise_time_split(df, train_ratio=0.9, val_ratio=0.05)

    feature_scaler, margin_scaler = fit_scalers(train_df, feature_cols)
    category_maps = build_category_maps(train_df)
    train_split = transform_split(train_df, feature_cols, feature_scaler, margin_scaler, category_maps)
    val_split = transform_split(val_df, feature_cols, feature_scaler, margin_scaler, category_maps)
    test_split = transform_split(test_df, feature_cols, feature_scaler, margin_scaler, category_maps)

    train_loader = DataLoader(NBAGameDataset(train_split), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NBAGameDataset(val_split), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(NBAGameDataset(test_split), batch_size=EVAL_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WinNBAPredictor(
        input_dim=len(feature_cols),
        num_home_teams=len(category_maps["TEAM_ID_HOME"]),
        num_away_teams=len(category_maps["TEAM_ID_AWAY"]),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_result = run_classification_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_result = run_classification_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            device=device,
        )

        scheduler.step(val_result.loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_result.loss:.4f} "
            f"val_loss={val_result.loss:.4f}"
        )

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_cols),
            "num_home_teams": len(category_maps["TEAM_ID_HOME"]),
            "num_away_teams": len(category_maps["TEAM_ID_AWAY"]),
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "task_type": "win_classification",
        },
        ARTIFACT_DIR / "win_classifier_model.pt",
    )
    save_preprocessors(ARTIFACT_DIR, feature_scaler, margin_scaler, feature_cols, category_maps)

    val_predictions = predict_classification(model, val_loader, device)
    test_predictions = predict_classification(model, test_loader, device)

    val_metrics_default = compute_classification_metrics(val_predictions)
    best_threshold, best_val_accuracy = find_best_accuracy_threshold(val_predictions)
    val_metrics = compute_classification_metrics_with_threshold(val_predictions, threshold=best_threshold)
    test_metrics = compute_classification_metrics_with_threshold(test_predictions, threshold=best_threshold)

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_threshold": best_threshold,
        "best_val_accuracy_at_threshold": best_val_accuracy,
        "num_features": len(feature_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "task_type": "win_classification",
        "val_metrics_default_threshold": val_metrics_default,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    test_preview = test_df[
        [
            "GAME_DATE",
            "TEAM_ABBREVIATION_HOME",
            "TEAM_ABBREVIATION_AWAY",
            "TARGET_WIN_HOME",
        ]
    ].copy()
    test_preview["PRED_WIN_PROB_HOME"] = test_predictions["win_probs"]
    test_preview["PRED_WIN_HOME"] = (test_predictions["win_probs"] >= best_threshold).astype(int)
    test_preview.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation threshold: {best_threshold:.4f}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
