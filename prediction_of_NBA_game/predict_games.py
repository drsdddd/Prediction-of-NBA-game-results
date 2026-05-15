from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch

from nba_nn.data import add_matchup_difference_features
from nba_nn.model import MultitaskNBAPredictor


DEFAULT_INPUT = Path("data_preparation/data/processed/features_2023_24.csv")
DEFAULT_ARTIFACT_DIR = Path("artifacts/multitask_nn")
DEFAULT_OUTPUT = Path("artifacts/multitask_nn/predictions.csv")


def parse_args() -> argparse.Namespace:
    # 支持命令行指定输入文件、模型目录和输出文件
    parser = argparse.ArgumentParser(description="Run trained NBA multitask model on matchup features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a matchup feature CSV.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing trained model and preprocessors.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save predictions.")
    return parser.parse_args()


def load_model(artifact_dir: Path, device: torch.device) -> MultitaskNBAPredictor:
    # 按训练时记录的结构参数重建模型，再加载权重
    checkpoint = torch.load(artifact_dir / "multitask_model.pt", map_location=device)
    model = MultitaskNBAPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    # 推理入口：加载产物、对输入 CSV 做同样的特征处理并输出预测
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not args.artifacts.exists():
        raise FileNotFoundError(f"Artifact directory not found: {args.artifacts}")

    feature_scaler = joblib.load(args.artifacts / "feature_scaler.joblib")
    margin_scaler = joblib.load(args.artifacts / "margin_scaler.joblib")
    feature_cols = joblib.load(args.artifacts / "feature_columns.joblib")

    df = pd.read_csv(args.input)
    # 预测阶段也要补出差值特征，否则和训练输入不一致
    feature_df = add_matchup_difference_features(df)

    missing_cols = [col for col in feature_cols if col not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required feature columns: {missing_cols}")

    # 严格按训练时的特征顺序取列并做标准化
    feature_matrix = feature_scaler.transform(feature_df[feature_cols].values)
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.artifacts, device)

    with torch.no_grad():
        outputs = model(feature_tensor.to(device))
        win_prob = torch.sigmoid(outputs["win_logits"]).cpu().numpy().reshape(-1)
        margin_pred_scaled = outputs["margin_pred"].cpu().numpy().reshape(-1, 1)

    # 回归输出从标准化空间还原为真实净胜分
    margin_pred = margin_scaler.inverse_transform(margin_pred_scaled).reshape(-1)

    result_df = df.copy()
    result_df["PRED_WIN_PROB_HOME"] = win_prob
    result_df["PRED_WIN_HOME"] = (win_prob >= 0.5).astype(int)
    result_df["PRED_PLUS_MINUS_HOME"] = margin_pred

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    # 如果基础信息列存在，就打印一个便于人工检查的预览
    preview_cols = [
        col
        for col in [
            "GAME_DATE",
            "TEAM_ABBREVIATION_HOME",
            "TEAM_ABBREVIATION_AWAY",
            "PRED_WIN_PROB_HOME",
            "PRED_WIN_HOME",
            "PRED_PLUS_MINUS_HOME",
        ]
        if col in result_df.columns
    ]

    print(f"Saved predictions to: {args.output}")
    if preview_cols:
        print(result_df[preview_cols].head())


if __name__ == "__main__":
    main()
