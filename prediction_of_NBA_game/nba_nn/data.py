from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


TARGET_WIN_COL = "TARGET_WIN_HOME"
TARGET_MARGIN_COL = "TARGET_PLUS_MINUS_HOME"


@dataclass
class SplitData:
    # 标准化后的输入特征
    features: np.ndarray
    # 胜负标签：主队赢=1，输=0
    win_target: np.ndarray
    # 标准化后的净胜分标签
    margin_target: np.ndarray


class NBAGameDataset(Dataset):
    """用于表格比赛特征的 PyTorch 数据集。"""

    def __init__(self, split_data: SplitData) -> None:
        # 提前转成 tensor，训练时读取更直接
        self.features = torch.tensor(split_data.features, dtype=torch.float32)
        self.win_target = torch.tensor(split_data.win_target, dtype=torch.float32).view(-1, 1)
        self.margin_target = torch.tensor(split_data.margin_target, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "win_target": self.win_target[index],
            "margin_target": self.margin_target[index],
        }


def load_processed_games(csv_path: str | Path) -> pd.DataFrame:
    # 读取处理后的比赛级数据，并按时间排序
    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE").reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """构造可直接建模的特征表，并返回最终使用的特征列名。"""
    feature_df = df.copy()

    rolling_metrics = [
        "PTS_5G_AVG",
        "FG_PCT_5G_AVG",
        "FG3_PCT_5G_AVG",
        "FT_PCT_5G_AVG",
        "REB_5G_AVG",
        "AST_5G_AVG",
        "TOV_5G_AVG",
        "PLUS_MINUS_5G_AVG",
        "WIN_5G_AVG",
    ]

    # 增加主客队差值特征，让模型更容易直接学习“对位差距”
    for metric in rolling_metrics:
        home_col = f"{metric}_HOME"
        away_col = f"{metric}_AWAY"
        diff_col = f"{metric}_DIFF"
        feature_df[diff_col] = feature_df[home_col] - feature_df[away_col]

    feature_cols = [
        col
        for col in feature_df.columns
        if col not in {
            "GAME_ID",
            "GAME_DATE",
            "GAME_DATE_AWAY",
            "TEAM_ABBREVIATION_HOME",
            "TEAM_ABBREVIATION_AWAY",
            TARGET_WIN_COL,
            TARGET_MARGIN_COL,
        }
    ]

    # 第一版模型只喂数值列，避免把球队缩写这类字符串直接送进网络
    numeric_feature_cols = feature_df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    return feature_df, numeric_feature_cols


def add_matchup_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """应用与训练阶段完全一致的差值特征逻辑。"""
    feature_df = df.copy()
    rolling_metrics = [
        "PTS_5G_AVG",
        "FG_PCT_5G_AVG",
        "FG3_PCT_5G_AVG",
        "FT_PCT_5G_AVG",
        "REB_5G_AVG",
        "AST_5G_AVG",
        "TOV_5G_AVG",
        "PLUS_MINUS_5G_AVG",
        "WIN_5G_AVG",
    ]

    for metric in rolling_metrics:
        home_col = f"{metric}_HOME"
        away_col = f"{metric}_AWAY"
        diff_col = f"{metric}_DIFF"
        feature_df[diff_col] = feature_df[home_col] - feature_df[away_col]

    return feature_df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 按时间顺序切分，避免未来比赛信息泄漏到过去
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[StandardScaler, StandardScaler]:
    # scaler 只在训练集上拟合，验证/测试只能复用
    feature_scaler = StandardScaler()
    margin_scaler = StandardScaler()

    feature_scaler.fit(train_df[feature_cols].values)
    margin_scaler.fit(train_df[[TARGET_MARGIN_COL]].values)
    return feature_scaler, margin_scaler


def transform_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    feature_scaler: StandardScaler,
    margin_scaler: StandardScaler,
) -> SplitData:
    # 特征标准化；净胜分也标准化，能让回归头训练更稳定
    features = feature_scaler.transform(df[feature_cols].values)
    win_target = df[TARGET_WIN_COL].astype(np.float32).values
    margin_target = margin_scaler.transform(df[[TARGET_MARGIN_COL]].values).astype(np.float32).reshape(-1)
    return SplitData(features=features, win_target=win_target, margin_target=margin_target)


def save_preprocessors(
    output_dir: str | Path,
    feature_scaler: StandardScaler,
    margin_scaler: StandardScaler,
    feature_cols: List[str],
) -> None:
    # 保存预处理器和特征列，方便后续推理时严格复现训练输入
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(feature_scaler, output_path / "feature_scaler.joblib")
    joblib.dump(margin_scaler, output_path / "margin_scaler.joblib")
    joblib.dump(feature_cols, output_path / "feature_columns.joblib")
