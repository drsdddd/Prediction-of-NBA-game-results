from __future__ import annotations

import torch
from torch import nn


class MultitaskNBAPredictor(nn.Module):
    """共享主干网络，同时输出胜负分类和净胜分回归。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        # 共享主干先提取比赛对位的公共表示
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        # 分类头：输出主队获胜 logits
        self.win_head = nn.Linear(hidden_dim // 2, 1)
        # 回归头：输出主队净胜分
        self.margin_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.backbone(x)
        return {
            "win_logits": self.win_head(shared),
            "margin_pred": self.margin_head(shared),
        }
