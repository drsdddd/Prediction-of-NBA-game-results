from __future__ import annotations

import torch
from torch import nn


class MarginNBAPredictor(nn.Module):
    """Tabular backbone with learned team/season embeddings plus numeric features."""

    def __init__(
        self,
        input_dim: int,
        num_home_teams: int,
        num_away_teams: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        team_embedding_dim: int = 8,
    ) -> None:
        super().__init__()

        self.home_team_embedding = nn.Embedding(num_home_teams, team_embedding_dim)
        self.away_team_embedding = nn.Embedding(num_away_teams, team_embedding_dim)

        embedding_dim = team_embedding_dim * 2
        total_input_dim = input_dim + embedding_dim

        self.backbone = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.margin_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, categorical_x: torch.Tensor) -> dict[str, torch.Tensor]:
        home_team_idx = categorical_x[:, 0]
        away_team_idx = categorical_x[:, 1]

        embeddings = torch.cat(
            [
                self.home_team_embedding(home_team_idx),
                self.away_team_embedding(away_team_idx),
            ],
            dim=1,
        )

        model_input = torch.cat([x, embeddings], dim=1)
        shared = self.backbone(model_input)
        return {"margin_pred": self.margin_head(shared)}


class WinNBAPredictor(nn.Module):
    """Tabular backbone with learned team embeddings for win classification."""

    def __init__(
        self,
        input_dim: int,
        num_home_teams: int,
        num_away_teams: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        team_embedding_dim: int = 8,
    ) -> None:
        super().__init__()

        self.home_team_embedding = nn.Embedding(num_home_teams, team_embedding_dim)
        self.away_team_embedding = nn.Embedding(num_away_teams, team_embedding_dim)

        embedding_dim = team_embedding_dim * 2
        total_input_dim = input_dim + embedding_dim

        self.backbone = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.win_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, categorical_x: torch.Tensor) -> dict[str, torch.Tensor]:
        home_team_idx = categorical_x[:, 0]
        away_team_idx = categorical_x[:, 1]

        embeddings = torch.cat(
            [
                self.home_team_embedding(home_team_idx),
                self.away_team_embedding(away_team_idx),
            ],
            dim=1,
        )

        model_input = torch.cat([x, embeddings], dim=1)
        shared = self.backbone(model_input)
        return {"win_logits": self.win_head(shared)}
