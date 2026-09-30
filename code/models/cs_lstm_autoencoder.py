from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CSLSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_seq, _ = self.encoder(x)
        latent = self.to_latent(encoded_seq[:, -1, :])
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded_seq, _ = self.decoder(repeated)
        recon = self.output_layer(decoded_seq)
        return recon


@dataclass
class CSLSTMResult:
    model: CSLSTMAutoencoder
    train_losses: Dict[str, float]
    train_scores: np.ndarray
    test_scores: np.ndarray
    threshold_99: float


def _score_sequences(model: nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.tensor(sequences, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model(x)
        errors = torch.mean((x - recon) ** 2, dim=(1, 2))
    return errors.detach().cpu().numpy()


def train_cs_lstm_ae(
    train_sequences: np.ndarray,
    test_sequences: np.ndarray,
    input_dim: int,
    seed: int = 42,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 80,
) -> CSLSTMResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSLSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    train_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        last_loss = epoch_loss / max(len(loader), 1)

    train_scores = _score_sequences(model, train_sequences, device)
    test_scores = _score_sequences(model, test_sequences, device)
    threshold = float(np.quantile(train_scores, 0.99))

    return CSLSTMResult(
        model=model,
        train_losses={"final_epoch_loss": last_loss},
        train_scores=train_scores,
        test_scores=test_scores,
        threshold_99=threshold,
    )
