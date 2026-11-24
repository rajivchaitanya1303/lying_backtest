"""
LSTM-VaR via quantile regression.

Single-layer LSTM (hidden_dim=64) predicts the alpha-quantile of the next
day's return from the trailing `seq_len` returns. Trained by minimizing
the pinball (quantile) loss.

Deliberately simple — this is the "ML entry," not a showcase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- module-level lazy import so the file loads without torch --------
def _torch():
    import torch, torch.nn as nn
    return torch, nn


class _LSTMQuantile:
    """Thin nn.Module wrapper built lazily so torch is optional at import."""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 1):
        torch, nn = _torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                    batch_first=True)
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = Net()

    def __call__(self, x):
        return self.net(x)


def _pinball_loss(pred, target, alpha):
    torch, _ = _torch()
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1) * diff))


class LstmVaR:
    """
    Parameters
    ----------
    alpha : float
        Tail probability (0.01 or 0.025).
    seq_len : int
        Look-back window fed into the LSTM.
    hidden_dim : int
        LSTM hidden state size.
    epochs, lr, batch_size, seed : training hyperparameters.
    device : "cuda" or "cpu" (auto-detect if None).
    """
    def __init__(self, alpha: float = 0.01, seq_len: int = 60,
                 hidden_dim: int = 64, epochs: int = 200,
                 lr: float = 1e-3, batch_size: int = 64, seed: int = 0,
                 device: str | None = None):
        self.alpha = float(alpha)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = device

    # -- helpers ---------------------------------------------------------
    def _build_windows(self, returns: pd.Series):
        r = returns.values.astype(np.float32)
        X, y = [], []
        for i in range(self.seq_len, len(r)):
            X.append(r[i - self.seq_len:i])
            y.append(r[i])
        X = np.stack(X)[..., None]     # (N, seq, 1)
        y = np.array(y, dtype=np.float32)
        return X, y

    def _pick_device(self):
        torch, _ = _torch()
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- API used by walk-forward harness --------------------------------
    def fit(self, train_returns: pd.Series) -> "LstmVaR":
        torch, nn = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X, y = self._build_windows(train_returns)
        device = self._pick_device()
        self._device = device

        model = _LSTMQuantile(1, self.hidden_dim, 1).net.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        n = len(Xt)

        model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                pred = model(Xt[idx])
                loss = _pinball_loss(pred, yt[idx], self.alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._model = model
        return self

    def forecast(self, test_returns: pd.Series,
                 train_returns: pd.Series) -> pd.Series:
        torch, _ = _torch()
        model = self._model
        model.eval()

        combined = pd.concat([train_returns, test_returns])
        r = combined.values.astype(np.float32)
        offset = len(train_returns)   # first test index in combined

        var_out = np.full(len(test_returns), np.nan)
        with torch.no_grad():
            for j in range(len(test_returns)):
                pos = offset + j
                if pos < self.seq_len:
                    continue
                window = r[pos - self.seq_len:pos][None, :, None]
                xt = torch.from_numpy(window).to(self._device)
                var_out[j] = float(model(xt).cpu().numpy()[0])
        return pd.Series(var_out, index=test_returns.index, name=self.name)

    @property
    def name(self) -> str:
        return f"LSTM(h={self.hidden_dim},L={self.seq_len})"
