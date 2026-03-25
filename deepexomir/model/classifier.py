"""Classification head with temperature scaling (Platt scaling).

The classifier maps the concatenated feature vector from all upstream
encoders into a 2-class prediction.  A learnable temperature parameter
enables post-hoc calibration of the output probabilities without
retraining the full model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Final classification head with optional Platt (temperature) scaling.

    Architecture::

        Linear(in_dim, 256) -> GELU -> Dropout(0.2)
        Linear(256, 128)    -> GELU -> Dropout(0.1)
        Linear(128, n_classes)
        (optional) temperature scaling on logits

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input feature vector (default: 704).
        This should equal  ``d_model * 2  +  bp_cnn_out  +  struct_mlp_out``
        = 256*2 + 128 + 64 = 704 with default config.
    hidden_dims : list[int]
        Sizes of hidden layers (default: ``[256, 128]``).
    n_classes : int
        Number of output classes (default: 2).
    dropout_rates : list[float]
        Dropout rate after each hidden layer (default: ``[0.2, 0.1]``).
        Must have the same length as *hidden_dims*.
    platt_scaling : bool
        If ``True`` (default), include a learnable temperature parameter
        for probability calibration.
    init_temperature : float
        Initial value of the temperature parameter (default: 1.0).
    """

    def __init__(
        self,
        in_dim: int = 704,
        hidden_dims: list[int] | None = None,
        n_classes: int = 2,
        dropout_rates: list[float] | None = None,
        platt_scaling: bool = True,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]
        if dropout_rates is None:
            dropout_rates = [0.2, 0.1]
        assert len(hidden_dims) == len(dropout_rates), (
            "hidden_dims and dropout_rates must have the same length"
        )

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim, drop_p in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.Dropout(drop_p),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.mlp = nn.Sequential(*layers)
        self.n_classes = n_classes

        # Temperature scaling
        self.platt_scaling = platt_scaling
        if platt_scaling:
            # nn.Parameter so it is saved/loaded with the model but can be
            # optimised separately during calibration.
            self.temperature = nn.Parameter(
                torch.tensor(init_temperature, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "temperature",
                torch.tensor(1.0, dtype=torch.float32),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits (optionally temperature-scaled).

        Parameters
        ----------
        x : Tensor [B, in_dim]
            Concatenated feature vector.

        Returns
        -------
        Tensor [B, n_classes]
            Un-normalised logits.  Apply ``F.softmax`` (or ``F.sigmoid``
            for the positive-class column) to get probabilities.
        """
        logits = self.mlp(x)  # [B, n_classes]

        # Temperature scaling: divide logits by T (T > 1 softens, T < 1 sharpens)
        logits = logits / self.temperature.clamp(min=1e-6)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated class probabilities.

        Parameters
        ----------
        x : Tensor [B, in_dim]

        Returns
        -------
        Tensor [B, n_classes]
            Softmax probabilities.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def set_temperature(self, value: float) -> None:
        """Manually set the temperature for calibration.

        This is useful after running a calibration procedure on a
        held-out validation set (e.g., minimising NLL on validation logits).
        """
        with torch.no_grad():
            self.temperature.fill_(value)
