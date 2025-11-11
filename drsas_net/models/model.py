from __future__ import annotations
import torch
import torch.nn as nn

class DRSASNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Placeholder CNN — replace with real DRSAS-Net
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_model(name: str = "DRSASNet", **kwargs):
    name = name.lower()
    if name in {"drsasnet", "drsas-net", "drsas"}:
        return DRSASNet(**kwargs)
    raise ValueError(f"Unknown model: {name}")
