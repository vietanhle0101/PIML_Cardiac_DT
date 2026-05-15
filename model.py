import torch
from torch import nn


class MLP_Net(nn.Module):
    """Simple tanh MLP for NN(x, y, t) -> (vm, h)."""

    def __init__(
        self,
        in_features=3,
        hidden_widths=None,
        out_features=2,
    ):
        super().__init__()
        if hidden_widths is None:
            hidden_widths = [64, 64, 64]
        if len(hidden_widths) == 0:
            raise ValueError("hidden_widths must contain at least one width.")

        layers = []
        previous_width = in_features

        for width in hidden_widths:
            layers.append(nn.Linear(previous_width, width))
            layers.append(nn.Tanh())
            previous_width = width

        layers.append(nn.Linear(previous_width, out_features))
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, coords):
        raw = self.net(coords)
        vm = torch.sigmoid(raw[:, 0:1])
        h = torch.sigmoid(raw[:, 1:2])
        return torch.cat([vm, h], dim=1)
