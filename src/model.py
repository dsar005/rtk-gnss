import torch
from torch import nn

class NavigationQualityNet(nn.Module):
    """Temporal CNN extracts local sensor patterns; GRU models their time evolution."""
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(),
                                    nn.Dropout(0.2))
        self.error_head = nn.Linear(64, 1)
        self.quality_head = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: batch x time x features
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, h = self.gru(z)
        h = self.shared(h[-1])
        error = torch.relu(self.error_head(h)).squeeze(-1)
        logits = self.quality_head(h)
        return error, logits
