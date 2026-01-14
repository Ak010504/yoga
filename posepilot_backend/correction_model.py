# correction_model.py
import torch
import torch.nn as nn


class CorrModel(nn.Module):
    """
    Temporal Convolutional Network (TCN) for pose correction.
    Input : (B, T, 9)
    Output: (B, 9)  -> correction for last frame
    """

    def __init__(self, input_size=9, channels=(64, 128, 64), kernel_size=3):
        super().__init__()

        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]

            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], input_size)

    def forward(self, x):
        # x: (B, T, 9)
        x = x.transpose(1, 2)       # (B, 9, T)
        y = self.tcn(x)             # (B, C, T)
        y = y[:, :, -1]             # last timestep
        return self.fc(y)
