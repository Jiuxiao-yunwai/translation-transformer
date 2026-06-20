import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 使用原始 Transformer 的正弦/余弦位置编码；注册为 buffer，随模型迁移设备但不参与训练。
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x 的形状是 [seq_len, batch, d_model]，只按当前序列长度截取位置编码。
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
