import json
import random
from pathlib import Path

import numpy as np
import torch

from vocab import PAD_IDX


def set_seed(seed: int) -> None:
    # 固定 Python、NumPy、PyTorch 随机性，便于复现实验结果。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def make_padding_mask(seq: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    # nn.Transformer 的 padding mask 形状是 [batch, seq_len]，输入张量是 [seq_len, batch]。
    return seq.transpose(0, 1).eq(pad_idx)


def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    # True 表示被遮挡；上三角 mask 阻止 decoder 看到未来 token。
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
