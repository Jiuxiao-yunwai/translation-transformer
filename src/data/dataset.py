import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from utils.constants import BOS_ID, EOS_ID, PAD_ID, UNK_ID


def load_vocab(data_dir: str | Path) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    data_dir = Path(data_dir)
    with (data_dir / "word2int_en.json").open(encoding="utf-8") as f:
        src_word2id = json.load(f)
    with (data_dir / "int2word_en.json").open(encoding="utf-8") as f:
        src_id2word = {int(k): v for k, v in json.load(f).items()}
    with (data_dir / "word2int_cn.json").open(encoding="utf-8") as f:
        tgt_word2id = json.load(f)
    with (data_dir / "int2word_cn.json").open(encoding="utf-8") as f:
        tgt_id2word = {int(k): v for k, v in json.load(f).items()}
    return src_word2id, src_id2word, tgt_word2id, tgt_id2word


def encode_tokens(tokens: list[str], vocab: dict[str, int], add_bos: bool = False) -> list[int]:
    ids = [vocab.get(token, UNK_ID) for token in tokens]
    if add_bos:
        ids = [BOS_ID] + ids
    return ids + [EOS_ID]


def ids_to_tokens(ids: list[int], id2word: dict[int, str]) -> list[str]:
    tokens: list[str] = []
    for idx in ids:
        if idx in (PAD_ID, BOS_ID):
            continue
        if idx == EOS_ID:
            break
        tokens.append(id2word.get(idx, "<UNK>"))
    return tokens


def detokenize_cn(tokens: list[str]) -> str:
    return "".join(token for token in tokens if token not in {"<PAD>", "<BOS>", "<EOS>"})


class TranslationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        src_word2id: dict[str, int],
        tgt_word2id: dict[str, int],
        max_src_len: int | None = None,
        max_tgt_len: int | None = None,
    ) -> None:
        self.samples: list[tuple[list[int], list[int], list[str], list[str]]] = []
        path = Path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            src_text, tgt_text = line.split("\t")
            src_tokens = src_text.split()
            tgt_tokens = tgt_text.split()
            src_ids = encode_tokens(src_tokens, src_word2id)
            tgt_ids = encode_tokens(tgt_tokens, tgt_word2id, add_bos=True)
            if max_src_len is not None and len(src_ids) > max_src_len:
                continue
            if max_tgt_len is not None and len(tgt_ids) > max_tgt_len:
                continue
            self.samples.append((src_ids, tgt_ids, src_tokens, tgt_tokens))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], list[str], list[str]]:
        return self.samples[index]


def make_collate_fn(pad_id: int = PAD_ID) -> Callable:
    def collate(batch: list[tuple[list[int], list[int], list[str], list[str]]]) -> dict[str, object]:
        src_ids, tgt_ids, src_tokens, tgt_tokens = zip(*batch)
        src_max_len = max(len(x) for x in src_ids)
        tgt_max_len = max(len(x) for x in tgt_ids)

        src = torch.full((len(batch), src_max_len), pad_id, dtype=torch.long)
        tgt = torch.full((len(batch), tgt_max_len), pad_id, dtype=torch.long)
        for i, (src_item, tgt_item) in enumerate(zip(src_ids, tgt_ids)):
            src[i, : len(src_item)] = torch.tensor(src_item, dtype=torch.long)
            tgt[i, : len(tgt_item)] = torch.tensor(tgt_item, dtype=torch.long)

        return {
            "src": src,
            "tgt_in": tgt[:, :-1],
            "tgt_out": tgt[:, 1:],
            "src_tokens": list(src_tokens),
            "tgt_tokens": list(tgt_tokens),
        }

    return collate


def build_dataloader(
    data_dir: str | Path,
    split: str,
    src_word2id: dict[str, int],
    tgt_word2id: dict[str, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TranslationDataset(Path(data_dir) / f"{split}.txt", src_word2id, tgt_word2id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(),
    )
