from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from vocab import BOS_IDX, EOS_IDX, PAD_IDX, Vocab


@dataclass
class ParallelExample:
    src_tokens: list[str]
    tgt_tokens: list[str]


def tokenize_src(text: str) -> list[str]:
    # 英文按空格分词并统一小写，和预处理生成的数据格式保持一致。
    return text.lower().strip().split()


def tokenize_tgt(text: str) -> list[str]:
    # 中文数据已经用空格切好词，这里不能再按字切分。
    return text.strip().split()


def read_parallel_data(path: str | Path, max_src_len: int = 80, max_tgt_len: int = 80) -> list[ParallelExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    examples: list[ParallelExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no} is not tab-separated as 'english sentence\\tchinese sentence'."
                )

            src_tokens = tokenize_src(parts[0])
            tgt_tokens = tokenize_tgt(parts[1])
            if not src_tokens or not tgt_tokens:
                continue
            # 过滤过长样本，避免极端长句拖慢训练并扩大显存占用。
            if len(src_tokens) > max_src_len or len(tgt_tokens) > max_tgt_len:
                continue
            examples.append(ParallelExample(src_tokens=src_tokens, tgt_tokens=tgt_tokens))

    if not examples:
        raise ValueError(f"no usable sentence pairs found in {path}")
    return examples


def build_vocabs(
    train_examples: list[ParallelExample],
    min_freq: int = 1,
    max_src_vocab_size: int | None = None,
    max_tgt_vocab_size: int | None = None,
) -> tuple[Vocab, Vocab]:
    src_vocab = Vocab.build(
        [example.src_tokens for example in train_examples],
        min_freq=min_freq,
        max_size=max_src_vocab_size,
    )
    tgt_vocab = Vocab.build(
        [example.tgt_tokens for example in train_examples],
        min_freq=min_freq,
        max_size=max_tgt_vocab_size,
    )
    return src_vocab, tgt_vocab


class TranslationDataset(Dataset):
    def __init__(self, examples: list[ParallelExample], src_vocab: Vocab, tgt_vocab: Vocab) -> None:
        self.examples = examples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        src_ids = self.src_vocab.encode(example.src_tokens, add_eos=True)
        # teacher forcing：decoder 输入以 <bos> 开头，监督目标以 <eos> 结尾，两者整体右移一位。
        tgt_input_ids = [BOS_IDX] + self.tgt_vocab.encode(example.tgt_tokens)
        tgt_output_ids = self.tgt_vocab.encode(example.tgt_tokens) + [EOS_IDX]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_input_ids, dtype=torch.long),
            "tgt_output_ids": torch.tensor(tgt_output_ids, dtype=torch.long),
            "src_text": " ".join(example.src_tokens),
            "tgt_text": " ".join(example.tgt_tokens),
            "tgt_tokens": example.tgt_tokens,
        }


def collate_fn(batch: list[dict]) -> dict:
    # PyTorch Transformer 默认使用 [seq_len, batch]，所以这里不转成 batch_first。
    src = pad_sequence([item["src_ids"] for item in batch], padding_value=PAD_IDX)
    tgt_input = pad_sequence([item["tgt_input_ids"] for item in batch], padding_value=PAD_IDX)
    tgt_output = pad_sequence([item["tgt_output_ids"] for item in batch], padding_value=PAD_IDX)
    return {
        "src": src,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "src_text": [item["src_text"] for item in batch],
        "tgt_text": [item["tgt_text"] for item in batch],
        "tgt_tokens": [item["tgt_tokens"] for item in batch],
    }


def create_dataloader(
    examples: list[ParallelExample],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TranslationDataset(examples, src_vocab, tgt_vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
