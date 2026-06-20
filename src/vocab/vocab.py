from collections import Counter
from dataclasses import dataclass


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(
        cls,
        token_lists: list[list[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # 先按词频降序、再按字典序排序，保证同样数据每次构建出的词表稳定一致。
        words = [
            word
            for word, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if freq >= min_freq and word not in SPECIAL_TOKENS
        ]
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIAL_TOKENS))]

        itos = SPECIAL_TOKENS + words
        stoi = {word: idx for idx, word in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        itos = list(data["itos"])
        return cls(stoi={word: idx for idx, word in enumerate(itos)}, itos=itos)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    def __len__(self) -> int:
        return len(self.itos)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return UNK_TOKEN

    def encode(self, tokens: list[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_IDX)
        ids.extend(self.token_to_id(token) for token in tokens)
        if add_eos:
            ids.append(EOS_IDX)
        return ids

    def decode(self, ids: list[int], remove_special: bool = True, stop_at_eos: bool = True) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            if stop_at_eos and idx == EOS_IDX:
                break
            # BLEU 和输出文件不应包含 padding、起止符等控制 token。
            if remove_special and idx in {PAD_IDX, BOS_IDX, EOS_IDX}:
                continue
            tokens.append(self.id_to_token(idx))
        return tokens
