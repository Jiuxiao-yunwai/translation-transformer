import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import build_dataloader, load_vocab
from models.transformer import TransformerMT
from utils.metrics import corpus_bleu


def main() -> None:
    data_dir = "data/cmn-eng-simple"
    src_word2id, _, tgt_word2id, _ = load_vocab(data_dir)
    loader = build_dataloader(data_dir, "validation", src_word2id, tgt_word2id, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = TransformerMT(
        src_vocab_size=len(src_word2id),
        tgt_vocab_size=len(tgt_word2id),
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        dropout=0.1,
        max_len=64,
    )
    logits = model(batch["src"], batch["tgt_in"])
    assert logits.shape[:2] == batch["tgt_in"].shape
    assert logits.shape[-1] == len(tgt_word2id)

    decoded = model.greedy_decode(batch["src"], max_len=12)
    assert decoded.shape[0] == batch["src"].shape[0]
    beam_decoded = model.beam_search_decode(batch["src"], max_len=12, beam_size=2)
    assert beam_decoded.shape[0] == batch["src"].shape[0]
    assert 0.0 <= corpus_bleu([["我"]], [["我"]]) <= 1.0
    print("smoke test passed")


if __name__ == "__main__":
    main()
