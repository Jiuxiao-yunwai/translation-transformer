import argparse

import torch

from data import tokenize_src
from decoding import beam_search_decode, greedy_decode
from models import Seq2SeqTransformer
from utils import get_device
from vocab import PAD_IDX, Vocab


def load_model(ckpt_path: str, device: torch.device) -> tuple[Seq2SeqTransformer, Vocab, Vocab]:
    ckpt = torch.load(ckpt_path, map_location=device)
    src_vocab = Vocab.from_dict(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_dict(ckpt["tgt_vocab"])
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        **ckpt["model_args"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Translate one English sentence.")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--min_decode_len", type=int, default=2)
    args = parser.parse_args()

    device = get_device(args.device)
    model, src_vocab, tgt_vocab = load_model(args.ckpt, device)
    # 单句输入也整理成 [seq_len, batch=1]，与训练/评估时的张量形状一致。
    src_ids = src_vocab.encode(tokenize_src(args.sentence), add_eos=True)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)

    if args.decode == "beam":
        pred = beam_search_decode(
            model,
            src,
            src_vocab,
            tgt_vocab,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            min_len=args.min_decode_len,
        )
    else:
        pred = greedy_decode(model, src, src_vocab, tgt_vocab, max_len=args.max_len)

    # 输出前去掉 <bos>/<eos>/<pad> 等控制符，只保留中文词序列。
    pred_ids = pred.squeeze(1).tolist()
    pred_tokens = tgt_vocab.decode(pred_ids, remove_special=True, stop_at_eos=True)
    print(" ".join(pred_tokens))


if __name__ == "__main__":
    main()
