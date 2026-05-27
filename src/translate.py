import argparse

import torch

from data.dataset import detokenize_cn, encode_tokens, ids_to_tokens, load_vocab
from eval import load_model
from train import get_device


def normalize_english(sentence: str) -> list[str]:
    return sentence.strip().lower().split()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Translate one English sentence with a trained checkpoint.")
    parser.add_argument("sentence", nargs="+")
    parser.add_argument("--data-dir", default="data/cmn-eng-simple")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    args = parser.parse_args()

    sentence = " ".join(args.sentence)
    device = get_device(args.device)
    src_word2id, _, _, _ = load_vocab(args.data_dir)
    model, tgt_id2word = load_model(args.checkpoint, args.data_dir, device)

    src_ids = encode_tokens(normalize_english(sentence), src_word2id)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    if args.decode == "beam":
        output = model.beam_search_decode(
            src,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )[0].tolist()
    else:
        output = model.greedy_decode(src, max_len=args.max_len)[0].tolist()
    tokens = ids_to_tokens(output, tgt_id2word)
    print(detokenize_cn(tokens))


if __name__ == "__main__":
    main()
