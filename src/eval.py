import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data.dataset import build_dataloader, detokenize_cn, ids_to_tokens, load_vocab
from models.transformer import TransformerMT
from train import get_device
from utils.metrics import corpus_bleu


def load_model(checkpoint_path: str | Path, data_dir: str | Path, device: torch.device) -> tuple[TransformerMT, dict[int, str]]:
    src_word2id, _, tgt_word2id, tgt_id2word = load_vocab(data_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    model = TransformerMT(
        src_vocab_size=len(src_word2id),
        tgt_vocab_size=len(tgt_word2id),
        **model_args,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tgt_id2word


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--data-dir", default="data/cmn-eng-simple")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="testing", choices=["training", "validation", "testing"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument("--decode", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    args = parser.parse_args()

    device = get_device(args.device)
    src_word2id, _, tgt_word2id, _ = load_vocab(args.data_dir)
    model, tgt_id2word = load_model(args.checkpoint, args.data_dir, device)
    dataloader = build_dataloader(args.data_dir, args.split, src_word2id, tgt_word2id, args.batch_size, False)

    references: list[list[str]] = []
    candidates: list[list[str]] = []
    examples: list[tuple[str, str, str]] = []
    progress = tqdm(dataloader, desc=f"{args.decode}-eval", leave=False, dynamic_ncols=True, file=sys.stdout)
    for batch in progress:
        src = batch["src"].to(device)
        if args.decode == "beam":
            output = model.beam_search_decode(
                src,
                max_len=args.max_len,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
            )
        else:
            output = model.greedy_decode(src, max_len=args.max_len)
        for src_tokens, pred_ids, ref_tokens in zip(batch["src_tokens"], output.tolist(), batch["tgt_tokens"]):
            pred_tokens = ids_to_tokens(pred_ids, tgt_id2word)
            candidates.append(pred_tokens)
            references.append(ref_tokens)
            if len(examples) < args.show:
                examples.append((" ".join(src_tokens), detokenize_cn(pred_tokens), detokenize_cn(ref_tokens)))

    print(f"{args.split} BLEU: {corpus_bleu(references, candidates):.4f}")
    for i, (src_text, pred_text, ref_text) in enumerate(examples, start=1):
        print(f"[{i}] EN: {src_text}")
        print(f"    Pred: {pred_text}")
        print(f"    Ref : {ref_text}")


if __name__ == "__main__":
    main()
