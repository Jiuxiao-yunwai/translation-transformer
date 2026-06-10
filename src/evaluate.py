import argparse
import re
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from data import create_dataloader, read_parallel_data
from evaluation import bleu_scores_from_records, evaluate, evaluate_loss_and_accuracy, write_predictions
from train import load_best_model
from utils import get_device, save_json
from vocab import PAD_IDX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer checkpoint.")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--test_path", default="data/test.txt")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name", default=None, help="Optional evaluation output subdirectory name.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_src_len", type=int, default=80)
    parser.add_argument("--max_tgt_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=0.8)
    parser.add_argument("--min_decode_len", type=int, default=2)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sanitize_run_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._-") or "eval"


def make_eval_output_dir(base_dir: str, run_name: str | None) -> Path:
    if run_name:
        return Path(base_dir) / sanitize_run_name(run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{timestamp}_eval"


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, src_vocab, tgt_vocab, ckpt = load_best_model(Path(args.ckpt), device)
    examples = read_parallel_data(args.test_path, args.max_src_len, args.max_tgt_len)
    dataloader = create_dataloader(examples, src_vocab, tgt_vocab, args.batch_size, False, args.num_workers)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.03, reduction="sum")

    test_loss, test_token_acc = evaluate_loss_and_accuracy(
        model,
        dataloader,
        criterion,
        device,
        desc="test loss",
        show_progress=args.progress,
    )
    test_bleu, records = evaluate(
        model,
        dataloader,
        src_vocab,
        tgt_vocab,
        device,
        max_len=args.max_len,
        decode=args.decode,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        min_len=args.min_decode_len,
        desc=f"test {args.decode}",
        show_progress=args.progress,
    )

    output_dir = make_eval_output_dir(args.output_dir, args.run_name)
    write_predictions(records, output_dir / "test_predictions.txt")
    bleu_scores = bleu_scores_from_records(records)
    result = {
        "test_bleu": round(bleu_scores["bleu4"], 2),
        "test_bleu1": round(bleu_scores["bleu1"], 2),
        "test_bleu2": round(bleu_scores["bleu2"], 2),
        "test_bleu3": round(bleu_scores["bleu3"], 2),
        "test_bleu4": round(bleu_scores["bleu4"], 2),
        "test_token_accuracy": round(test_token_acc, 4),
        "test_loss": round(test_loss, 4),
        "best_valid_bleu": round(float(ckpt.get("valid_bleu", 0.0)), 2),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "decode": args.decode,
        "beam_size": args.beam_size if args.decode == "beam" else None,
        "length_penalty": args.length_penalty if args.decode == "beam" else None,
        "output_dir": str(output_dir),
    }
    save_json(result, output_dir / "result.json")
    print(
        f"test_bleu1={bleu_scores['bleu1']:.2f} "
        f"test_bleu2={bleu_scores['bleu2']:.2f} "
        f"test_bleu3={bleu_scores['bleu3']:.2f} "
        f"test_bleu4={bleu_scores['bleu4']:.2f} "
        f"test_token_acc={test_token_acc:.4f}"
    )
    print(result)


if __name__ == "__main__":
    main()
