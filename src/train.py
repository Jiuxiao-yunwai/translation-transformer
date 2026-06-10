import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from data import build_vocabs, create_dataloader, read_parallel_data
from evaluation import bleu_scores_from_records, evaluate, evaluate_loss_and_accuracy, write_predictions
from models import Seq2SeqTransformer
from utils import ensure_dir, generate_square_subsequent_mask, get_device, make_padding_mask, save_json, set_seed
from vocab import PAD_IDX, Vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Transformer for English-to-Chinese translation.")
    parser.add_argument("--train_path", default="data/train.txt")
    parser.add_argument("--valid_path", default="data/valid.txt")
    parser.add_argument("--test_path", default="data/test.txt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name", default=None, help="Experiment name. Defaults to an auto timestamped name.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_src_len", type=int, default=80)
    parser.add_argument("--max_tgt_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument(
        "--test_decode",
        choices=["greedy", "beam"],
        default=None,
        help="Decode method for final test. Defaults to --decode.",
    )
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--min_decode_len", type=int, default=2)
    parser.add_argument("--bleu_every", type=int, default=1, help="Compute validation BLEU every N epochs.")
    parser.add_argument(
        "--final_ckpt",
        choices=["best", "last"],
        default="best",
        help="Checkpoint used for final test evaluation.",
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation BLEU epochs. 0 disables early stopping.",
    )
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum BLEU improvement for early stopping.")
    return parser.parse_args()


def sanitize_run_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._-") or "run"


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return sanitize_run_name(args.run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return sanitize_run_name(
        f"{timestamp}_dm{args.d_model}_bs{args.batch_size}_{args.decode}"
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None

    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(max(args.warmup_steps, 1), total_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    min_lr_ratio = args.eta_min / args.lr

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        if step <= warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: Seq2SeqTransformer,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    desc: str,
    show_progress: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch in progress:
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0), device)
        src_pad_mask = make_padding_mask(src)
        tgt_pad_mask = make_padding_mask(tgt_input)

        logits = model(
            src,
            tgt_input,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        non_pad_tokens = int(tgt_output.ne(PAD_IDX).sum().item())
        total_loss += loss.item()
        total_tokens += non_pad_tokens
        progress.set_postfix(
            loss=f"{total_loss / max(total_tokens, 1):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return total_loss / max(total_tokens, 1)


def save_checkpoint(
    path: Path,
    model: Seq2SeqTransformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    args: argparse.Namespace,
    epoch: int,
    valid_bleu: float,
    valid_loss: float,
    valid_token_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "model_args": {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers or args.num_layers,
                "num_decoder_layers": args.num_decoder_layers or args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "max_len": max(args.max_src_len + 1, args.max_tgt_len + 1, args.max_len + 1, 100),
            },
            "train_args": vars(args),
            "epoch": epoch,
            "valid_bleu": valid_bleu,
            "valid_loss": valid_loss,
            "valid_token_acc": valid_token_acc,
        },
        path,
    )


def load_best_model(path: Path, device: torch.device) -> tuple[Seq2SeqTransformer, Vocab, Vocab, dict]:
    ckpt = torch.load(path, map_location=device)
    src_vocab = Vocab.from_dict(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_dict(ckpt["tgt_vocab"])
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        **ckpt["model_args"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, src_vocab, tgt_vocab, ckpt


def main() -> None:
    args = parse_args()
    args.num_encoder_layers = args.num_encoder_layers or args.num_layers
    args.num_decoder_layers = args.num_decoder_layers or args.num_layers
    args.test_decode = args.test_decode or args.decode
    if args.bleu_every < 1:
        raise ValueError("--bleu_every must be >= 1")
    set_seed(args.seed)

    device = get_device(args.device)
    run_name = make_run_name(args)
    checkpoint_dir = ensure_dir(Path(args.checkpoint_dir) / run_name)
    output_dir = ensure_dir(Path(args.output_dir) / run_name)
    print(f"device={device}")
    print(f"run_name={run_name}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"output_dir={output_dir}")

    train_examples = read_parallel_data(args.train_path, args.max_src_len, args.max_tgt_len)
    valid_examples = read_parallel_data(args.valid_path, args.max_src_len, args.max_tgt_len)
    test_examples = read_parallel_data(args.test_path, args.max_src_len, args.max_tgt_len)
    print(f"data: train={len(train_examples)} valid={len(valid_examples)} test={len(test_examples)}")

    src_vocab, tgt_vocab = build_vocabs(train_examples, min_freq=args.min_freq)
    print(f"vocab: src={len(src_vocab)} tgt={len(tgt_vocab)}")

    batch_size = args.batch_size
    if device.type == "cuda":
        try:
            _ = torch.empty((batch_size, args.max_len, args.d_model), device=device)
            del _
        except RuntimeError:
            batch_size = min(batch_size, 64)
            torch.cuda.empty_cache()
            print(f"GPU memory check failed; using batch_size={batch_size}")

    train_loader = create_dataloader(train_examples, src_vocab, tgt_vocab, batch_size, True, args.num_workers)
    valid_loader = create_dataloader(valid_examples, src_vocab, tgt_vocab, batch_size, False, args.num_workers)
    test_loader = create_dataloader(test_examples, src_vocab, tgt_vocab, batch_size, False, args.num_workers)

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=max(args.max_src_len + 1, args.max_tgt_len + 1, args.max_len + 1, 100),
        pad_idx=PAD_IDX,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=args.label_smoothing, reduction="sum")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    best_valid_bleu = -1.0
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args.grad_clip,
            desc=f"epoch {epoch:03d} train",
            show_progress=args.progress,
        )
        valid_loss, valid_token_acc = evaluate_loss_and_accuracy(
            model,
            valid_loader,
            criterion,
            device,
            desc=f"epoch {epoch:03d} valid",
            show_progress=args.progress,
        )
        should_eval_bleu = epoch == 1 or epoch % args.bleu_every == 0 or epoch == args.epochs
        if should_eval_bleu:
            valid_bleu, _ = evaluate(
                model,
                valid_loader,
                src_vocab,
                tgt_vocab,
                device,
                max_len=args.max_len,
                decode=args.decode,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                min_len=args.min_decode_len,
                desc=f"epoch {epoch:03d} {args.decode}",
                show_progress=args.progress,
            )
        else:
            valid_bleu = None

        bleu_text = f"{valid_bleu:.2f}" if valid_bleu is not None else "SKIP"
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_bleu={bleu_text} "
            f"valid_token_acc={valid_token_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        checkpoint_bleu = valid_bleu if valid_bleu is not None else best_valid_bleu
        save_checkpoint(
            last_path,
            model,
            src_vocab,
            tgt_vocab,
            args,
            epoch,
            checkpoint_bleu,
            valid_loss,
            valid_token_acc,
        )

        if valid_bleu is None:
            continue

        improved = valid_bleu > best_valid_bleu + args.min_delta
        if improved:
            best_valid_bleu = valid_bleu
            bad_epochs = 0
            save_checkpoint(
                best_path,
                model,
                src_vocab,
                tgt_vocab,
                args,
                epoch,
                valid_bleu,
                valid_loss,
                valid_token_acc,
            )
            print(f"saved best checkpoint to {best_path}")
        elif args.early_stopping_patience > 0:
            bad_epochs += 1
            print(
                f"early_stopping: no BLEU improvement for "
                f"{bad_epochs}/{args.early_stopping_patience} epochs"
            )
            if bad_epochs >= args.early_stopping_patience:
                print(
                    f"early stopping at epoch {epoch}: "
                    f"best_valid_bleu={best_valid_bleu:.2f}"
                )
                break

    final_ckpt_path = last_path if args.final_ckpt == "last" else best_path
    best_model, best_src_vocab, best_tgt_vocab, best_ckpt = load_best_model(final_ckpt_path, device)
    test_loss, test_token_acc = evaluate_loss_and_accuracy(
        best_model,
        test_loader,
        criterion,
        device,
        desc="test loss",
        show_progress=args.progress,
    )
    test_bleu, records = evaluate(
        best_model,
        test_loader,
        best_src_vocab,
        best_tgt_vocab,
        device,
        max_len=args.max_len,
        decode=args.test_decode,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        min_len=args.min_decode_len,
        desc=f"test {args.test_decode}",
        show_progress=args.progress,
    )
    write_predictions(records, output_dir / "test_predictions.txt")
    test_bleu_scores = bleu_scores_from_records(records)

    result = {
        "test_bleu": round(test_bleu_scores["bleu4"], 2),
        "test_bleu1": round(test_bleu_scores["bleu1"], 2),
        "test_bleu2": round(test_bleu_scores["bleu2"], 2),
        "test_bleu3": round(test_bleu_scores["bleu3"], 2),
        "test_bleu4": round(test_bleu_scores["bleu4"], 2),
        "test_token_accuracy": round(test_token_acc, 4),
        "test_loss": round(test_loss, 4),
        "best_valid_bleu": round(float(best_ckpt["valid_bleu"]), 2),
        "best_epoch": int(best_ckpt["epoch"]),
        "final_ckpt": args.final_ckpt,
        "final_ckpt_path": str(final_ckpt_path),
        "model": "Transformer",
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "eta_min": args.eta_min,
        "batch_size": batch_size,
        "valid_decode": args.decode,
        "test_decode": args.test_decode,
        "bleu_every": args.bleu_every,
        "beam_size": args.beam_size if args.test_decode == "beam" else None,
        "length_penalty": args.length_penalty if args.test_decode == "beam" else None,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "run_name": run_name,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
    }
    save_json(result, output_dir / "result.json")
    print(
        f"test_bleu1={test_bleu_scores['bleu1']:.2f} "
        f"test_bleu2={test_bleu_scores['bleu2']:.2f} "
        f"test_bleu3={test_bleu_scores['bleu3']:.2f} "
        f"test_bleu4={test_bleu_scores['bleu4']:.2f} "
        f"test_token_acc={test_token_acc:.4f}"
    )
    print(f"wrote {output_dir / 'test_predictions.txt'} and {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
