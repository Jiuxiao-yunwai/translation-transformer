import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from data.dataset import build_dataloader, load_vocab
from models.transformer import TransformerMT
from utils.constants import PAD_ID
from utils.metrics import corpus_bleu


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_model(args: argparse.Namespace, src_vocab_size: int, tgt_vocab_size: int) -> TransformerMT:
    return TransformerMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
    )


def run_epoch(
    model: TransformerMT,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float = 1.0,
    desc: str = "train",
    show_progress: bool = True,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    progress = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        disable=not show_progress,
    )
    for batch in progress:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        logits = model(src, tgt_in)
        raw_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        ntokens = tgt_out.ne(PAD_ID).sum().item()
        loss = raw_loss / max(ntokens, 1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += raw_loss.item()
        total_tokens += ntokens
        postfix = {"loss": f"{total_loss / max(total_tokens, 1):.4f}"}
        if optimizer is not None:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        progress.set_postfix(postfix)

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_bleu(
    model: TransformerMT,
    dataloader,
    tgt_id2word: dict[int, str],
    device: torch.device,
    max_len: int,
    decode: str = "beam",
    beam_size: int = 4,
    length_penalty: float = 0.6,
    desc: str = "bleu",
    show_progress: bool = True,
) -> float:
    from data.dataset import ids_to_tokens

    model.eval()
    references: list[list[str]] = []
    candidates: list[list[str]] = []
    progress = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        disable=not show_progress,
    )
    for batch in progress:
        src = batch["src"].to(device)
        if decode == "beam":
            output = model.beam_search_decode(
                src,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
        else:
            output = model.greedy_decode(src, max_len=max_len)
        for ids, ref in zip(output.tolist(), batch["tgt_tokens"]):
            candidates.append(ids_to_tokens(ids, tgt_id2word))
            references.append(ref)
    return corpus_bleu(references, candidates)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "warmup_cosine":
        total_steps = max(args.epochs * steps_per_epoch, 1)
        warmup_steps = max(args.warmup_steps, 1)
        decay_steps = max(total_steps - warmup_steps, 1)
        min_lr_ratio = args.eta_min / args.lr

        def warmup_cosine_lambda(step: int) -> float:
            step = max(step, 1)
            if step <= warmup_steps:
                return step / warmup_steps
            progress = min((step - warmup_steps) / decay_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    def noam_lambda(step: int) -> float:
        step = max(step, 1)
        warmup = max(args.warmup_steps, 1)
        scale = (args.d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
        return args.lr_factor * scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)


def save_checkpoint(
    path: Path,
    model: TransformerMT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    bleu: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "val_loss": val_loss,
            "bleu": bleu,
            "model_args": {
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "max_len": args.max_len,
            },
            "train_args": vars(args),
            "data_dir": args.data_dir,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer for English-Chinese translation.")
    parser.add_argument("--data-dir", default="data/cmn-eng-simple")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scheduler", choices=["noam", "warmup_cosine", "none"], default="noam")
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bleu-every", type=int, default=1)
    parser.add_argument("--decode", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    src_word2id, _, tgt_word2id, tgt_id2word = load_vocab(args.data_dir)

    train_loader = build_dataloader(
        args.data_dir, "training", src_word2id, tgt_word2id, args.batch_size, True, args.num_workers
    )
    val_loader = build_dataloader(
        args.data_dir, "validation", src_word2id, tgt_word2id, args.batch_size, False, args.num_workers
    )

    model = make_model(args, len(src_word2id), len(tgt_word2id)).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        reduction="sum",
        label_smoothing=args.label_smoothing,
    )
    base_lr = 1.0 if args.scheduler == "noam" else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    best_bleu = -1.0
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scheduler,
            args.grad_clip,
            desc="train",
        )
        val_loss = run_epoch(model, val_loader, criterion, device, desc="valid")
        bleu = float("nan")
        if args.bleu_every > 0 and epoch % args.bleu_every == 0:
            bleu = evaluate_bleu(
                model,
                val_loader,
                tgt_id2word,
                device,
                args.max_len,
                decode=args.decode,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                desc=f"{args.decode}-bleu",
            )

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} bleu={bleu:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = not math.isnan(bleu) and bleu > best_bleu + args.min_delta
        if improved:
            best_bleu = bleu
            bad_epochs = 0
            save_checkpoint(Path(args.save_dir) / "best.pt", model, optimizer, scheduler, epoch, val_loss, bleu, args)
        elif not math.isnan(bleu):
            bad_epochs += 1

        save_checkpoint(Path(args.save_dir) / "last.pt", model, optimizer, scheduler, epoch, val_loss, bleu, args)
        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            print(f"early stopping: BLEU did not improve for {bad_epochs} evaluation rounds")
            break


if __name__ == "__main__":
    main()
