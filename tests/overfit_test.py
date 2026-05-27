import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import TranslationDataset, detokenize_cn, ids_to_tokens, load_vocab, make_collate_fn
from models.transformer import TransformerMT
from utils.constants import PAD_ID


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def batch_loss(model: TransformerMT, criterion: nn.Module, batch: dict[str, object], device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        logits = model(src, tgt_in)
        raw_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        ntokens = tgt_out.ne(PAD_ID).sum().item()
    return raw_loss.item() / max(ntokens, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny batch to validate the Transformer training path.")
    parser.add_argument("--data-dir", default="data/cmn-eng-simple")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    src_word2id, _, tgt_word2id, tgt_id2word = load_vocab(args.data_dir)
    dataset = TranslationDataset(Path(args.data_dir) / "training.txt", src_word2id, tgt_word2id)
    subset = Subset(dataset, list(range(args.samples)))
    loader = DataLoader(subset, batch_size=args.samples, shuffle=False, collate_fn=make_collate_fn())
    batch = next(iter(loader))

    model = TransformerMT(
        src_vocab_size=len(src_word2id),
        tgt_vocab_size=len(tgt_word2id),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=0.0,
        max_len=args.max_len,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    initial_loss = batch_loss(model, criterion, batch, device)
    print(f"device={device}")
    print(f"samples={args.samples} steps={args.steps}")
    print(f"initial_loss={initial_loss:.4f} initial_ppl={math.exp(min(initial_loss, 20)):.2f}")

    src = batch["src"].to(device)
    tgt_in = batch["tgt_in"].to(device)
    tgt_out = batch["tgt_out"].to(device)

    progress = tqdm(range(1, args.steps + 1), desc="overfit", dynamic_ncols=True, file=sys.stdout)
    for step in progress:
        model.train()
        logits = model(src, tgt_in)
        raw_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        ntokens = tgt_out.ne(PAD_ID).sum().item()
        loss = raw_loss / max(ntokens, 1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    final_loss = batch_loss(model, criterion, batch, device)
    print(f"final_loss={final_loss:.4f} final_ppl={math.exp(min(final_loss, 20)):.2f}")
    print(f"loss_drop={initial_loss - final_loss:.4f}")

    model.eval()
    decoded = model.greedy_decode(src[: min(5, args.samples)], max_len=args.max_len)
    for i, pred_ids in enumerate(decoded.tolist()):
        pred_tokens = ids_to_tokens(pred_ids, tgt_id2word)
        ref_tokens = batch["tgt_tokens"][i]
        src_text = " ".join(batch["src_tokens"][i])
        print(f"[{i + 1}] EN  : {src_text}")
        print(f"    Pred: {detokenize_cn(pred_tokens)}")
        print(f"    Ref : {detokenize_cn(ref_tokens)}")

    if final_loss > initial_loss * 0.35:
        raise SystemExit("overfit check failed: loss did not drop enough")


if __name__ == "__main__":
    main()
