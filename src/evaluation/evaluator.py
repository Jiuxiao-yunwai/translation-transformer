from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from data import create_dataloader
from decoding import beam_search_decode, greedy_decode
from metrics import corpus_bleu, corpus_bleu_scores
from utils import generate_square_subsequent_mask, make_padding_mask
from vocab import PAD_IDX, Vocab


def sequence_to_tokens(ids: list[int], vocab: Vocab) -> list[str]:
    return vocab.decode(ids, remove_special=True, stop_at_eos=True)


@torch.no_grad()
def evaluate_loss_and_accuracy(
    model,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "valid",
    show_progress: bool = True,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

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
        non_pad = tgt_output.ne(PAD_IDX)
        pred = logits.argmax(dim=-1)

        total_loss += loss.item()
        total_tokens += int(non_pad.sum().item())
        correct += int(pred.eq(tgt_output).logical_and(non_pad).sum().item())
        progress.set_postfix(
            loss=f"{total_loss / max(total_tokens, 1):.4f}",
            acc=f"{correct / max(total_tokens, 1):.4f}",
        )

    return total_loss / max(total_tokens, 1), correct / max(total_tokens, 1)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    src_vocab: Vocab | None = None,
    tgt_vocab: Vocab | None = None,
    device: torch.device | None = None,
    max_len: int = 80,
    decode: str = "beam",
    beam_size: int = 5,
    length_penalty: float = 0.6,
    min_len: int = 2,
    desc: str = "bleu",
    show_progress: bool = True,
) -> tuple[float, list[dict]]:
    model.eval()
    if src_vocab is None:
        src_vocab = dataloader.dataset.src_vocab
    if tgt_vocab is None:
        tgt_vocab = dataloader.dataset.tgt_vocab
    if device is None:
        device = next(model.parameters()).device

    references: list[list[str]] = []
    candidates: list[list[str]] = []
    records: list[dict] = []

    progress = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch in progress:
        src = batch["src"].to(device)
        if decode == "beam":
            decoded_tensor = beam_search_decode(
                model,
                src,
                src_vocab,
                tgt_vocab,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                min_len=min_len,
            )
        else:
            decoded_tensor = greedy_decode(model, src, src_vocab, tgt_vocab, max_len=max_len)

        decoded = decoded_tensor.transpose(0, 1).cpu().tolist()
        for src_text, ref_tokens, hyp_ids in zip(batch["src_text"], batch["tgt_tokens"], decoded):
            hyp_tokens = sequence_to_tokens(hyp_ids, tgt_vocab)
            references.append(ref_tokens)
            candidates.append(hyp_tokens)
            records.append(
                {
                    "src": src_text,
                    "ref": " ".join(ref_tokens),
                    "hyp": " ".join(hyp_tokens),
                }
            )
        progress.set_postfix(samples=len(records))
    return corpus_bleu(references, candidates), records


def write_predictions(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(f"SRC: {record['src']}\n")
            f.write(f"REF: {record['ref']}\n")
            f.write(f"HYP: {record['hyp']}\n\n")


def make_eval_dataloader(examples, src_vocab: Vocab, tgt_vocab: Vocab, batch_size: int, num_workers: int = 0):
    return create_dataloader(examples, src_vocab, tgt_vocab, batch_size, shuffle=False, num_workers=num_workers)


def bleu_scores_from_records(records: list[dict]) -> dict[str, float]:
    references = [record["ref"].split() for record in records]
    candidates = [record["hyp"].split() for record in records]
    return corpus_bleu_scores(references, candidates, max_n=4)
