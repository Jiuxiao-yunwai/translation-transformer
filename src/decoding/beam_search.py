import torch

from utils import generate_square_subsequent_mask, make_padding_mask
from vocab import BOS_IDX, EOS_IDX, Vocab


@torch.no_grad()
def beam_search_decode(
    model,
    src: torch.Tensor,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_len: int = 80,
    beam_size: int = 5,
    length_penalty: float = 0.6,
    min_len: int = 2,
) -> torch.Tensor:
    model.eval()
    device = src.device
    if src.dim() == 1:
        src = src.unsqueeze(1)

    results: list[list[int]] = []
    for batch_idx in range(src.size(1)):
        one_src = src[:, batch_idx : batch_idx + 1]
        src_key_padding_mask = make_padding_mask(one_src, src_vocab.token_to_id("<pad>"))
        memory = model.encode(one_src, src_key_padding_mask=src_key_padding_mask)
        beams: list[tuple[list[int], float, bool]] = [([BOS_IDX], 0.0, False)]

        for step in range(max_len):
            candidates: list[tuple[list[int], float, bool]] = []
            for token_ids, score, finished in beams:
                if finished:
                    candidates.append((token_ids, score, True))
                    continue

                ys = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
                tgt_mask = generate_square_subsequent_mask(ys.size(0), device)
                out = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
                log_probs = torch.log_softmax(model.generator(out[-1, 0]), dim=-1)
                if step < min_len:
                    log_probs[EOS_IDX] = -1e9

                top_scores, top_ids = torch.topk(log_probs, beam_size)
                for next_score, next_id in zip(top_scores.tolist(), top_ids.tolist()):
                    new_ids = token_ids + [int(next_id)]
                    candidates.append((new_ids, score + float(next_score), int(next_id) == EOS_IDX))

            def normalized(item: tuple[list[int], float, bool]) -> float:
                token_ids, score, _ = item
                length = max(len(token_ids) - 1, 1)
                return score / (length**length_penalty)

            beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]
            if all(finished for _, _, finished in beams):
                break

        best_ids = max(beams, key=lambda item: item[1] / (max(len(item[0]) - 1, 1) ** length_penalty))[0]
        results.append(best_ids[1:])

    max_out_len = max((len(ids) for ids in results), default=1)
    output = torch.full((max_out_len, len(results)), EOS_IDX, dtype=torch.long, device=device)
    for idx, ids in enumerate(results):
        output[: len(ids), idx] = torch.tensor(ids, dtype=torch.long, device=device)
    return output
