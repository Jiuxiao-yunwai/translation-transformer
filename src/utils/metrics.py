import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    references: list[list[str]],
    candidates: list[list[str]],
    max_n: int = 4,
    smooth: float = 1e-9,
) -> float:
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length")

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    cand_len = 0

    for ref, cand in zip(references, candidates):
        ref_len += len(ref)
        cand_len += len(cand)
        for n in range(1, max_n + 1):
            ref_ngrams = _ngrams(ref, n)
            cand_ngrams = _ngrams(cand, n)
            total_counts[n - 1] += max(len(cand) - n + 1, 0)
            clipped_counts[n - 1] += sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())

    if cand_len == 0:
        return 0.0

    precisions = [
        (clipped_counts[i] + smooth) / (total_counts[i] + smooth)
        for i in range(max_n)
    ]
    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1.0 - ref_len / cand_len)
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
