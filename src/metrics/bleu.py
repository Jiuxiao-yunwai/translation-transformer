import math
from collections import Counter


def get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references: list[list[str]], candidates: list[list[str]], max_n: int = 4) -> float:
    return corpus_bleu_scores(references, candidates, max_n=max_n)[f"bleu{max_n}"]


def corpus_bleu_scores(references: list[list[str]], candidates: list[list[str]], max_n: int = 4) -> dict[str, float]:
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length")
    if not references:
        return {f"bleu{n}": 0.0 for n in range(1, max_n + 1)}

    clipped_matches = [0 for _ in range(max_n)]
    total_candidates = [0 for _ in range(max_n)]
    cand_len = 0
    ref_len = 0

    for ref, cand in zip(references, candidates):
        cand_len += len(cand)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            cand_ngrams = get_ngrams(cand, n)
            ref_ngrams = get_ngrams(ref, n)
            total_candidates[n - 1] += sum(cand_ngrams.values())
            for ngram, count in cand_ngrams.items():
                # clipped precision：候选 n-gram 命中数不能超过参考译文中出现次数。
                clipped_matches[n - 1] += min(count, ref_ngrams.get(ngram, 0))

    if cand_len == 0:
        return {f"bleu{n}": 0.0 for n in range(1, max_n + 1)}
    # brevity penalty：候选总长度短于参考时惩罚过短译文。
    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - ref_len / cand_len)

    scores: dict[str, float] = {}
    for bleu_n in range(1, max_n + 1):
        log_precision_sum = 0.0
        for matched, total in zip(clipped_matches[:bleu_n], total_candidates[:bleu_n]):
            # 加一平滑避免高阶 n-gram 零命中时 BLEU 直接变成 0。
            precision = (matched + 1.0) / (total + 1.0)
            log_precision_sum += (1.0 / bleu_n) * math.log(precision)
        scores[f"bleu{bleu_n}"] = bp * math.exp(log_precision_sum) * 100.0
    return scores
