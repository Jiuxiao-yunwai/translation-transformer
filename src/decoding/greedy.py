import torch

from utils import generate_square_subsequent_mask, make_padding_mask
from vocab import BOS_IDX, EOS_IDX, Vocab


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int = 80) -> torch.Tensor:
    model.eval()
    device = src.device
    if src.dim() == 1:
        src = src.unsqueeze(1)
    src_key_padding_mask = make_padding_mask(src, src_vocab.token_to_id("<pad>"))
    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)

    batch_size = src.size(1)
    ys = torch.full((1, batch_size), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        # 自回归解码：每轮只把当前已生成前缀送入 decoder，再取最后一步的最大概率 token。
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device)
        out = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = model.generator(out[-1])
        next_tokens = logits.argmax(dim=-1)
        # 已经生成 <eos> 的样本后续强制保持 <eos>，避免 batch 内不同长度样本互相影响。
        next_tokens = torch.where(finished, torch.full_like(next_tokens, EOS_IDX), next_tokens)
        ys = torch.cat([ys, next_tokens.unsqueeze(0)], dim=0)
        finished |= next_tokens.eq(EOS_IDX)
        if finished.all():
            break
    return ys[1:]
