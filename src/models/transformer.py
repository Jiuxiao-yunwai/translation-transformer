import math

import torch
from torch import nn

from utils.constants import BOS_ID, EOS_ID, PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(query.size(0), query.size(1), self.num_heads * self.d_head)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_out))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_out))


class TransformerMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        seq_len = tgt.size(1)
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device, dtype=torch.bool))
        return pad_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt_in)
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt_in, memory, tgt_mask, src_mask)
        return self.generator(decoded)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int, bos_id: int = BOS_ID, eos_id: int = EOS_ID) -> torch.Tensor:
        self.eval()
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(ys)
            decoded = self.decode(ys, memory, tgt_mask, src_mask)
            logits = self.generator(decoded[:, -1])
            next_word = logits.argmax(dim=-1)
            next_word = torch.where(finished, torch.full_like(next_word, eos_id), next_word)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            finished |= next_word.eq(eos_id)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        beam_size: int = 4,
        length_penalty: float = 0.6,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
    ) -> torch.Tensor:
        self.eval()
        if beam_size <= 1:
            return self.greedy_decode(src, max_len=max_len, bos_id=bos_id, eos_id=eos_id)

        decoded_batch: list[list[int]] = []
        for sample in src:
            sample_src = sample.unsqueeze(0)
            src_mask = self.make_src_mask(sample_src)
            memory = self.encode(sample_src, src_mask)
            beam_tokens = torch.full((1, 1), bos_id, dtype=torch.long, device=src.device)
            beam_scores = torch.zeros(1, dtype=torch.float, device=src.device)
            finished = torch.zeros(1, dtype=torch.bool, device=src.device)

            for _ in range(max_len - 1):
                current_beam_size = beam_tokens.size(0)
                expanded_memory = memory.expand(current_beam_size, -1, -1)
                expanded_src_mask = src_mask.expand(current_beam_size, -1, -1, -1)
                tgt_mask = self.make_tgt_mask(beam_tokens)
                decoded = self.decode(beam_tokens, expanded_memory, tgt_mask, expanded_src_mask)
                logits = self.generator(decoded[:, -1])
                log_probs = torch.log_softmax(logits, dim=-1)

                if finished.any():
                    log_probs = log_probs.masked_fill(finished.unsqueeze(1), torch.finfo(log_probs.dtype).min)
                    log_probs[finished, eos_id] = 0.0

                candidate_scores = beam_scores.unsqueeze(1) + log_probs
                flat_scores = candidate_scores.reshape(-1)
                top_scores, top_indices = torch.topk(flat_scores, k=min(beam_size, flat_scores.numel()))
                beam_indices = top_indices // log_probs.size(1)
                token_indices = top_indices % log_probs.size(1)

                beam_tokens = torch.cat([beam_tokens[beam_indices], token_indices.unsqueeze(1)], dim=1)
                beam_scores = top_scores
                finished = finished[beam_indices] | token_indices.eq(eos_id)
                if finished.all():
                    break

            lengths = beam_tokens.ne(eos_id).sum(dim=1).clamp_min(1).float()
            normalized_scores = beam_scores / (lengths ** length_penalty)
            best_index = int(normalized_scores.argmax().item())
            decoded_batch.append(beam_tokens[best_index].tolist())

        max_output_len = max(len(tokens) for tokens in decoded_batch)
        output = torch.full((len(decoded_batch), max_output_len), eos_id, dtype=torch.long, device=src.device)
        for i, tokens in enumerate(decoded_batch):
            output[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=src.device)
        return output
