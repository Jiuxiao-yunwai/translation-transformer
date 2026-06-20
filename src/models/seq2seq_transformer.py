import math

from torch import Tensor, nn

from models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 100,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 对矩阵参数使用 Xavier 初始化，和 Transformer 常见训练设置一致。
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        # embedding 乘 sqrt(d_model) 用于稳定 embedding 与位置编码的尺度。
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # 解码阶段只能看到当前位置之前的 token，因果 mask 由调用方传入。
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_mask is not None,
        )

    def forward(
        self,
        src: Tensor,
        tgt_input: Tensor,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decode(
            tgt_input,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(output)
