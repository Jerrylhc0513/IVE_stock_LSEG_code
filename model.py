"""
model.py — PaperIVE model for IVE-style replication

Aligned with Section 3.2 of the IVE paper:
  - Two-tower encoder-decoder Transformer
  - 4 encoder layers and 4 decoder layers
  - Student-t distribution head
  - Multi-step prediction
  - Sinusoidal positional embedding
  - Time encoding through Time2Vec
  - Stock embedding
  - Linear input projection

Approximate number of parameters: 3M.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Time2Vec module."""
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.zeros(k))

    def forward(self, t):
        linear = (self.w0 * t + self.b0).unsqueeze(-1)
        periodic = torch.sin(t.unsqueeze(-1) * self.w + self.b)
        return torch.cat([linear, periodic], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PaperIVE(nn.Module):
    def __init__(self,
                 n_features: int = 15,
                 n_stocks: int = 10,
                 d_model: int = 192,
                 n_heads: int = 6,
                 n_layers: int = 4,
                 dim_ff: int = 384,
                 dropout: float = 0.1,
                 max_seq_len: int = 500,
                 time2vec_dim: int = 16,
                 multi_step: int = 3):
        super().__init__()
        self.d_model = d_model
        self.multi_step = multi_step

        self.enc_input_proj = nn.Linear(n_features, d_model)
        self.dec_input_proj = nn.Linear(n_features, d_model)

        self.stock_embed = nn.Embedding(n_stocks, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        self.time2vec = Time2Vec(k=time2vec_dim)
        self.time_proj = nn.Linear(time2vec_dim + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.head_mu = nn.Linear(d_model, multi_step)
        self.head_log_sigma = nn.Linear(d_model, multi_step)
        self.head_log_nu = nn.Linear(d_model, multi_step)

    @staticmethod
    def _causal_mask(T: int, device):
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, yesterday, today, today_mask, stock_id, day_idx):
        B, T_y, _ = yesterday.shape
        T_t = today.size(1)

        stock_emb = self.stock_embed(stock_id).unsqueeze(1)
        stock_emb_y = stock_emb.expand(B, T_y, -1)
        stock_emb_t = stock_emb.expand(B, T_t, -1)

        time_vec = self.time2vec(day_idx)
        time_emb = self.time_proj(time_vec).unsqueeze(1)
        time_emb_y = time_emb.expand(B, T_y, -1)
        time_emb_t = time_emb.expand(B, T_t, -1)

        h_y = self.enc_input_proj(yesterday)
        h_y = h_y + stock_emb_y + time_emb_y
        h_y = self.pos_enc(h_y)
        memory = self.encoder(h_y)

        h_t = self.dec_input_proj(today)
        h_t = h_t + stock_emb_t + time_emb_t
        h_t = self.pos_enc(h_t)

        tgt_mask = self._causal_mask(T_t, today.device)
        tgt_key_padding_mask = ~today_mask

        h_dec = self.decoder(
            tgt=h_t, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        last_idx = today_mask.sum(dim=1) - 1
        last_idx_expand = last_idx.view(B, 1, 1).expand(B, 1, self.d_model)
        h_last = h_dec.gather(dim=1, index=last_idx_expand).squeeze(1)

        mu = self.head_mu(h_last)
        sigma = F.softplus(self.head_log_sigma(h_last)) + 1e-3
        nu = F.softplus(self.head_log_nu(h_last)) + 2.0

        return mu, sigma, nu


def student_t_nll(y, mu, sigma, nu):
    dist = torch.distributions.StudentT(df=nu, loc=mu, scale=sigma)
    return -dist.log_prob(y).mean()


if __name__ == "__main__":
    print("=== Testing PaperIVE model ===\n")
    B, T_y, T_t, F_dim = 4, 390, 200, 15
    yesterday = torch.randn(B, T_y, F_dim)
    today = torch.randn(B, T_t, F_dim)
    actual_lens = torch.tensor([50, 100, 150, 200])
    today_mask = torch.arange(T_t).unsqueeze(0) < actual_lens.unsqueeze(1)
    stock_id = torch.tensor([0, 1, 5, 3])
    day_idx = torch.tensor([0.1, 0.3, 0.7, 0.9])
    target = torch.randn(B, 3)

    model = PaperIVE(n_features=15, n_stocks=10, multi_step=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")

    mu, sigma, nu = model(yesterday, today, today_mask, stock_id, day_idx)
    print(f"\nForward output:")
    print(f"  mu:    {tuple(mu.shape)}, range=[{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  sigma: {tuple(sigma.shape)}, range=[{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  nu:    {tuple(nu.shape)}, range=[{nu.min():.2f}, {nu.max():.2f}]")

    loss = student_t_nll(target, mu, sigma, nu)
    print(f"\nLoss: {loss.item():.4f}")
    loss.backward()
    print("Backward pass completed successfully")