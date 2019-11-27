"""
Seq2seq based: Attention Is All You Need
https://arxiv.org/abs/1706.03762
"""
import math
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocabulary, device, hid_dim=512, n_layers=6, n_heads=8,
                 pf_dim=2048, dropout=0.1, max_positions=100):
        super().__init__()

        input_dim = len(vocabulary)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_positions, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.do = nn.Dropout(dropout)

        self.scale = math.sqrt(hid_dim)

    def mask_src(self, src):
        # src = [batch size, src sent len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask # (batch_size, 1, 1, src_len)

    def forward(self, src):
        #src = [batch size, src sent len]
        src_mask = self.mask_src(src) # (batch_size, src_len)

        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        src = self.do((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch size, src sent len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src sent len, hid dim]
        #src_mask = [batch size, src sent len]

        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))

        src = self.ln(src + self.do(self.pf(src)))

        return src

class Decoder(nn.Module):
    def __init__(self, vocabulary, device, hid_dim=512, n_layers=6,
                 n_heads=8, pf_dim=2048, dropout=0.1, max_positions=100):
        super().__init__()

        output_dim = len(vocabulary)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_positions, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])

        self.fc = nn.Linear(hid_dim, output_dim)

        self.do = nn.Dropout(dropout)

        self.scale = math.sqrt(hid_dim)

    def forward(self, trg, src, trg_mask, src_mask):

        #trg = [batch_size, trg sent len]
        #src = [batch_size, src sent len]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]

        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)

        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        #trg = [batch size, trg sent len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        return self.fc(trg)

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):

        #trg = [batch size, trg sent len, hid dim]
        #src = [batch size, src sent len, hid dim]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):

        bsz = query.shape[0]

        #query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        #Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        #Q, K, V = [batch size, n heads, sent len, hid dim // n heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, sent len, sent len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(torch.softmax(energy, dim=-1))

        #attention = [batch size, n heads, sent len, sent len]

        x = torch.matmul(attention, V)

        #x = [batch size, n heads, sent len, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, sent len, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        #x = [batch size, src sent len, hid dim]

        x = self.fc(x)

        #x = [batch size, sent len, hid dim]

        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        #x = [batch size, sent len, hid dim]

        x = self.do(torch.relu(self.fc_1(x)))

        #x = [batch size, sent len, pf dim]

        x = self.fc_2(x)

        #x = [batch size, sent len, hid dim]

        return x

def mask_src(self, src):
    # src = [batch size, src sent len]
    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask # (batch_size, 1, 1, src_len)

def mask_trg(self, trg):
    # trg = (batch_size, trg_len)
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3) # (batch_size, 1, trg_len, 1)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool() # (trg_len, trg_len)
    trg_mask = trg_pad_mask & trg_sub_mask # (batch_size, 1, trg_len, trg_len)

    return trg_mask


def make_masks(self, src, trg):

        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        #src_mask = [batch size, 1, 1, src sent len]
        #trg_pad_mask = [batch size, 1, trg sent len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        #trg_sub_mask = [trg sent len, trg sent len]

        trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, 1, trg sent len, trg sent len]

        return src_mask, trg_mask

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=512, factor=1, warmup=2000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()