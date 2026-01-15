import math
import torch
import torch.nn as nn
from .utils import set_seed

class UnifiedMultiHeadTransformerLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, heads=4, lstm_layers=1, mem_len=32, seed: int = 1234):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.lstm_layers = lstm_layers
        self.mem_len = mem_len

        # set seed for deterministic initialization (attempt to mirror C#/TorchSharp defaults)
        set_seed(seed)

        # single-step LSTMCell to match C# LSTMCell behavior
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, heads)
        self.cross_attn = nn.MultiheadAttention(hidden_size, heads)

        # optional input projection not implemented here; caller should match sizes
        self.residual_project = nn.Linear(hidden_size, hidden_size)
        self.router_mlp = nn.Linear(hidden_size, 2)

        # CT-gate params
        self.bottleneck_dim = max(1, hidden_size // 4)
        self.W_ct_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_ct_compress = nn.Linear(hidden_size, self.bottleneck_dim)
        self.W_ct_expand = nn.Linear(self.bottleneck_dim, hidden_size)

        # use explicit dropout module to mirror C# functional.dropout behavior with same p
        self._dropout = nn.Dropout(p=0.2)

        # internal rolling memory (S,B,H)
        self.register_buffer("memK", None)
        self.register_buffer("memV", None)

        # initialize weights similarly to TorchSharp defaults (kaiming_uniform for linear weights, zeros for bias)
        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.LSTMCell):
                # default PyTorch LSTMCell init is already fine; for parity we can use xavier for weights
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _ensure_batch(self, x):
        t = x
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def forward_step(self, x, h, c, externalK=None, externalV=None):
        # x: [B, input_size] or [input_size]
        xIn = self._ensure_batch(x)  # [B, input_size]

        # Ensure h,c have batch dim
        h_in = self._ensure_batch(h).squeeze(0) if h.dim() == 2 and h.shape[0] == 1 else h
        c_in = self._ensure_batch(c).squeeze(0) if c.dim() == 2 and c.shape[0] == 1 else c

        # LSTMCell update (single step)
        h_lstm, c_next = self.lstm_cell(xIn, (h_in, c_in))  # both [B, H]

        # Write into rolling memory memK/memV as [S,B,H]
        cur = h_lstm.unsqueeze(0)  # [1,B,H]
        if getattr(self, 'memK', None) is None or self.memK is None:
            # create memory tensors on same device
            self.memK = cur.detach()
            self.memV = cur.detach()
        else:
            newK = torch.cat([self.memK, cur.detach()], dim=0)
            newV = torch.cat([self.memV, cur.detach()], dim=0)
            if newK.shape[0] > self.mem_len:
                start = newK.shape[0] - self.mem_len
                # keep last mem_len entries
                self.memK = newK[start: , :, :]
                self.memV = newV[start: , :, :]
            else:
                self.memK = newK
                self.memV = newV

        # SELF attention over rolling memory
        # q: [1,B,H], key/value: [S,B,H]
        qSelf = h_lstm.unsqueeze(0)
        # nn.MultiheadAttention expects (L, N, E)
        self_out, _ = self.self_attn(qSelf, self.memK, self.memV)
        self_out = self_out.squeeze(0)  # [B,H]

        # EXTERNAL attention if provided
        if externalK is not None and externalV is not None:
            qCross = h_lstm.unsqueeze(0)
            ext_out, _ = self.cross_attn(qCross, externalK, externalV)
            external_out = ext_out.squeeze(0)
            has_external = True
        else:
            external_out = torch.zeros_like(h_lstm)
            has_external = False

        # ROUTER over self vs external
        router_logits = self.router_mlp(h_lstm)  # [B,2]
        router_weights = torch.softmax(router_logits, dim=1)  # [B,2]
        wSelf = router_weights[:, 0].unsqueeze(1)  # [B,1]
        wExt = router_weights[:, 1].unsqueeze(1)   # [B,1]

        routed_attn = wSelf * self_out + wExt * external_out  # [B,H]

        # residual projection and combined routed
        routed = h_lstm + routed_attn + self.residual_project(routed_attn)

        # CT-GATE temporal summary and bottleneck
        if getattr(self, 'memK', None) is None or self.memK is None or self.memK.shape[0] == 0:
            temporal_summary = torch.zeros_like(h_lstm)
        else:
            temporal_summary = self.memK.mean(dim=0)  # [B,H]

        compressed_small = self.W_ct_compress(temporal_summary)  # [B, bottleneck]
        expanded = self.W_ct_expand(compressed_small)            # [B, H]

        cs_dim = compressed_small.shape[1]
        reps = (self.hidden_size + cs_dim - 1) // cs_dim
        compressed_expanded = compressed_small.repeat(1, reps)[:, :self.hidden_size]

        gate_input = torch.cat([xIn, routed], dim=1)  # [B, input+H]
        g = torch.sigmoid(self.W_ct_gate(gate_input))  # [B,H]

        hCT = torch.lerp(compressed_expanded, expanded, g)

        # Ensemble and final output
        stacked = torch.stack([h_lstm, routed, hCT], dim=0)  # [3, B, H]
        hFinal = stacked.mean(dim=0)  # [B,H]

        hFinal = self._dropout(hFinal)

        # Return exactly like C# forward_step: (output, h, c)
        # output: hFinal [B,H]
        # h: h_lstm [B,H]
        # c: c_next [B,H]
        return hFinal, h_lstm, c_next

    def ResetMemory(self):
        self.memK = None
        self.memV = None
