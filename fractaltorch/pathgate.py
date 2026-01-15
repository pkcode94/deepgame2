import torch
import torch.nn as nn

class CombinatorialPathGate(nn.Module):
    def __init__(self, hidden_size, basis_count=4, depth=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = max(1, basis_count)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_experts)])
        self.expert_router = nn.Linear(hidden_size, self.num_experts)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.last_winning_expert = -1

    def forward(self, x):
        t = x
        if t.dim() == 1:
            t = t.unsqueeze(0)
        logits = self.expert_router(t)
        winner = torch.argmax(logits, dim=1)
        if winner.numel() != 1:
            raise RuntimeError("Batch > 1 not supported")
        expert_idx = int(winner.item())
        self.last_winning_expert = expert_idx
        mix = torch.tanh(self.experts[expert_idx](t))
        g = torch.sigmoid(self.gate(t))
        out = g * mix + (1 - g) * t
        return out
