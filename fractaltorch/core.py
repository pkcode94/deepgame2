import torch
import torch.nn as nn
from .opponent import FractalOpponent
from .ramanujan import ramanujan_sum

class MultiAgentFractalCore(nn.Module):
    def __init__(self, hidden_size, max_depth, agent_count):
        super().__init__()
        self.hidden_size = hidden_size
        self.agent_count = agent_count

        # agents: use FractalOpponent-like cells; here lightweight opponents
        self.agents = nn.ModuleList([FractalOpponent(hidden_size, max_depth) for _ in range(agent_count)])

        self.cross_q = nn.Linear(hidden_size, hidden_size)
        self.cross_k = nn.Linear(hidden_size, hidden_size)
        self.cross_v = nn.Linear(hidden_size, hidden_size)

        self.global_fuse = nn.Linear(hidden_size * agent_count, hidden_size)
        self.global_ramanujan_head = nn.Linear(hidden_size * 2, hidden_size)
        self.global_anchor_gate = nn.Linear(hidden_size * 2, hidden_size)

    @staticmethod
    def ramanujan_sum_static(states, head):
        return ramanujan_sum(states, head)

    def reason_once(self, x, h_cur, c_cur):
        h_local = []
        c_next = []
        output = []
        for i in range(self.agent_count):
            out, h_n, c_n = self.agents[i].forward(x, h_cur[i], c_cur[i])
            output.append(out)
            h_local.append(h_n)
            c_next.append(c_n)

        H = torch.cat(h_local, dim=0)  # [A,H]

        Q = self.cross_q(H)
        K = self.cross_k(H)
        V = self.cross_v(H)

        scores = Q @ K.transpose(0, 1)
        weights = torch.softmax(scores, dim=1)

        H_attn = weights @ V

        hNext = [H_attn[i].unsqueeze(0) for i in range(self.agent_count)]

        H_flat = H_attn.flatten(0, 1).unsqueeze(0)
        global_state = torch.tanh(self.global_fuse(H_flat))

        return hNext, c_next, global_state

    def forward(self, x, order_k=1):
        h = [torch.zeros(1, self.hidden_size, device=x.device) for _ in range(self.agent_count)]
        c = [torch.zeros(1, self.hidden_size, device=x.device) for _ in range(self.agent_count)]
        globals = []

        steps = max(1, order_k)
        last_global = None

        for k in range(steps):
            h, c, global_state = self.reason_once(x, h, c)
            globals.append(global_state)
            last_global = global_state

        global_anchor = ramanujan_sum(globals, self.global_ramanujan_head)

        gate_input = torch.cat([last_global, global_anchor], dim=1)
        gate_raw = self.global_anchor_gate(gate_input)
        g = torch.sigmoid(gate_raw)

        global_out = g * global_anchor + (1 - g) * last_global
        return global_out
