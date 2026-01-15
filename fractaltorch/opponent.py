import torch
import torch.nn as nn
from .pathgate import CombinatorialPathGate
from .umht import UnifiedMultiHeadTransformerLSTMCell
from .ramanujan import ramanujan_sum

class FractalOpponent(nn.Module):
    def __init__(self, hidden_size, max_depth):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max(0, max_depth)
        self.cell = UnifiedMultiHeadTransformerLSTMCell(hidden_size, hidden_size, heads=4, lstm_layers=1)
        self.path_gate = CombinatorialPathGate(hidden_size=hidden_size, basis_count=4, depth=max_depth)
        self.depth_router = nn.Linear(hidden_size, max_depth + 1)
        self.depth_ramanujan_head = nn.Linear(hidden_size * 2, hidden_size)
        self.depth_anchor_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.last_winning_expert = -1

    def debug_feedback_loop(self, h_cur, h_next_raw, depth, damping=1.0):
        signal_mag = h_cur.abs().mean().item()
        correction_mag = (h_next_raw.abs().mean() * damping).item()
        feedback_ratio = correction_mag / (signal_mag + 1e-6)
        print(f"[Depth {depth}] Feedback-Ratio: {feedback_ratio:.2%} (Signal: {signal_mag:.4f})")
        if feedback_ratio != feedback_ratio or feedback_ratio > 10.0:
            print(f"CRITICAL: Feedback Loop Divergence at Depth {depth}")

    def forward(self, x, h, c, depth=0):
        # call cell.forward_step(x,h,c) exactly like C# signature
        shallow_out, h_shallow, c_shallow = self.cell.forward_step(x, h, c)

        h_path = self.path_gate(h_shallow)
        self.last_winning_expert = self.path_gate.last_winning_expert

        # depth decision via depth_router
        with torch.no_grad():
            depth_logits = self.depth_router(h_path)
            depth_probs = torch.softmax(depth_logits, dim=1)
            depth_idx = torch.argmax(depth_probs, dim=1)
            chosen_depth = int(depth_idx.item())
            remaining = max(0, self.max_depth - depth)
            steps = min(chosen_depth, remaining)

        if steps <= 0:
            return shallow_out.clone(), h_path.clone(), c_shallow.clone()

        h_cur = h_path.clone()
        c_cur = c_shallow.clone()
        final_out = shallow_out.clone()

        depth_states = [h_cur.clone()]
        for i in range(steps):
            damping = float(1.0 / ((depth + 1) ** 2))
            h_recursive_out, h_next_raw, c_next = self.forward(x, h_cur, c_cur, depth + 1)

            if (torch.randint(0, 50, (1,)).item() == 0):
                self.debug_feedback_loop(h_cur, h_next_raw, depth, damping)

            h_next = torch.lerp(h_cur, h_next_raw, damping)
            depth_states.append(h_next.clone())

            # clean up and set for next loop
            h_cur = h_next
            c_cur = c_next
            final_out = h_recursive_out

        # compute ramanujan anchor from depth states using helper
        h_depth_anchor = ramanujan_sum(depth_states, self.depth_ramanujan_head)

        depth_gate_input = torch.cat([h_cur, h_depth_anchor], dim=1)
        depth_gate_raw = self.depth_anchor_gate(depth_gate_input)
        depth_g = torch.sigmoid(depth_gate_raw)

        h_out = depth_g * h_depth_anchor + (1 - depth_g) * h_cur

        return final_out, h_out, c_cur
