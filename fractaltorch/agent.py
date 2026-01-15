import torch
import torch.nn as nn
from .core import MultiAgentFractalCore
from .umht import UnifiedMultiHeadTransformerLSTMCell

class FractalAgent:
    def __init__(self, hidden_size, max_depth, reasoning_order=1, agent_count=5):
        self.hidden_size = hidden_size
        self.reasoning_order = reasoning_order
        self.core = MultiAgentFractalCore(hidden_size, max_depth, agent_count)

    def choose_move(self, board_state_tensor):
        # board_state_tensor: torch.Tensor [1, H]
        # Dummy: choose argmax of global mean over legal moves (legal moves not modeled here)
        global_state = self.core.forward(board_state_tensor, order_k=self.reasoning_order)
        score = global_state.mean().item()
        return 0  # placeholder
