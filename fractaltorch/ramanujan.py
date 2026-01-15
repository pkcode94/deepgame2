import torch


def ramanujan_sum(states, head):
    """
    states: list of [1,H] tensors
    head: nn.Module mapping [1,2H] -> [1,H]
    returns: [1,H]
    """
    if len(states) == 0:
        raise ValueError("states array is empty")
    M = torch.cat(states, dim=0)  # [S,H]
    alpha1 = 0.4
    alpha2 = 0.8
    anchor1 = torch.zeros(1, M.shape[1], device=M.device)
    anchor2 = torch.zeros(1, M.shape[1], device=M.device)
    for t in range(M.shape[0]):
        h_t = M[t].unsqueeze(0)
        anchor1 = (1 - alpha1) * anchor1 + alpha1 * h_t
        anchor2 = (1 - alpha2) * anchor2 + alpha2 * h_t
    concat = torch.cat([anchor1, anchor2], dim=1)  # [1,2H]
    hStar = head(concat)
    return hStar
