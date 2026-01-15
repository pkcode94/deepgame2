import torch
from fractaltorch.umht import UnifiedMultiHeadTransformerLSTMCell
from fractaltorch.utils import set_seed


def run_test():
    set_seed(1234)
    B = 1
    H = 64
    input_size = H
    model = UnifiedMultiHeadTransformerLSTMCell(input_size, H, heads=4, mem_len=8)

    # initial hidden and cell states
    h = torch.zeros(1, H)
    c = torch.zeros(1, H)

    # random input
    x = torch.randn(1, input_size)

    out, h_new, c_new = model.forward_step(x, h, c)

    print("Output shape:", out.shape)
    print("h_new shape:", h_new.shape)
    print("c_new shape:", c_new.shape)
    print("Output sample:", out[0, :5].tolist())

if __name__ == '__main__':
    run_test()
