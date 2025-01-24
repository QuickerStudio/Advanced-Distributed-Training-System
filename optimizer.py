import deepspeed
import torch.optim as optim

def get_optimizer(model):
    optimizer = deepspeed.zero.Init(
        model.parameters(),
        optimizer=optim.Adam(model.parameters(), lr=0.001)
    )
    return optimizer
