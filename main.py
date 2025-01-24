import torch.multiprocessing as mp
from train import train

if __name__ == '__main__':
    world_size = 4
    dataset = torch.utils.data.TensorDataset(torch.randn(1000, 128), torch.randn(1000, 128))
    mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)
