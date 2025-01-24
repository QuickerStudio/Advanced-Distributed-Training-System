import torch

def get_dataloader(dataset, batch_size, num_workers, rank, world_size):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler
    )
    return dataloader
