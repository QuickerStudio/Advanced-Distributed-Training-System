import torch
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Setup completed for rank {rank} in world size {world_size}.")

def cleanup():
    dist.destroy_process_group()
    print("Cleanup completed.")
