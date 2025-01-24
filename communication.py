import torch.distributed as dist

def compressed_allreduce(tensor, world_size):
    tensor_to_send = tensor / world_size
    dist.all_reduce(tensor_to_send, op=dist.ReduceOp.SUM)
    return tensor_to_send
