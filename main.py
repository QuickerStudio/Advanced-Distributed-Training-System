import torch.multiprocessing as mp
from train import train
import grpc_communication

if __name__ == '__main__':
    world_size = 4
    dataset = torch.utils.data.TensorDataset(torch.randn(1000, 128), torch.randn(1000, 128))
    mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)
    
    # Start the gRPC server
    grpc_communication.serve()
    
    # Run the gRPC client
    grpc_communication.run_client()