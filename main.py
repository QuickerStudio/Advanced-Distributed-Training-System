import torch.multiprocessing as mp
from train import train
import grpc_communication
from data_collector import DataCollector

if __name__ == '__main__':
    api_key = "your_api_key_here"
    kafka_servers = ["localhost:9092"]
    data_types = ["text", "image", "audio", "video", "code"]
    collector = DataCollector(api_key, kafka_servers)
    collector.collect_data_threaded(data_types)
    classified_data = collector.classify_data()
    collector.store_data(classified_data)

    world_size = 4
    dataset = torch.utils.data.TensorDataset(torch.randn(1000, 128), torch.randn(1000, 128))
    mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)
    
    # Start the gRPC server
    grpc_communication.serve()
    
    # Run the gRPC client
    grpc_communication.run_client()