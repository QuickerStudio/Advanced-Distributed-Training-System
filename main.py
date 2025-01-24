import torch.multiprocessing as mp
from train import train
import grpc_communication
from data_collector import DataCollector
from ai_resource_manager import AIResourceManager
import ctypes
import sys

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if __name__ == '__main__':
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1
        )
        sys.exit()

    api_key = "your_api_key_here"
    kafka_servers = ["localhost:9092"]
    data_types = ["text", "image", "audio", "video", "code"]
    collector = DataCollector(api_key, kafka_servers)
    collector.collect_data_threaded(data_types)
    classified_data = collector.classify_data()
    collector.store_data(classified_data)

    ai_resource_manager = AIResourceManager()
    ai_resource_manager.start_monitoring()

    world_size = 4
    dataset = torch.utils.data.TensorDataset(torch.randn(1000, 128), torch.randn(1000, 128))
    mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)
    
    # Start the gRPC server
    grpc_communication.serve()
    
    # Run the gRPC client
    grpc_communication.run_client()
