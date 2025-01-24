import torch.multiprocessing as mp
from train import train
import grpc_communication
from data_collector import DataCollector
from ai_resource_manager import AIResourceManager
import ctypes
import sys
import os
import threading
from storage import create_bucket, upload_file_minio, download_file_minio, multi_thread_download, multi_thread_upload, generate_key, initialize_rscode, backup_data, schedule_backups, serve, run_client, setup_rdma, rdma_send, rdma_receive, compressed_allreduce

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

    # 数据加密与解密
    key = generate_key()
    rsc = initialize_rscode(n=10, k=8)  # 例如 n=10, k=8 表示 2 个冗余码

    # 数据备份与恢复
    source_dir = "/path/to/data"
    backup_dir = "/path/to/backup"
    target_dir = "/path/to/restore"
    backup_data(source_dir, backup_dir)
    schedule_backups(source_dir, backup_dir, backup_interval=3600)  # 每小时自动备份一次

    # 创建存储桶
    client = None  # 初始化 MinIO 客户端
    create_bucket(client, "training-data")

    # 上传和下载文件到 MinIO
    upload_file_minio(client, "training-data", "local/path/to/your/training_data.csv", "training_data.csv")
    download_file_minio(client, "training-data", "training_data.csv", "local/path/to/save/training_data.csv")

    # 多线程上传与下载
    ips = ["127.0.0.1"]
    port = 8080
    download_dir = "downloads"
    os.makedirs(download_dir, exist_ok=True)

    multi_thread_upload("127.0.0.1", port, ["file1.txt", "file2.txt"], key, rsc)
    multi_thread_download(ips, port, download_dir, key, rsc)

    # gRPC 服务端
    threading.Thread(target=serve).start()

    # gRPC 客户端
    run_client()

    # RDMA 传输
    server_id, client_id, pd, qp = setup_rdma()
    
    # 发送数据
    data = b"hello RDMA"
    rdma_send(client_id, qp, data)
    
    # 接收数据
    rdma_receive(server_id, pd)

    # 分布式计算
    tensor = torch.randn(10, 10)
    compressed_allreduce(tensor, world_size=4)

    # 启动前端接口
    app.run(debug=True)

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

    # 启动前端接口
    app.run(debug=True)
