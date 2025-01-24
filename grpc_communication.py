import grpc
import ssl
from concurrent import futures
import distributed_pb2
import distributed_pb2_grpc

class DistributedServicer(distributed_pb2_grpc.DistributedServicer):
    def __init__(self, model):
        self.model = model

    def send_gradients(self, request, context):
        # 处理接收到的梯度
        return distributed_pb2.Response(message='Gradients received')

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    distributed_pb2_grpc.add_DistributedServicer_to_server(DistributedServicer(), server)

    # 使用SSL/TLS加密
    with open('server.crt', 'rb') as f:
        certificate_chain = f.read()
    with open('server.key', 'rb') as f:
        private_key = f.read()

    server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain),))
    server.add_secure_port('[::]:50051', server_credentials)
    server.start()
    server.wait_for_termination()

def run_client():
    with open('client.crt', 'rb') as f:
        trusted_certs = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
    channel = grpc.secure_channel('localhost:50051', credentials)
    stub = distributed_pb2_grpc.DistributedStub(channel)
    response = stub.send_gradients(distributed_pb2.GradientMessage(data='gradient data'))
    print("Client received: " + response.message)