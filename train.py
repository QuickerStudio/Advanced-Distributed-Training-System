import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from distributed_setup import setup, cleanup
from model import MyModel, MyPipeModel
from optimizer import get_optimizer
from data_loader import get_dataloader
from communication import compressed_allreduce

def train(rank, world_size, dataset, num_epochs=10, batch_size=32):
    setup(rank, world_size)

    model = MyPipeModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = get_optimizer(ddp_model)
    scaler = GradScaler()
    train_loader = get_dataloader(dataset, batch_size, num_workers=4, rank=rank, world_size=world_size)

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            with autocast():
                outputs = ddp_model(inputs)
                loss = F.mse_loss(outputs, labels)
            scaler.scale(loss).backward()
            for param in ddp_model.parameters():
                param.grad = compressed_allreduce(param.grad, world_size)
            scaler.step(optimizer)
            scaler.update()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        # Save checkpoint
        if dist.get_rank() == 0:
            torch.save(ddp_model.state_dict(), f'model_epoch_{epoch}.pth')

    cleanup()
