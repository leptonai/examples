import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import os
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get local rank from environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    print(f"Running on rank {rank} (local_rank: {local_rank})")

    def transform(example):
        imgs = [transforms.ToTensor()(img) for img in example["image"]]
        imgs = [transforms.Normalize((0.1307,), (0.3081,))(img) for img in imgs]
        example["image"] = torch.stack(imgs)
        example["label"] = torch.tensor(example["label"])
        return example
    
    dataset = load_dataset("mnist", split="train")
    dataset = dataset.with_transform(transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = MNISTModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, 11):
        sampler.set_epoch(epoch)
        for batch_idx, batch_data in enumerate(train_loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "mnist_model.pth")
        print("Model saved as mnist_model.pth")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    train()