#coding=utf-8
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time

# 初始化分布式训练环境
def setup(local_rank):
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 作为通信后端
        init_method='env://'
    )
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的 GPU

# 清理分布式训练环境
def cleanup():
    dist.destroy_process_group()

# 定义训练函数
def train(world_size, batch, epochs=5):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Running DDP training on rank {rank}, local_rank {local_rank}.")
    setup(local_rank)

    # 数据集和数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler)

    # 模型、损失函数和优化器
    model = models.resnet18(pretrained=False).cuda()
    model = DDP(model, device_ids=[local_rank])  # 分布式封装
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练循环
    for epoch in range(epochs):
        t1 = time.time()
        model.train()
        train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"one epoch cost {time.time()-t1}s")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=256, help='Number of batch size')
    args = parser.parse_args()
    # 启动训练
    world_size = int(os.environ["WORLD_SIZE"])
    train(world_size, args.batch)
