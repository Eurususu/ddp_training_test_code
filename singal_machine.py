#coding=utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse
import time


# 定义训练函数
def train(batch, epochs=5):

    # 数据集和数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch)

    # 模型、损失函数和优化器
    model = models.resnet18(pretrained=False).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练循环
    for epoch in range(epochs):
        t1 = time.time()
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"one epoch cost {time.time()-t1}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Number of batch size')
    args = parser.parse_args()
    train(args.batch)
