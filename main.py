from math import pi
from math import cos
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from resnet import *
# from utils import progress_bar
import wandb
import time
import copy


def proposed_lr(initial_lr , epochs, stages, epoch, burnin=0.1, func=None, gamma=0.2):
    # proposed learning late function
    # func = None(cosine), 3steplr,
    num_burnin = epochs * burnin
    epoch_per_cycle = (epochs - num_burnin) // stages
    percent = ((epoch-num_burnin) % epoch_per_cycle) / epoch_per_cycle
    if epoch < num_burnin:
        res = initial_lr
    # elif func is None:
    #     return initial_lr * (cos(pi * percent + 1)) / 2
    elif func == '3steplr':
        if percent < 0.33:
            res = initial_lr
        elif percent < 0.66:
            res = initial_lr * gamma
        else:
            res = initial_lr * gamma * gamma
    else:
        res = initial_lr * (cos(pi * percent + 1)) / 2
    return res


def train_se(epochs, stages, model, criterion, optimizer,
             train_loader, test_loader,
             scheduler=None, burnin=0.1,
             path='save_model/'):
    # train_errs = []
    # train_loss = []
    snapshots = []
    epochs_per_cycle = epochs // stages
    wandb.watch(model)
    for epoch in range(epochs):
        start = time.time()
        train_err, loss1 = train_epoch(model, criterion, optimizer, train_loader)
        test_err, loss2 = test(model, test_loader, criterion)
        print(
            'Epoch {:03d}/{:03d}, train error: {:.2%} || test error {:.2%}'.format(epoch, epochs, train_err, test_err))
        # train_errs.append(train_err)
        # train_loss.append(loss1)

        if scheduler is None:
            lr_epoch = proposed_lr(lr, epochs, stages, epoch, func='3steplr')
            optimizer.param_groups[0]['lr'] = lr_epoch
        else:
            scheduler.step()

        if (epoch + 1) % epochs_per_cycle == 0:
            # torch.save(model.state_dict(), path+'ext_epoch=%d.pt'%epoch)
            snapshots.append(copy.deepcopy(model.state_dict()))
        # Log training..
        wandb.log({'train_loss': loss1, 'val_loss': loss2,
                   "train_err": train_err, "val_err": test_err,
                   "lr": optimizer.param_groups[0]["lr"],
                   "epoch_time": time.time() - start})
    return snapshots


def train_epoch(model, criterion, optimizer, loader):
    total_correct = 0.
    total_samples = 0.
    loss_sum = 0.
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)

        predictions = output.data.max(1, keepdim=True)[1]
        total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
        total_samples += len(target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return 1 - total_correct / total_samples, loss_sum / (batch_idx + 1)


def test(model, loader, criterion):
    total_correct = 0.
    total_samples = 0.
    loss_sum = 0.
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()

            predictions = output.data.max(1, keepdim=True)[1]
            total_correct += predictions.eq(target.data.view_as(predictions)).sum().item()
            total_samples += len(target)

    return 1 - total_correct / total_samples, loss_sum / (batch_idx + 1)


def test_se(snapshots, use_model_num, test_loader, path='save_model/', ensemble='average'):
    index = len(snapshots) - use_model_num
    snapshots = snapshots[index:]
    model_list = [ResNet18() for _ in snapshots]

    for model, weight in zip(model_list, snapshots):
        model.load_state_dict(weight)
        model.eval()
        if device == 'cuda':
            model.cuda()

    total_correct = 0
    total_samples = 0
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if device == 'cuda':
            data, target = data.cuda(), target.cuda()

        output_list = [model(data) for model in model_list]
        loss_ = [criterion(output, target).item() for output in output_list]

        # predictions = output.data.max(1, keepdim=True)[1]
        pred_list = [nn.Softmax(dim=1)(output) for output in output_list]
        # todo add more ensemble strategy
        if ensemble == 'average':
            predictions = sum(pred_list) / use_model_num

        else:  # ensemble == 'vote'
            pred_label_list = []
            for pred in pred_list:
                pred_m = torch.zeros_like(pred)
                pred_m[torch.arange(len(pred)), pred.argmax(1)] = 1
                pred_label_list.append(pred_m)
            predictions = sum(pred_label_list)

        total_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()
        total_samples += len(target)
        # todo add more ensemble strategy
        loss_sum += sum(loss_) / len(model_list)

    test_loss = loss_sum / (batch_idx + 1)
    test_err = 1 - total_correct / total_samples
    print('\nTest set: Average loss: {:.4f}, Error rate: {:.2%}\n'.format(
        test_loss, test_err))

    return test_loss, test_err


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root='data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    net1 = ResNet18().cuda()

    wandb.init(project="Snapshot-cifar10",
               name='resnet_snapshot_100')

    lr = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net1.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-4)
    snapshots = train_se(epochs=100, stages=5, model=net1,
                         criterion=criterion, optimizer=optimizer,
                         train_loader=trainloader, test_loader=testloader, burnin=0.1)
    #  scheduler=scheduler)

