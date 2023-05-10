import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile,ProfilerActivity
import torch.nn.parallel

from time import perf_counter

import sys
import os

n=len(sys.argv)
if n>=4:
    resnetsize=int(sys.argv[1])
    gpu = int(sys.argv[2])
    mixed = int(sys.argv[3])
    mixed = mixed == 1
else:
    resnetsize=18
    gpu=1
    mixed=False

envi = {1:'0',2:'0,1',4:'0,1,2,3'}
os.environ['CUDA_VISIBLE_DEVICES']=envi[gpu]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

scale = torch.cuda.amp.GradScaler()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4)

def train(net, epochs=2):
    ttime=0
    t0 = perf_counter()
    for epoch in range(epochs): 

        t1=perf_counter()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        t2=perf_counter()

        if epoch>0:
            ttime+=t2-t1
    t4=perf_counter()
    print("Average time:",ttime/(epochs-1))
    print("Total time:",t4-t0)

def train_mixed(net, epochs=2):
    ttime=0
    t0 = perf_counter()
    for epoch in range(epochs): 

        t1=perf_counter()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            scale.scale(loss).backward()
            scale.step(optimizer)
            scale.update()
        t2=perf_counter()
        if epoch>0:
            ttime+=t2-t1
    t4=perf_counter()
    print("Average time:",ttime/(epochs-1))
    print("Total time:",t4-t0)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
if __name__=="__main__":
    if resnetsize == 18:
        net = torchvision.models.resnet18()
    elif resnetsize == 34:
        net = torchvision.models.resnet34()
    elif resnetsize == 50:
        net = torchvision.models.resnet50()
    if gpu>1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    if not mixed:
        print("Non-mixed ","Using Resnet-",resnetsize," with ",gpu," GPUs", sep="")
        train(net,20)
        test(net)
    else:
        print("Mixed precision ","Using Resnet-",resnetsize," with ",gpu," GPUs", sep="")
        train_mixed(net,20)
        test(net)