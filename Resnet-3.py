import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile,ProfilerActivity
import torch.nn.parallel
import torch.distributed as dist

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
os.environ['WORLD_SIZE']=str(gpu)

def init_distributed():

    dist_url = "env://" 

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    torch.cuda.set_device(local_rank)
    dist.barrier()

init_distributed()

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
train_sampler = DistributedSampler(dataset=trainset, shuffle=True) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          sampler=train_sampler, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4)

def train(net, epochs=2):
    ttime=0
    t0 = perf_counter()
    for epoch in range(epochs): 
        trainloader.sampler.set_epoch(epoch)
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
    if dist.get_rank()==0:
        print("Average time:",ttime/(epochs-1))
        print("Total time:",t4-t0)

def train_mixed(net, epochs=2):
    ttime=0
    t0 = perf_counter()
    for epoch in range(epochs): 

        t1=perf_counter()
        for i, data in enumerate(trainloader, 0):
            trainloader.sampler.set_epoch(epoch)
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
    if dist.get_rank()==0:
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
        net.to(device)
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    if not mixed:
        if dist.get_rank()==0:
            print("Non-mixed ","Using Resnet-",resnetsize," with ",gpu," GPUs", sep="")
        train(net,20)
    else:
        if dist.get_rank()==0:
            print("Mixed precision ","Using Resnet-",resnetsize," with ",gpu," GPUs", sep="")
        train_mixed(net,20)

    dist.destroy_process_group()

    test(net)
    path = './models/Resnet-'+str(resnetsize)
    if mixed:
        path+="-mixed"
    path +='-'+str(gpu)+'gpus'
    path+='.pth'
    torch.save(net.state_dict(),path)

