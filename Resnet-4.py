import torch
import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
from time import perf_counter

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

parser = argparse.ArgumentParser()

parser.add_argument('-q',choices=[0,1],default=0,type=int)
parser.add_argument('-r',choices=[18,34,50],default=18,type=int)
parser.add_argument('-m',choices=[0,1],default=0,type=int)
parser.add_argument('-ts',choices=[0,1],default=0,type=int)


args = parser.parse_args()
path = './models/Resnet-'+str(args.r)+'-'
if args.m == 1:
    path+='mixed-'
path+='1gpu.pth'

Quantize = args.q==1
TorchScript = args.ts==1

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

if args.r==18:
    net=torchvision.models.resnet18()
elif args.r==34:
    net=torchvision.models.resnet34()
elif args.r==50:
    net=torchvision.models.resnet50()

net.load_state_dict(torch.load(path))


def test(net, Quantized = False):
    net.eval()
    correct = 0
    total = 0
    t1 = perf_counter()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    t2=perf_counter()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Time used:',t2-t1)


if __name__=='__main__':
    print('Model is Resnet-',args.r,sep='')
    if Quantize:
        print('Quantizing model...')
        net=torch.quantization.quantize_dynamic(net,{torch.nn.Conv2d},dtype=torch.qint8)
        device = 'cpu'
    if TorchScript:
        print("Creating torchScript...")
        input_t=torch.randn(1,3,224,224)
        traced = torch.jit.trace(net,input_t)
        net = traced
        
    net.to(device)
    test(net,Quantize)