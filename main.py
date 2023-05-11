# Refer to https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T
from transformers import RobertaForSequenceClassification
import os
import argparse
from tqdm import tqdm
import time
from resnet import ResNet18
from utils import progress_bar
from nlp_utils import WikiFactDataset

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--model_name', default='roberta-base', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--mixed', action='store_true')
args = parser.parse_args()

# Data loading
train_set = WikiFactDataset("./wiki/fact_classification/train")
eval_set = WikiFactDataset("./wiki/fact_classification/dev")
total_data_size = len(train_set)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt

def main(batch_size, local_rank):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if n_gpu > 1 else None
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=(train_sampler is None), sampler=train_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set) if n_gpu > 1 else None
    eval_loader = torch.utils.data.DataLoader(eval_set, shuffle=False, sampler=eval_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    scale = torch.cuda.amp.GradScaler()

    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=4).to(device)
    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)    

    def eval(epoch, dataloader, local_rank):
        print('\nEval Epoch: %d' % epoch)
        # Change the trainloader in that way, so that we could profiling the runtime performance of data loading
        # And seems it shouldn't happen in regular training, I just try to exclude them in the time of epoch.
        iter_loader = iter(dataloader)
        eval_data_size = len(iter_loader)
        
        # Put this perf_counter() here to exclude those operations related to our perf profiling.
        epoch_start = time.perf_counter()  

        model.eval()
        correct = 0
        total = 0
        if local_rank == 0:
            bar = tqdm(range(eval_data_size))
        else:
            bar = range(eval_data_size)
        for batch_idx in bar:
            # data loading
            d = next(iter_loader) 
            inputs = d["inputs"]
            labels = d["targets"]
            inputs, labels = inputs.to(device), labels.to(device)

            # compute
            if args.mixed:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs).logits
            else:
                outputs = model(inputs).logits

            # metric calculation          
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_end = time.perf_counter()

        epoch_time = epoch_end - epoch_start

        if n_gpu > 1:
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
            accuracy_tensor = torch.tensor([correct / total], dtype=torch.float32, device=local_rank)
            torch.distributed.all_reduce(accuracy_tensor)
            accuracy = accuracy_tensor.item() / world_size
        else:
            accuracy = correct / total

        if local_rank == 0:
            f = open(f"perf-{args.model_name.split('/')[-1]}.log", "a")
            f.writelines(f"Total running time for epoch {epoch} @ {n_gpu}-gpu {batch_size}-batch-size: {epoch_time}\n")
            f.writelines(f"Eval for epoch {epoch} @ {n_gpu}-gpu {batch_size}-batch-size: {accuracy}\n")
            f.close()
        return accuracy


    def train(epoch, dataloader, local_rank):
        model.train()
        print('\nEpoch: %d' % epoch)
        # Change the trainloader in that way, so that we could profiling the runtime performance of data loading
        # And seems it shouldn't happen in regular training, I just try to exclude them in the time of epoch.
        iter_trainloader = iter(dataloader)
        train_data_size = len(iter_trainloader)
        
        # Put this perf_counter() here to exclude those operations related to our perf profiling.
        epoch_start = time.perf_counter()  

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        # bar = tqdm(range(train_data_size))
        if local_rank == 0:
            bar = tqdm(range(train_data_size))
        else:
            bar = range(train_data_size)
        for batch_idx in bar:
            # data loading
            d = next(iter_trainloader) 
            inputs = d["inputs"]
            labels = d["targets"]
            inputs, labels = inputs.to(device), labels.to(device)

            # compute
            
            if args.mixed:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)
                    if n_gpu > 1:
                        torch.distributed.barrier()
                        loss = loss.mean()
                scale.scale(loss).backward()
                scale.step(optimizer)
                scale.update()
            else:
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                if n_gpu > 1:
                    torch.distributed.barrier()
                    loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # metric calculation            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_end = time.perf_counter()

        epoch_time = epoch_end - epoch_start

        if n_gpu > 1:
            torch.distributed.barrier()
            world_size = torch.distributed.get_world_size()
            accuracy_tensor = torch.tensor([correct / total], dtype=torch.float32, device=local_rank)
            torch.distributed.all_reduce(accuracy_tensor)
            accuracy = accuracy_tensor.item() / world_size

            loss_tensor = torch.tensor([train_loss / train_data_size], dtype=torch.float32, device=local_rank)
            torch.distributed.all_reduce(loss_tensor)
            loss = loss_tensor.item() / world_size
        else:
            accuracy = correct / total
            loss = train_loss / train_data_size

        if local_rank == 0:
            f = open(f"perf-{args.model_name.split('/')[-1]}.log", "a")
            f.writelines(f"Total running time for epoch {epoch} @ {n_gpu}-gpu {batch_size}-batch-size: {epoch_time}\n")
            f.writelines(f"Loss for epoch {epoch} @ {n_gpu}-gpu {batch_size}-batch-size: {loss}\n")
            f.writelines(f"Accuracy for epoch {epoch} @ {n_gpu}-gpu {batch_size}-batch-size: {accuracy}\n")
            f.close()
        
    best_acc = 0
    for epoch in range(args.epochs):
        train(epoch, train_loader, local_rank)
        acc = eval(epoch, eval_loader, local_rank)
        if local_rank == 0 and acc > best_acc:
            best_acc = acc
            output_dir = f"checkpoint-{args.model_name.split('/')[-1]}-{args.batch_size}-{n_gpu}-gpu-{args.mixed}-mixed"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)

if __name__ == "__main__":
    # Device configuration
    assert int(os.environ['LOCAL_RANK']) == args.local_rank, f"LOCAL_RANK {os.environ['LOCAL_RANK']} does not match args.local_rank {args.local_rank}"
    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    torch.cuda.set_device(local_rank)
    n_gpu = torch.cuda.device_count()
    device = local_rank % n_gpu if n_gpu > 0 else torch.device("cpu")

    if n_gpu > 1:
        torch.distributed.init_process_group(backend='nccl')
    print("n_gpu, local_rank: ", n_gpu, local_rank)

    main(args.batch_size, local_rank)