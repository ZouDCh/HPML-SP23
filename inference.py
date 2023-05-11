import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from nlp_utils import WikiFactDataset
import os
import sys
from tqdm import tqdm
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--quant', action='store_true')
parser.add_argument('--script', action='store_true')
args = parser.parse_args()

def eval(model, dataloader, device, ttype="dev"):
    model.eval()
    dev_acc = 0
    dev_loss = 0
    total, correct = 0, 0
    dev_data_size = len(dataloader)
    
    # Put this perf_counter() here to exclude those operations related to our perf profiling.
    epoch_start = time.perf_counter()  
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            logits = model(inputs)[0]

            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    
    epoch_end = time.perf_counter()

    epoch_time = epoch_end - epoch_start
    print('Setting : ', args.model_path, 'Device: ', device, 'Quant: ', args.quant, 'Script: ', args.script)
    print('Test Accuracy: ', correct/total, 'Time:', epoch_time)
    with open("inference.log", "a") as f:
        f.writelines(f"========================{args.model_path}========================\n")
        f.writelines(f"Setting : {args.model_path} Device: {device} Quant: {args.quant} Script: {args.script}\n")
        f.writelines(f"Total {ttype} time: {epoch_time}\n")
        f.writelines(f"Accuracy for {ttype}: {correct / total}\n")
        f.close()

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if args.script:
        model = RobertaForSequenceClassification.from_pretrained(args.model_path, torchscript=True)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    if args.quant:
        model = torch.quantization.quantize_dynamic(model,{torch.nn.Conv2d},dtype=torch.qint8)
        
    model.to(device)

    eval_dataset = WikiFactDataset("./wiki/fact_classification/dev")
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    eval(model, eval_dataloader, device)