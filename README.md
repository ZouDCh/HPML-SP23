Resnet-1.py is for 1 GPU with baseline and mixed precision.
```
python Resnet-1.py
```

Resnet-2.py is for dataparallel with 1,2,4 GPUs.
```
torchrun Resnet-2.py [18/34/50] [1/2/4] [0/1]
```
The first argument determines size of the model, the second indicates number of GPUs, and the third for turing on or off mixed precision.

Resnet-3.py is for Distributed dataparallel for 1,2,4 GPUs.
```
torchrun --nproc-per-node [1/2/4] Resnet-3.py [18/34/50] [1/2/4] [0/1]
```
The first argument determines size of the model, the second indicates number of GPUs, and the third for turing on or off mixed precision. The `--nproc-per-node` must match the number of GPUs.

Resnet-4.py is for evaluating torchscript and Quantization on CPUs.
```
python Resnet-4.py -q [0/1] -r [18/34/50] -m [0/1] -ts [0/1]
```
It will try to load the models from the models/ directory, and -q -r -m -ts specifies the quantization, resnet size, mixed precision and using torchscript.

Resnet-5 is for evaluating torchscript on GPU.
```
python Resnet-4.py -m [0/1] -r [18/34/50]
```
The -m refers to mixed precisoin, and -r refers to Resnet size. The program will use the specified model with and without Torchscript.