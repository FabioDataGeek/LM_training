# Traditional fine-tuning with Pytorch
This folder contains the code needed to perform fine-tuning without considering distributed training and unfreeze all the parameters from the model to train.

Although it is inefficient considering modern fine-tuning approaches, it is easier to implement in Pytorch and quite common for small models. 
This is a good approach if you want to train small models, i.e., models with less than 300 million parameters, with small datasets.
However, suppose your unfrozen model only fits in one GPU. In that case, you want to accelerate the fine-tuning with multiple GPUs or make it more efficient, so you should consider other approaches:

- Your model fits in one single GPU but wants to accelerate with multiple GPUs, use distributed data parallel (DDP) in Pytorch.
- Your entire model doesn't fit in one GPU: use Fully Sharded Data Parallel (FSDP) in Pytorch.
- You want to make the fine-tunning more efficient, getting results as good or almost as traditional fine-tuning but with fewer resources: try parameter efficient fine-tuning techniques like LoRA,
  GaLore or IA3, unfroze specific layers inside your model instead of the entire model or quantize your model so each parameter uses less resources.
