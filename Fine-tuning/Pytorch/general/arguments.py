import argparse
from dataset_tokenizers import *

# Initialize the parser

BENCHMARK_CLASSES = {
    'glue': GLUE_Dataset_Tokenizers,
    'super_glue': SUPERGLUE_Dataset_Tokenizers,
}

def arguments(benchmark, task):
    parser = argparse.ArgumentParser(description="Deep Learning Model Parameters")

    # Data parameters
    parser.add_argument('--benchmark', type=str, default=benchmark, help='Benchmark') # [glue, superglue]
    parser.add_argument('--task', type=str, default=task, help='Task') # [cola, mnli, mnli_mismatched, mnli_matched, mrpc, qnli, qqp, rte, sst2, stsb, wnli, ax]
    
    # LM parameters
    parser.add_argument('--model_name', type=str, default="google-bert/bert-base-uncased", help='Model name') # [pysentimiento/robertuito-base-uncased, bertin-project/bertin-roberta-base-spanish, PlanTL-GOB-ES/roberta-base-bne, scjnugacj/jurisbert, xlm-roberta-base]
    parser.add_argument('--model_type', type=str, default='bert', help='Model type')

    # scheduler & optimizer parameters
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='warmup_linear', help='Learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')   #1e-5 mejor de momento, en lora en teoría el learning rate tiene que ser más alto (un orden de magnitud más alto), lo mismo para Galore
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--warmup_proportion', type=float, default=0.06, help='Warmup proportion')

    # device parameters
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--devices', type=str, default='cuda:0', help='Devices to use')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Parse the arguments
    args = parser.parse_args()
    all_data = vars(args)
    return all_data