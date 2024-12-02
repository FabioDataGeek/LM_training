import torch
from transformers import AdamW
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import json
from architecture import *
from utils_github import *
from datasets_utils import *
from arguments import *
from datasets import load_dataset

BENCHMARKS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
    "super_glue": ["boolq", "cb", "copa", "rte", "wic", "wsc"],
}

for key in BENCHMARKS.keys():
    for dataset in BENCHMARKS[key]:
        args = arguments(key, dataset)
        print(f"Running {key} - {dataset}")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        all_data = {}
        model_tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
        benchmark_tokenizer = BENCHMARK_CLASSES[args["benchmark"]](
            args["task"], model_tokenizer
        )
        

        # DATASET
        benchmark = args["benchmark"]
        splits = BENCHMARK_MAPPER[benchmark][args["task"]]

        tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
        if args["task"] == "mnli":
            train_dataset = load_dataset(
                args["benchmark"], args["task"], split="train", trust_remote_code=True
            )
            dev_dataset = load_dataset(
                benchmark,
                args["task"],
                split="validation_matched",
                trust_remote_code=True,
            )
            test_dataset = load_dataset(
                benchmark, args["task"], split="test_matched", trust_remote_code=True
            )

        else:
            train_dataset = load_dataset(
                args["benchmark"], args["task"], split="train", trust_remote_code=True
            )
            dev_dataset = load_dataset(
                benchmark, args["task"], split="dev", trust_remote_code=True
            )
            test_dataset = load_dataset(
                benchmark, args["task"], split="test", trust_remote_code=True
            )

        # Apply tokenization
        train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
        dev_dataset = dev_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
        test_dataset = test_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)

        # Set the format for PyTorch
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "label"],
        )
        dev_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "label"],
        )
        test_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "label"],
        )

        # DATALOADER
        if args["task"] == "mnli":
            train_loader = DataLoader(
                train_dataset,
                batch_size=args["batch_size"],
                shuffle=True,
                generator=torch.Generator(device="cuda:0"),
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                generator=torch.Generator(device="cuda:0"),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                generator=torch.Generator(device="cuda:0"),
            )

        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args["batch_size"],
                shuffle=True,
                generator=torch.Generator(device="cuda:0"),
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                generator=torch.Generator(device="cuda:0"),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                generator=torch.Generator(device="cuda:0"),
            )

        # MODEL
        model = BertForSequenceClassification.from_pretrained(args["model_name"])
        model.config.num_labels = NUMBER_OF_LABELS[args["task"]]

        # DEVICE
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available, using CPU instead.")

        total_params = parameters(model=model)

        print("parameters:")
        # Count the number of trainable parameters in the model
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    "\t{:45}\ttrainable\t{}\tdevice:{}".format(
                        name, param.size(), param.device
                    )
                )
            else:
                print(
                    "\t{:45}\tfixed\t{}\tdevice:{}".format(
                        name, param.size(), param.device
                    )
                )

        num_params = sum(
            p.numel() for name, p in model.named_parameters() if p.requires_grad
        )
        print("\ttotal:", num_params)

        # Optimizer
        optimizer = AdamW(
            params=total_params["params"],
            lr=args["lr"],
            weight_decay=args["weight_decay"],
            eps=1e-8,
        )

        scheduler = select_scheduler(
            optimizer=optimizer,
            lr_scheduler="warmup_linear",
            num_epochs=10,
            num_batches=len(train_loader),
            warmup_proportion=args["warmup_proportion"],
        )

        for i in tqdm(range(len(args["num_epochs"])), desc="Epoch"):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                problem_type=PROBLEM_TYPE[args["task"]],
                epoch=i,
            )
            evaluation(
                model,
                dev_loader,
                device,
                problem_type=PROBLEM_TYPE[args["task"]],
                epoch=i,
            )
            test(
                model,
                test_loader,
                device,
                problem_type=PROBLEM_TYPE[args["task"]],
                epoch=i,
            )
