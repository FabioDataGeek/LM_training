import torch

GLUE_TASKS_SPLITS = {
    'cola': ['train', 'validation', 'test'],
    'mnli': ['train', 'validation_matched', 'validation_mismatched'],
    'mrpc': ['train', 'validation', 'test'],
    'qnli': ['train', 'validation', 'test'],
    'qqp': ['train', 'validation', 'test'],
    'rte': ['train', 'validation', 'test'],
    'sst2': ['train', 'validation', 'test'],
    'stsb': ['train', 'validation', 'test'],
    'wnli': ['train', 'validation', 'test'],
}

SUPERGLUE_TASKS_SPLITS = {
    'boolq': ['train', 'validation', 'test'],
    'cb': ['train', 'validation', 'test'],
    'copa': ['train', 'validation', 'test'],
    'rte': ['train', 'validation', 'test'],
    'wic': ['train', 'validation', 'test'],
    'wsc': ['train', 'validation', 'test'],
}

BENCHMARK_MAPPER = {
    'glue': GLUE_TASKS_SPLITS,
    'super_glue': SUPERGLUE_TASKS_SPLITS,
}

NUMBER_OF_LABELS = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'sst2': 2,
    'stsb': 1,
    'wnli': 2,
    'ax': 2,
    'boolq': 2,
    'cb': 3,
    'copa': 2,
    'wic': 2,
    'wsc': 2,
}

# type of problem according to huggingface
PROBLEM_TYPE = {
    'cola': 'binary_classification',
    'mnli': 'multi_class_classification',
    'mrpc': 'binary_classification',
    'qnli': 'binary_classification',
    'qqp': 'binary_classification',
    'rte': 'binary_classification',
    'sst2': 'binary_classification',
    'stsb': 'regression',
    'wnli': 'binary_classification',
    'ax': 'binary_classification',
    'boolq': 'binary_classification',
    'cb': 'multi_class_classification',
    'copa': 'binary_classification',
    'wic': 'binary_classification',
    'wsc': 'binary_classification',
}