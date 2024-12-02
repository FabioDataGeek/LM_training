from transformers import AutoTokenizer


LABEL_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}



def squeeze_inputs(example):
    example['input_ids'] = example['input_ids'].squeeze(1)
    example['attention_mask'] = example['attention_mask'].squeeze(1)
    example['token_type_ids'] = example['token_type_ids'].squeeze(1)
    return example


def convert_labels_to_numbers(examples):
    for example in range(len(examples['answer'])):
        examples['answer'][example] = LABEL_MAPPING[examples['answer'][example]]
    return examples


# FOR GLUE DATASET
class GLUE_Dataset_Tokenizers():
    def __init__(self, dataset_name, tokenizer):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        #self.tokenizer.model_max_length = 128  # Set the tokenizer's max length to 128

    def tokenize_ax(self, examples):
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_qnli(self, examples):
        return self.tokenizer(examples['question'], examples['sentence'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_rte(self, examples):
        tokenized_examples = self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)
        return tokenized_examples

    # Binary Classification
    def tokenize_wnli(self, examples):
        return self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_sst2(self, examples):
        return self.tokenizer(examples['sentence'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_mrpc(self, examples):
        return self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_cola(self, examples):
        return self.tokenizer(examples['sentence'], 
                            truncation=True,
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_qqp(self, examples):
        return self.tokenizer(examples['question1'], examples['question2'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Multiclass Classification
    def tokenize_mnli(self, examples):
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Regression
    def tokenize_stsb(self, examples):
        return self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    GLUE_DATASET_DICT = {
    'qnli': tokenize_qnli,
    'rte': tokenize_rte,
    'wnli': tokenize_wnli,
    'sst2': tokenize_sst2,
    'mrpc': tokenize_mrpc,
    'cola': tokenize_cola,
    'qqp': tokenize_qqp,
    'mnli': tokenize_mnli,
    'stsb': tokenize_stsb,
    'ax': tokenize_ax
    }

    def get_tokenizer(self):
        tokenize_func = self.GLUE_DATASET_DICT[self.dataset_name]
        def wrapper(examples):
            return tokenize_func(self, examples)
        return wrapper
    

# FOR SUPERGLUE DATASET
class SUPERGLUE_Dataset_Tokenizers():
    def __init__(self, dataset_name, tokenizer):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    # Binary Classification
    def tokenize_boolq(self, examples):
        return self.tokenizer(examples['question'], examples['passage'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Multiclass Classification
    def tokenize_cb(self, examples):
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_copa(self, examples):
        texts1, texts2 = [], []
        for p, c1, c2 in zip(examples['premise'], examples['choice1'], examples['choice2']):
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                texts1.append(p)
                texts2.append(c1 + '[SEP]' + c2)
            else:
                return_token_type_ids = False
                texts1.append(p)
                texts2.append(c1 + ' ' + c2)
        tokenized_texts = self.tokenizer(texts1, texts2, truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

        # Squeeze the tensors
        for key in tokenized_texts.keys():
            tokenized_texts[key] = tokenized_texts[key].squeeze(1)
    
        return tokenized_texts

    # Binary Classification
    def tokenize_rte(self, examples):
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)

    # Binary Classification
    def tokenize_wic(self, examples):
        texts1 = []
        texts2 = []
        for w, s1, s2, label in zip(examples['word'], examples['sentence1'], examples['sentence2'], examples['label']):
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                texts1.append(w)
                texts2.append(s1 + '[SEP]' + s2)
            else:
                return_token_type_ids = False
                texts1.append(w)
                texts2.append(s1 + ' ' + s2)
        
        
        return self.tokenizer(texts1, texts2, truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_wsc(self, examples):
        return self.tokenizer(examples['text'], 
                            truncation=True,  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True)
    
    SUPERGLUE_DATASET_DICT = {
        'boolq': tokenize_boolq,
        'cb': tokenize_cb,
        'copa': tokenize_copa,
        'rte': tokenize_rte,
        'wic': tokenize_wic,
        'wsc': tokenize_wsc
    }

    def get_tokenizer(self):
        tokenize_func = self.SUPERGLUE_DATASET_DICT[self.dataset_name]
        def wrapper(examples):
            return tokenize_func(self, examples)
        return wrapper