import torch
from tqdm import tqdm
import json
from sklearn.metrics import classification_report
from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from arguments import *
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import one_hot
import time

# PARAMETERS
def parameters(model):
    params = {'params': [p for n, p in model.named_parameters() if p.requires_grad]}
    return params

# SCHEDULER UTILS
def calculate_warmup_steps(total_epochs, num_batches, warmup_proportion):
    total_steps = total_epochs * num_batches
    warmup_steps = int(total_steps * warmup_proportion)
    return warmup_steps

def select_scheduler(optimizer, lr_scheduler, num_epochs, num_batches, warmup_proportion):
    if lr_scheduler == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif lr_scheduler == 'warmup_constant':
            warmup_steps = calculate_warmup_steps(total_epochs=num_epochs, num_batches=num_batches, warmup_proportion=warmup_proportion)
            scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps= warmup_steps)
    elif lr_scheduler == 'warmup_linear':
        warmup_steps = calculate_warmup_steps(total_epochs=num_epochs, num_batches=num_batches, warmup_proportion=warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_batches*num_epochs)
    return scheduler

def reporting_regression(y_true_list, y_pred_list, epoch, dict_name, dict={}):
    mse = mean_squared_error(y_true_list, y_pred_list)
    r2 = r2_score(y_true_list, y_pred_list)
    pearson_corr, _ = pearsonr(y_true_list, y_pred_list)
    spearman_corr, _ = spearmanr(y_true_list, y_pred_list)
    print(f"{dict_name} REGRESSION REPORT:")
    print(f'Mean Squared Error: {mse}, R-Squared: {r2}, Pearson Correlation: {pearson_corr}, Spearman Correlation')
    dict = {'mse': mse, 'r2': r2, 'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr}
    with open(f"{dict_name}{epoch}.json", 'w') as fp:
        json.dump(dict, fp)
    return dict

def reporting_classification(y_true_list, y_pred_list, epoch, dict_name, dict={}):
    report = classification_report(y_true_list, y_pred_list, output_dict=True)
    print(f"{dict_name} CLASSIFICATION REPORT: ")
    print(classification_report(y_true_list, y_pred_list))
    dict = report
    with open(f"{dict_name}{epoch}.json", 'w') as fp:
        json.dump(dict, fp)
    return report


def train(model, train_loader, optimizer, scheduler, device, problem_type, epoch):
    log_interval = 10
    global_step = 0
    start_time = time.time()
    
    # Set the model in training mode for the dropout and normalization
    model.train()
    y_true_list = []
    y_pred_list = []
    
    # Iterate through the training batches of our training dataloader
    for i, batch in tqdm(enumerate(train_loader)):
        
        # Optimizer gradient set to zero
        optimizer.zero_grad()

        """ We take all the input data for our model according to the Dataset
        class __getitem__ function implemented, here we consider the output
        of the tokenizer. Beware of setting all the needed tensors to the same
        device where your model is loaded, in our case is 'cuda:0' """
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.float()
        

        """ In HuggingFace models the output tensors are inside a 
        list in the first term, we will squeeze them in case they have an
        additional dimension like [1, batch_size, output_size] instead of
        [batch_size, output_size]"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]
        outputs = outputs.squeeze()

        """ We need to set the labels in specific formats according to the 
        task we are dealing with, in regression there are no modifications as
        the labels obtained for our problem are in float numbers. We will 
        set the loss function here too"""
        if problem_type == 'binary_classification':
            labels = labels.long()
            labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
            loss_fn = torch.nn.BCEWithLogitsLoss()      

        if problem_type == 'multi_class_classification':
            labels = labels.long()
            labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
            loss_fn = torch.nn.CrossEntropyLoss()          

        if problem_type == 'regression':
            loss_fn = torch.nn.MSELoss()

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # backpropagation
        loss.backward()

        # LOGS
        loss = loss.item()
        ms_per_batch = 1000 * (time.time() - start_time) / log_interval
        print('|step {:5} |lr: {:9.7f} |loss {:7.4f} |ms/batch {:7.2f}|'.format(global_step, scheduler.get_last_lr()[0], loss, ms_per_batch))

        global_step += 1

        # new step for the optimizer
        optimizer.step()

        """ This part is only necessary if we don't have a fixed scheduler, 
        i.e., the learning rate changes during training process"""
        if scheduler is not None:
            scheduler.step()

        # Set the labels and outputs in recommended format for the reports
        if problem_type == 'regression':
            labels = labels.float()
            y_true_list.append(labels)
            y_pred_list.append(outputs)
        elif 'classification' in problem_type:
            y_true_list.append(labels)
            y_pred_list.append(outputs)
        else:
            raise ValueError('Problem type not implemented')
        
    # Get the reports
    if problem_type == 'regression':
        reporting_regression(y_true_list, y_pred_list, epoch, dict_name='report_dict_train')
    elif 'classification' in problem_type:
        reporting_classification(y_true_list, y_pred_list, epoch, dict_name='report_dict_train')
    else:
        raise ValueError('Problem type not implemented')


def evaluation(model, dev_loader, device, problem_type, epoch):
    
    # Set the model in evaluation mode for dropout and regularization
    model.eval()
    y_true_list = []
    y_pred_list = []
    """ With this command all the code inside the loop will perform
    every operation without gradient"""
    with torch.no_grad():
        
        # Iterate through the batches of our development dataloader
        for batch in tqdm(dev_loader):
            
            # Get all the input data from the batch and send to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = batch['labels']
                
            # Output of the language model in format [batch_size, ouput_size]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            outputs = outputs.squeeze(1)
            
        
            # Set the labels format and loss function
            if problem_type == 'binary_classification':
                        labels = labels.long()
                        labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
                        loss_fn = torch.nn.BCEWithLogitsLoss()      
    
            if problem_type == 'multi_class_classification':
                labels = labels.long()
                labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
                loss_fn = torch.nn.CrossEntropyLoss()          
    
            if problem_type == 'regression':
                loss_fn = torch.nn.MSELoss()
            
            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Set the labels and outputs in recommended format for the reports
            if problem_type == 'regression':
                labels = labels.float()
                y_true_list.append(labels)
                y_pred_list.append(outputs)
            elif 'classification' in problem_type:
                y_true_list.append(labels)
                y_pred_list.append(outputs)
            else:
                raise ValueError('Problem type not implemented')

    # Get the reports
    if problem_type == 'regression':
        reporting_regression(y_true_list, y_pred_list, epoch, dict_name='report_dict_test')
    elif 'classification' in problem_type:
        reporting_classification(y_true_list, y_pred_list, epoch, dict_name='report_dict_test')
    else:
        raise ValueError('Problem type not implemented')


def test(model, test_loader, device, problem_type, epoch):
    # Set the model in evaluation mode for dropout and regularization
    model.eval()
    
    """ With this command all the code inside the loop will perform
    every operation without gradient"""
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():

        # Iterate through the batches of our test dataloader
        for batch in tqdm(test_loader):
            
            # Get all the input data from the batch and send to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.labels.to(device)
            labels = batch['labels']

            # Output of the language model in format [batch_size, ouput_size]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            outputs = outputs.squeeze()   

            # Set the labels format
            if problem_type == 'binary_classification':
                        labels = labels.long()
                        labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
                                    
            if problem_type == 'multi_class_classification':
                labels = labels.long()
                labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
    
            # Set the labels and outputs in recommended format for the reports
            if problem_type == 'regression':
                labels = labels.float()
                y_true_list.append(labels)
                y_pred_list.append(outputs)
            elif 'classification' in problem_type:
                y_true_list.append(labels)
                y_pred_list.append(outputs)
            else:
                raise ValueError('Problem type not implemented')


    # Get the reports
    if problem_type == 'regression':
        reporting_regression(y_true_list, y_pred_list, epoch, dict_name='report_dict_test')
    elif 'classification' in problem_type:
        reporting_classification(y_true_list, y_pred_list, epoch, dict_name='report_dict_test')
    else:
        raise ValueError('Problem type not implemented')
