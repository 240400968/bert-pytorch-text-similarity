"""
Utility functions for training and validating models.
"""

import time
import torch

import torch.nn as nn
import codecs, sys
from tqdm import tqdm
from bert.utils import correct_predictions
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np

def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, probs = model(input_ids,
                              token_type_ids,
                              attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    
    y_p_float = []
    y_t = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(input_ids,
                                  token_type_ids,
                                  attention_mask)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

            for item, item3 in zip(probs[:, 1], labels):
                y_p_float.append(item.item())
                y_t.append(int(item3.item()))

    precisions, recalls, f1s = [], [], []        
    for i in range(0, 1000, 1):
        precision, recall, f1 = evalue(y_p_float, y_t, i/1000.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy, precisions, recalls, f1s


def test(model, dataloader, data, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    print("test_data:", data[0])
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    
    outputs = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(input_ids,
                                  token_type_ids,
                                  attention_mask)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
            outputs.extend([item for item in probs[:, 1]])
          
    # old format: sent1, sent2, label, qid, docid, seg_query, doc_seg
    # new format: [ input_ids, token_type_ids, attention_mask, label, query_id, doc_id, sentence1, sentence2]
    outputs = np.array(outputs)
    data = np.array(data)
    query_indexes = group_queries(data, 4)
    top1_num = 0.0
    top3_num = 0.0
    top5_num = 0.0
    top10_num = 0.0
    k = 10
    predictions = []
    for query in query_indexes:
        results = outputs[query_indexes[query]]
        predicted_sorted_indexs = np.argsort(results)[::-1]
        t_results = data[query_indexes[query], 3]
        t_results = t_results[predicted_sorted_indexs]
        query_doc_results = data[query_indexes[query]]
        query_doc_results = query_doc_results[predicted_sorted_indexs]
        results_sorted = results[predicted_sorted_indexs]
        predictions_query = [str(item[4])+"\t"+str(item[5])+"\t"+str(predict_score)+"\t"+str(item[3])+"\t"+item[6]+"\t"+item[7] for item, predict_score, true_score_ in zip(query_doc_results, results_sorted, t_results)]
        predictions.extend(predictions_query)
        rank_sent_pos = t_results.tolist().index(1.0)
        docids = query_doc_results[:rank_sent_pos+1, 5]
        uniq_docids = []
        for item in docids:
            if item not in uniq_docids:
                uniq_docids.append(item)
        rank_pos = len(uniq_docids)
        if rank_pos == 1:
            top1_num += 1
            top3_num += 1
            top5_num += 1 
            top10_num += 1                       
        elif rank_pos >1 and rank_pos <= 3:
            top3_num += 1
            top5_num += 1 
            top10_num += 1             
        elif rank_pos > 3 and rank_pos <=5:
            top5_num += 1
            top10_num += 1               
        elif rank_pos >5 and rank_pos <=10:
            top10_num += 1
    query_num = len(query_indexes.keys())
    top1_num /= query_num
    top3_num /= query_num
    top5_num /= query_num
    top10_num /= query_num
    with codecs.open("../../data/preprocessed/text_similarity/test_predict", 'w', encoding='utf-8') as f:
        f.write(u"\n".join(predictions))
        f.write("\n")
        f.write("Final test, top1 acc: "+str(top1_num)+" top3 acc: "+str(top3_num)+" top5 acc: "+str(top5_num)+" top10 acc: "+str(top10_num))
    print("Final test,  top1 acc: "+str(top1_num)+" top3 acc: "+str(top3_num)+" top5 acc: "+str(top5_num)+" top10 acc: "+str(top10_num)) 
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy



def evalue(y_p_float, y_t, threshold):
    y_p_int = []
    for item in y_p_float:
        if item > threshold:
            y_p_int.append(1)
        else:
            y_p_int.append(0)

    return precision_score(y_t, y_p_int), recall_score(y_t, y_p_int), f1_score(y_t, y_p_int)
    
def group_queries(training_data, qid_index):
    """
        Returns a dictionary that groups the documents by their query ids.
        Parameters
        ----------
        training_data : Numpy array of lists
            Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
        qid_index : int
            This is the index where the qid is located in the training data
        
        Returns
        -------
        query_indexes : dictionary
            The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
    """
    query_indexes = {}
    index = 0
    for record in training_data:
        query_indexes.setdefault(record[qid_index], [])
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes
