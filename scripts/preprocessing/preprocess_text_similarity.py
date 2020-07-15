# -*- coding: utf-8 -*-
import torch
from pytorch_pretrained_bert import BertTokenizer

import pickle as  cPickle
import json
import jieba
import random
import codecs, sys
import time
import numpy as np
import itertools
import gensim
from itertools import combinations
import os

bert_path = "../../data/bert_pretrain/"
tokenizer = BertTokenizer(bert_path+"vocab.txt")
padding_size = 128

def seg_line(line):
    return list(jieba.cut(line))

def seg_data_for_train(path, docid_doc, rate=8):
    flog = open(path+".log", "w")
    dirpath = os.path.dirname(path)
    docid_sentids = {}
    sentid_query = {}
    sid = 0
    docids = docid_doc.keys()
    for docid in docids:
        sentid_query[sid] = tokenizer.tokenize(docid_doc[docid])
        docid_sentids[docid] = [sid]
        sid += 1
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    len_data = len(lines)
    for line in lines:
        split = line.strip().split('##')
        query = split[0].replace(" ", "")
        docid = int(split[1])
        sentid_query[sid] = tokenizer.tokenize(query)
        docid_sentids[docid].append(sid)
        sid += 1
    train_set = []
    valid_set = []
    total_sent_id = sid
    print("total_sent_id:", total_sent_id)
    for docid in docids:
        pos_pair = list(combinations(docid_sentids[docid], 2))
        random.shuffle(pos_pair)
        pos_pair = pos_pair if len(pos_pair)<10 else pos_pair[:10]
        pos_num = len(pos_pair)
        sents_num = len(docid_sentids[docid])
        pos_samples = [bert_sample(sentid_query[pair[0]], sentid_query[pair[1]], 1.0) for pair in pos_pair]
        train_set.extend(pos_samples[0:int(len(pos_pair)*0.8)])
        valid_set.extend(pos_samples[int(len(pos_pair)*0.8):])
        other_docid_sentids = [sid for sid in list(range(total_sent_id)) if sid not in docid_sentids[docid]]
        random.shuffle(other_docid_sentids)
        other_docid_sentids = other_docid_sentids[:2*sents_num]
        neg_pair = list(itertools.product(docid_sentids[docid], other_docid_sentids))
        random.shuffle(neg_pair)
        neg_pair = neg_pair[:pos_num*1]
        neg_num = len(neg_pair)
        neg_samples = [ bert_sample(sentid_query[pair[0]], sentid_query[pair[1]], 0.0) for pair in neg_pair]
        train_set.extend(neg_samples[0:int(neg_num*0.8)])
        valid_set.extend(neg_samples[int(neg_num*0.8):])
        flog.write("train_set size:"+str(len(train_set)))
        flog.write("valid_set size:"+str(len(valid_set)))
        print("train_set size:", len(train_set))
        print("valid_set size:", len(valid_set))

    random.shuffle(train_set)
    random.shuffle(valid_set)
    print("train_set size:", len(train_set))
    print("valid_set size:", len(valid_set))
    flog.close()
    return train_set, valid_set

def bert_sample(s1, s2, label):
    '''
    return: [input_ids, token_type_ids, attention_mask, label]
    '''
    word_pieces = ["[CLS]"] + s1 + ["[SEP]"] 
    len_s1 = len(word_pieces)
    word_pieces = word_pieces + s2 + ["[SEP]"]
    input_length = len(word_pieces)
    len_s2 = input_length - len_s1
    input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
    token_type_ids = [0]*len_s1 + [1]*len_s2
    attention_mask = [1]*input_length
    if len(input_ids) <= padding_size:
        input_ids =  input_ids + [0] * (padding_size-input_length)
        token_type_ids = token_type_ids + [0] * (padding_size-input_length)
        attention_mask = attention_mask + [0] * (padding_size-input_length)
    else:
        input_ids = input_ids[:padding_size]
        token_type_ids = token_type_ids[:padding_size]
        attention_mask = attention_mask[:padding_size]
    return [input_ids, token_type_ids, attention_mask, label]



def transform_test_data_to_id(raw_data):
    '''
    input item: [query_id, doc_id, label, sentence1, sentence2, sentence1_seg, sentence2_seg ]
    return item: [ input_ids, token_type_ids, attention_mask, label, query_id, doc_id, sentence1, sentence2]
    # old return item: sent1, sent2, label, qid, docid, seg_query, doc_seg
    '''
    data = []
    for one in raw_data:
        x = bert_sample(one[5], one[6], one[2])
        data.append(x + one[0:2] + one[3:5])
    return data
# sent1, sent2, label, qid, docid, seg_query, doc_seg



def seg_data(path, docid_doc, keep_num=20):
    # 还款 的 计息 方式 是 等额 本息 吗##217673
    # label qid docid query doc
    print ('start process ', path)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        qid = 1
        lines = f.readlines()
        len_data = len(lines)
        for line in lines:
            split = line.strip().split('##')
            seg_query = split[0].split()  # word list of query
            docid = split[1]               # true docid
            docids_except = list(docid_doc.keys())
            docids_except.remove(docid)  # false docid list
            random.shuffle(docids_except)
            docids = docids_except[:keep_num-1]
            docids.insert(0, docid)
            #print("docids:", docids)
            #input()
            labels = [1.0] + [0.0] * (keep_num-1) 
            query_docs = [[label, qid, int(docid), seg_query, docid_doc[docid]] for docid, label in zip(docids, labels)]
            random.shuffle(query_docs)
            data.extend(query_docs)
            qid += 1
    return data

def seg_data_for_test(path, docid_doc):
    # 还款 的 计息 方式 是 等额 本息 吗##217673
    # label qid docid query doc
    print ('start process ', path)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        qid = 1
        lines = f.readlines()
        len_data = len(lines)
        for line in lines:
            split = line.strip().split('##')
            seg_query = split[0].split()  # word list of query
            docid = int(split[1])               # true docid
            docids_except = list(docid_doc.keys())
            #docids_except.sort()
            docids_except.remove(docid)  # false docid list
            random.shuffle(docids_except)
            docids = docids_except
            docids.insert(0, docid)
            labels = [1.0] + [0.0] * (len(docids)-1)
            query_docs = [[label, qid, docid, seg_query, docid_doc[docid]] for docid, label in zip(docids, labels)]
            random.shuffle(query_docs)
            data.extend(query_docs)
            qid += 1
    return data    


def seg_data_for_test_retrieve_train(test_path, docid_doc, docid_doc_seg, train_path):
    '''
    # 还款 的 计息 方式 是 等额 本息 吗##217673
    # label qid docid query doc
    return
        data item: [query_id, doc_id, label, sentence1, sentence2, sentence1_seg, sentence2_seg ]
    '''
    print ('start process ', test_path)
    data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        train_data = []
        random.shuffle(lines)   # 每个标准问有5个相似问和一个标准问参与测试query的检索
        docid_simi_count = {}
        for line in lines:
            split = line.strip().split('##')
            query = split[0].replace(" ","")
            docid = int(split[1])
            docid_simi_count[docid] = docid_simi_count[docid]+1 if docid in docid_simi_count else 1
            if docid_simi_count[docid]>2: continue
            train_data.append([docid, query, tokenizer.tokenize(query)])
    with open(test_path, 'r', encoding='utf-8') as f:
        qid = 1
        lines = f.readlines()
        len_data = len(lines)
        for line in lines:
            split = line.strip().split('##')
            query = split[0].replace(" ","")  # word list of query
            query_seg = tokenizer.tokenize(query)
            docid = int(split[1])               # true docid
            docids_except = list(docid_doc.keys())
            #docids_except.sort()
            docids_except.remove(docid)  # false docid list
            random.shuffle(docids_except)
            docids = docids_except
            docids.insert(0, docid)
            labels = [1.0] + [0.0] * (len(docids)-1)
            query_docs = [[qid, docid, label, query, docid_doc[docid], query_seg, docid_doc_seg[docid]] for docid, label in zip(docids, labels)]
            query_train_docs = [[qid, int(item[0]), 1.0, query, item[1], query_seg, item[2] ] if item[0]==docid else [qid, int(item[0]), 0.0, query, item[1], query_seg, item[2] ] for item in train_data]
            query_relevance = query_docs + query_train_docs
            random.shuffle(query_relevance)
            data.extend(query_relevance)
            qid += 1
    return data   

def build_docid_doc(id_docid_path):
    docid_doc = {}
    docid_doc_seg = {}
    with open(id_docid_path, 'r', encoding='utf-8') as f:
        for line in f:
            split = line.strip().split()
            docid_doc[int(split[1])] = split[2]
            docid_doc_seg[int(split[1])] = tokenizer.tokenize( split[2] ) 
    return docid_doc, docid_doc_seg

def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:2]]
    print ('word type size ', len(wordCount))
    return wordCount


def build_word2id(wordCount, threshold=5):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for word in wordCount:
        if wordCount[word] >= threshold:
            if word not in word2id:
                word2id[word] = len(word2id)
        else:
            chars = list(word)
            for char in chars:
                if char not in word2id:
                    word2id[char] = len(word2id)
    print ('processed word size ', len(word2id))
    return word2id


def split_valid_data(data):
    '''
    data item:[input_ids, token_type_ids, attention_mask, label]
    '''
    matched_data = []
    mismatched_data = []
    for item in data:
        score = int(item[3])
        if score == 1:
            matched_data.append(item)
        else:
            mismatched_data.append(item)
    return matched_data, mismatched_data


def process_data(data_path, rate):
    train_file_path = data_path + '/train.txt'
    test_file_path1 = data_path + '/valid.txt'
    id_docid_path = data_path + '/id_docId'
    train_data, matched_valid_data, mismatched_valid_data = _process_train_data(train_file_path, id_docid_path, rate)
    test_data1 = _process_test_data(test_file_path1,train_file_path, id_docid_path)
    with open('../../data/preprocessed/text_similarity/train.pickle', 'wb') as f:
        cPickle.dump(np.array(train_data), f)
    with open('../../data/preprocessed/text_similarity/matched_valid.pickle', 'wb') as f:
        cPickle.dump(np.array(matched_valid_data), f)
    with open('../../data/preprocessed/text_similarity/mismatched_valid.pickle', 'wb') as f:
        cPickle.dump(np.array(mismatched_valid_data), f)        
    with open('../../data/preprocessed/text_similarity/test1.pickle', 'wb') as f:
        cPickle.dump(np.array(test_data1), f)                          

def _process_train_data(train_file_path, id_docid_path, rate):
    '''
    data format:
    [question_wordidlist, doc_wordidlist, label, question_seg, doc_seg]
    '''
    docid_doc, docid_doc_seg = build_docid_doc(id_docid_path)
    train_set, valid_set = seg_data_for_train(train_file_path, docid_doc, rate)
    matched_valid_data, mismatched_valid_data = split_valid_data(valid_set)
    return train_set, matched_valid_data, mismatched_valid_data
    
def _process_test_data(test_file_path, train_file_path, id_docid_path):
    '''
    data format:
    [sent1_wordidlist, sent2_wordidlist, label, qid, docid, seg_query, doc_seg]
    '''
    docid_doc, docid_doc_seg = build_docid_doc(id_docid_path)
    test_set = seg_data_for_test_retrieve_train(test_file_path, docid_doc, docid_doc_seg, train_file_path) 
    test_data2id = transform_test_data_to_id(test_set)
    print("test data size:", len(test_data2id))
    return test_data2id

if __name__ == '__main__':

    process_data('./data/', 8)
    
