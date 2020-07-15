#!/bin/bash
set -u
set -e
train_test(){
    cd /data/exp/bert-pytorch-text-similarity/scripts/preprocessing
    ln -s ../../../bert_cls_duiqi/${1} data
    python preprocess_text_similarity.py > ../training/log${1}
    unlink data
    cd ../training
    python train_text_similarity.py >> log${1}
}

train_test anxin