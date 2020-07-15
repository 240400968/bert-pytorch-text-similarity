# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer

class BERT(nn.Module):

    def __init__(self, bert_path, 
                 hidden_size=768, 
                 num_classes=2,
                 device="cpu"):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, input_ids, token_type_ids, attention_mask):
        # input_ids = x[0]  # 输入的句子[batch_size, sequence_length]
        # token_type_ids = x[1]   # [batch_size, sequence_length], 0对应sentenceA，1对应sentenceB
        # attention_mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        logits = self.fc(pooled)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities
