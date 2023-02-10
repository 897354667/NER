import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from config import *

class Bert_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.lstm = nn.LSTM(
            768,
            HIDDEN_SIZE,
            batch_first = True,
            bidirectional = True
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(HIDDEN_SIZE*2, LABEL_SIZE)
        self.crf = CRF(LABEL_SIZE, batch_first=True)
        
    def forward(self, input_ids, labels=None, mask=None):
        att_mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        # bert_output: (last_hidden_state, pooler_output)
        sequence_output = bert_output[0]    # (last_hidden_state, pooler_output)
        # sequence_output: (batch_size, seq_len, embed_dim)
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        # sequence_output: (batch_size, seq_len, HIDDEN_SIZE*2)
        logits = self.linear(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE)
        tags = self.crf.decode(logits, att_mask)
        return tags 
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        # bert_output: (last_hidden_state, pooler_output)
        sequence_output = bert_output[0]      # (last_hidden_state, pooler_output)
        # sequence_output: (batch_size, seq_len, embed_dim)
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        # sequence_output: (batch_size, seq_len, HIDDEN_SIZE*2)
        logits = self.linear(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE)
        if label_ids != None:
            loss = self.crf(logits, label_ids, att_mask)
        return loss*(-1)     
