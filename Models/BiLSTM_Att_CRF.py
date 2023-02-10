import torch.nn as nn
from torchcrf import CRF

from config import *

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first = True,
            bidirectional = True
        )       # self - attention
        self.attention = nn.MultiheadAttention(2*HIDDEN_SIZE, 4)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(2*HIDDEN_SIZE, LABEL_SIZE)
        self.crf = CRF(LABEL_SIZE, batch_first=True)
    
    def forward(self, input, mask):   
        output = self.get_lstm_results(input)
        output, _ = self.attention(output, output, output)
        output = self.linear(output)
        output = self.crf.decode(output, mask)
        return output

    def get_lstm_results(self, input):
        output = self.embed(input)
        output, _ = self.lstm(output)       # 只取hidden state
        return self.dropout(output)
    
    def loss_fn(self, input, label, mask):
        output = self.get_lstm_results(input)
        output, _ = self.attention(output, output, output)
        output = self.linear(output)
        if label != None:
            loss = self.crf(output, label, mask)
        return (-1)*loss