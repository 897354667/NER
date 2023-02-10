import torch
import torch.utils.data as data

import pandas as pd

from config import *

class Dataset(data.Dataset):
    def __init__(self, data_path, data_set) -> None:
        super().__init__()
        if data_path:
            assert data_path in [TRAIN_PATH, TEST_PATH], 'no such data path!'
            self.data = pd.read_csv(data_path)
        else:
            self.data = data_set
        self.char2id = self.get_dict(VOCAB_PATH)
        self.label2id = self.get_dict(LABEL_PATH)

    def get_dict(self, dict_path):
        '''
        将存起来的vocab和labels文件转化成字典映射
        '''
        assert dict_path in [VOCAB_PATH, LABEL_PATH], 'wrong path!'
        df = pd.read_csv(dict_path, names=['char', 'id'])
        mapping = {}
        for i in range(1, len(df)):
            mapping[df['char'][i]] = df['id'][i]
        return mapping
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        text = self.data['text'][idx]
        label = self.data['BIO_anno'][idx]
        unk_id = UNK_ID     # 包外字符
        o_id = LABEL_PAD_ID     # 包外label
        text_num = [int(self.char2id.get(v, unk_id)) for v in list(text)]
        label_num = [int(self.label2id.get(v, o_id)) for v in label.split(' ')] 
        return text_num, label_num
    

def get_dataLoader(data_path, data_set):
    if data_path:
        assert data_path in [TRAIN_PATH, TEST_PATH], 'wrong data path!'
    def collate_fn(batch):
        '''
        在一个batch中，我们需要让他们的长度相等(max_len)
        '''
        batch.sort(key=lambda x : len(x[0]), reverse=True)     # x:[text_num, label_num]
        max_len = len(batch[0][0])
        text_num, label_num, mask = [], [], []
        for item in batch:
            pad_len = max_len - len(item[0])
            text_num.append(item[0] + [WORD_PAD_ID]*pad_len)
            label_num.append(item[1] + [LABEL_PAD_ID]*pad_len)
            mask.append([1]*len(item[0]) + [0]*pad_len)
        return torch.LongTensor(text_num), torch.LongTensor(label_num), torch.LongTensor(mask).bool()
            
    dataset = Dataset(data_path, data_set)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return dataloader 

class Dataset_Bert(data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def get_dict(self, dict_path):
        '''
        将存起来的labels文件转化成字典映射
        '''
        assert dict_path in [VOCAB_PATH, LABEL_PATH], 'wrong path!'
        df = pd.read_csv(dict_path, names=['char', 'id'])
        mapping = {}
        for i in range(1, len(df)):
            mapping[df['char'][i]] = df['id'][i]
        return mapping
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data['text'][idx]
        labels = self.data['BIO_anno'][idx]
        label2id = self.get_dict(LABEL_PATH)
        label_num = [int(label2id.get(v, LABEL_PAD_ID)) for v in labels.split(' ')]
        assert len(text) == len(label_num)
        return text, label_num

def get_bert_dataloader(data_set, tokenizer):
    def collate_fn(batch_samples):
        batch_samples.sort(key = lambda x:len(x[0]), reverse=True)
        max_len = len(batch_samples[0][0])      # 当前批次最长序列长度
        batch_text, batch_label = [], []
        for sample in batch_samples: 
            pad_len = max_len - len(sample[0])    # pad长度
            text = sample[0] + pad_len*BERT_PAD
            batch_text.append(text)
            
            label_ids = [LABEL_PAD_ID]+sample[1] + [LABEL_PAD_ID]*pad_len+[LABEL_PAD_ID]
            batch_label.append(label_ids)
            
        text_dic = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt', max_length=256)
        input_ids = text_dic['input_ids']
        assert len(batch_text) == len(batch_label), f'input: {len(batch_text)}, label: {len(batch_label)}'
        
        return input_ids, torch.tensor(batch_label, dtype=torch.long), torch.tensor(1)
            
    dataset = Dataset_Bert(data_set)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return dataloader 

