import torch
from tqdm import tqdm
from transformers import get_scheduler
from sklearn.metrics import f1_score, precision_score, recall_score

from config import *

def get_optimizer(model, dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=0,
        num_training_steps=EPOCH*len(dataloader)
    )
    return optimizer, lr_scheduler
    
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch):     # 一轮训练
    total_loss = 0.
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'epoch: {epoch}, loss: {0:>4f}')
    
    model.train()  
    
    for batch, (input_ids, label_ids, mask) in enumerate(dataloader):
        input_ids, label_ids, mask = input_ids.to(DEVICE), label_ids.to(DEVICE), mask.to(DEVICE)
        loss = model.loss_fn(input_ids, label_ids, mask)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_description(f'epoch: {epoch+1}, loss: {total_loss/(batch+1)}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode):
    assert mode in ['validat', 'test'], 'mode must be validation or test!'
    print(f'-------------------------{mode}ing----------------------------')
    model.eval()
    P, R, F1 = [], [], []
    with torch.no_grad():
        for batch, (input_ids, label_ids, mask) in enumerate(dataloader):
            input_ids, label_ids, mask = input_ids.to(DEVICE), label_ids.to(DEVICE), mask.to(DEVICE)
            # y_pred的例子 ：[[1, 2, 3], [2, 3], [1]], 之所以长度不同是因为有mask的存在。长度递减是因为我dataloader中按照长度给输入排序了。
            y_pred = model(input_ids, mask)
            p, r, f1 = eval(y_pred, label_ids)
            P.append(p)
            R.append(r)
            F1.append(f1)
    return P, R, F1

def eval(y_pred, label_ids):
    '''
    y_pred : model预测的结果
    label_num : 真值
    我们这里用strict F1 score， 定义见<https://www.datafountain.cn/competitions/529/datasets>
    '''
    # 先把y_pred填充成label_num的大小：缺的补0(mask 用 O填充的， O的id是0)
    max_len = len(label_ids[0])
    for row in y_pred:
        if len(row) < max_len:
            row += [LABEL_PAD_ID]*(max_len-len(row))
    y_pred = torch.tensor(y_pred)  
    assert y_pred.shape == label_ids.shape, 'wrong dimension!'
    y_pred = y_pred.flatten().cpu()
    label_ids = label_ids.flatten().cpu()
    R = recall_score(label_ids, y_pred, average='macro', zero_division=0)
    P = precision_score(label_ids, y_pred, average='macro', zero_division=0)
    F1 = f1_score(label_ids, y_pred, average='macro', zero_division=0)
    
    return P, R, F1
