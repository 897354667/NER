from training import train_loop, test_loop, get_optimizer
from config import *
from ner_utils import get_dataLoader, get_bert_dataloader
from Models.BiLSTM_Att_CRF import Model
# from Models.BiLSTM_CRF import Model
# from Models.BERT_CRF import Bert_Model
# from Models.D_BERT_CRF import Bert_Model
# from Models.BERT_BiLSTM_CRF import Bert_Model
from Models.D_BERT_BiLSTM_CRF import Bert_Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import torch
import time
import os

def get_train_val(i):       # 第i轮
    data = pd.read_csv(TRAIN_PATH)
    total_len = len(data)
    val_len = total_len//K
    train_data = pd.concat([data[:i*val_len], data[(i+1)*val_len:]], join='inner')
    val_data = data[i*val_len:(i+1)*val_len]
    return train_data.reset_index(), val_data.reset_index()

def visualize_loss(losses, img_save_path):
    figure, axs = plt.subplots(1, K, sharey=True)
    figure.suptitle(f'Loss in {K} folds')
    for i in range(K):
        xx = range(len(losses[i]))
        axs[i].plot(xx, losses[i])
        axs[i].set_xlabel('round '+str(i+1))
    plt.savefig(img_save_path + 'Losses.jpg')

def visualize_p_r_f1(data, mode, img_save_path):
    figure, axs = plt.subplots(1, K, sharey=True)
    figure.suptitle(f'{mode} in {K} folds')
    for i in range(K):
        axs[i].boxplot(data[i])
        axs[i].set_xlabel('round '+str(i+1))
    plt.savefig(img_save_path+mode+'.jpg')
    

def k_folds(model_name):
    save_path = '/project1/liqingshan/NLP/NER/Data/'+ model_name +'/'
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    best_f1 = 0.
    losses, Ps, Rs, F1s = [], [], [], []
    averaged = [[], [], []]     # p, r, f1
    for i in range(K):
        print(f'------------the {i+1}th round begin, {K}rounds in total--------------------')
        train_data, val_data = get_train_val(i)
        # 根据模型决定dataloader加载方式
        if 'BERT' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
            train_dataloader = get_bert_dataloader(train_data, tokenizer)
            val_dataloader = get_bert_dataloader(val_data, tokenizer)
            model = Bert_Model().to(DEVICE)
        else:
            train_dataloader = get_dataLoader(None, train_data)
            val_dataloader = get_dataLoader(None, val_data)
            model = Model().to(DEVICE)
                
        if i == 0:
            print(f'-------------------------Using { model_name } model----------------------------')
        optimizer, lr_scheduler = get_optimizer(model, train_dataloader)
        
        # -------------------------------training----------------------------
        loss = []
        for epoch in range(EPOCH):
            total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch)
            loss.append(total_loss)
            
        losses.append(loss)
        
        # ---------------------------validation-----------------------------
        P, R, F1 = test_loop(val_dataloader, model, mode='validat')
        averaged_p = np.mean(P)
        averaged_r = np.mean(R)
        averaged_f1 = np.mean(F1)
        
        if averaged_f1 > best_f1:
            best_f1 = averaged_f1
            print('saving weights...\n')
            
            torch.save(model.state_dict(), save_path+'_weights.bin')
        print(f'validation averaged: precision: {averaged_p},  recall: {averaged_r},  F1 score: {averaged_f1}')
        
        F1s.append(F1)
        Ps.append(P)
        Rs.append(R)
        averaged[0].append(averaged_p)
        averaged[1].append(averaged_r)
        averaged[2].append(averaged_f1)
        
    
    # -------------------------------visualising-------------------------------
    visualize_loss(losses, save_path)
    visualize_p_r_f1(F1s, 'F1 score', save_path)
    visualize_p_r_f1(Ps, 'Precision', save_path)
    visualize_p_r_f1(Rs, 'Recall', save_path)
    print(f'validation final averaged: precision: {np.mean(averaged[0])}, recall: {np.mean(averaged[1])}, f1 score: {np.mean(averaged[2])}')

    # -------------------------------------testing----------------------------------
    test_set = pd.read_csv(TEST_PATH)
    if 'BERT' in model_name:
        dataloader = get_bert_dataloader(test_set, tokenizer)
    else:
        dataloader = get_dataLoader(None, test_set)
    model.load_state_dict(torch.load(save_path+'_weights.bin'))
    model = model.to(DEVICE)
    P, R, F1 = test_loop(dataloader, model, mode='test')    
    print(f'test averaged: precision: {np.mean(P)},  recall: {np.mean(R)},  F1 score: {np.mean(F1)}')    
        
    print('Done')
    end_time = time.time()
    print(f'total time : {end_time - start_time}')
