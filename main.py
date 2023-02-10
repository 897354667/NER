from config import *
from Models.BiLSTM_Att_CRF import Model
from ner_utils import get_dataLoader
from training import get_optimizer, train_loop
from k_fold import k_folds

def main(k_folds_flag, model):
    if not k_folds_flag:
        model = Model().to(DEVICE)
        dataloader = get_dataLoader(TRAIN_PATH, None)
        optimizer, lr_scheduler = get_optimizer(model, dataloader)
        losses = []
        for epoch in range(EPOCH):
            total_loss = train_loop(dataloader, model, optimizer, lr_scheduler)
            losses.append(total_loss)
        print(f'final loss : {losses[-1]}')
    else:
        k_folds(model)


if __name__ == '__main__':
    model = 'D_BERT_BiLSTM_CRF'
    main(True, model)
