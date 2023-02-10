TRAIN_PATH = '/project1/liqingshan/NLP/NER/Data/train.csv'
TEST_PATH = '/project1/liqingshan/NLP/NER/Data/test.csv'

VOCAB_PATH = '/project1/liqingshan/NLP/NER/Data/vocab.txt'
LABEL_PATH = '/project1/liqingshan/NLP/NER/Data/label.txt'

BERT_PATH = '/project1/liqingshan/NLP/Models/bert-base-chinese'
# ERNIE_PATH = '/project1/liqingshan/NLP/Models/ERNIE'
# RoBERTa_PATH = '/project1/liqingshan/NLP/Models/chinese-roberta-wwm-ext'

PAD = '<PAD>'
WORD_PAD_ID = 0
LABEL_PAD = 'O'
LABEL_PAD_ID = 0

BERT_PAD = '[PAD]'
BERT_PAD_ID = 1
UNK = '<UNK>'
UNK_ID = 1

DEVICE = 'cuda:1'

VOCAB_SIZE = 1970
BATCH_SIZE = 32

# MAX_LEN = 60
# BASE_LEN = 50

EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LABEL_SIZE = 9
LR = 5e-5
EPOCH = 10
DROPOUT = 0.2

WEIGHT_DECAY = 1e-2
K = 5      # k-folds training
