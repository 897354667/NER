# 基于CRF的多模型命名实体识别任务
本项目使用了多个模型，包括BiLSTM+CRF(基准)、BiLSTM+MultiheadAtttion+CRF、BERT+CRF、BERT+BiLSTM+CRF、Dynamic_BERT+CRF、Dynamic_BERT+BiLSTM+CRF。

模型文件放在`Models`文件夹中

模型运行的结果放在`Data`文件夹中

模型的预处理和坏例分析(Bad-case analysis)过程用notebook记录。
