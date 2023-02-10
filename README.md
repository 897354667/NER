# 基于CRF的多模型命名实体识别任务
本项目使用了多个模型，包括BiLSTM+CRF(基准)、BiLSTM+MultiheadAtttion+CRF、BERT+CRF、BERT+BiLSTM+CRF、Dynamic_BERT+CRF、Dynamic_BERT+BiLSTM+CRF。

预训练模型路径`BERT`请自行更改为自己的

模型文件放在`Models`文件夹中

模型运行的结果放在`Data`文件夹中

# 项目食用指南
`main.py`: 主程序，需要在这里输入模型。

`k_folds.py`: K-折交叉验证。

`training.py`: 单轮训练和单论测试文件。

`ner_utils.py`: 用来加载数据、构建数据集、构建DataLoader。

`config.py`: 配置文件，用来修改超参数。

`bad_case_analysis.ipynb`: 坏例分析文件；数据增强。

`pre_process.ipynb`: 简要介绍项目数据来源，构建词表文件，标签文件。

# 各模型效果(测试集)
符号示意：
* L: BiLSTM、
* C: CRF、
* B: BERT、
* D: Dynamic merging、
* Au: Augmented data.

| | L+C | L+A+C | B+C | B+L+C | D+B+C | D+B+L+C | D+B+L+C(Au) |
| :---: | :--------: | :----: | :-----: | :-------: | :------: | :-----: | :-----: |
| F1 score | 0.7857 | 0.8148 | 0.8068 | 0.8256 | 0.8253 | 0.8262 | **0.8317** |
| Precision | 0.8414 | **0.8445** | 0.8080 | 0.8314 | 0.8323 | 0.8289 | 0.8377 |
| Recall | 0.7514 | 0.7971 | 0.8143 | 0.8298 | 0.8270 | 0.8314 | **0.8362** |
观察数据至少可以得出的结论：
1. Dynamic merging作用与LSTM起到的作用相似(B+C & D+B+C)，说明对Precision来说，中间态也很重要，而不能只取the last hidden state.
2. LSTM对于Recall的判别力不强，原因可能是LSTM参数过少，导致其无法拟合到足够的情形，因此出现某些TP找不出来的情况
