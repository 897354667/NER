# 基于CRF的多模型命名实体识别任务
本项目使用了多个模型，包括BiLSTM+CRF(基准)、BiLSTM+MultiheadAtttion+CRF、BERT+CRF、BERT+BiLSTM+CRF、Dynamic_BERT+CRF、Dynamic_BERT+BiLSTM+CRF。

模型文件放在`Models`文件夹中

模型运行的结果放在`Data`文件夹中

# 项目食用指南
`main.py`: 主程序，需要在这里输入模型。

`k_folds.py`: K-折交叉验证。

`training.py`: 单轮训练和单论测试文件。

`ner_utils.py`: 用来加载数据、构建数据集、构建DataLoader。

`config.py`: 配置文件，用来修改超参数。

`bad_case_analysis.ipynb`: 坏例分析文件；数据增强。
