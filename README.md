# JustAProject

## 文件描述

- `read_sample_data.py`: 读取训练数据
- `sample_data_desc.txt`: 采样后数据的描述文件
- `c3d_model.py`: c3d模型的网络
- `train.py`: 训练文件
- `./model`: 运行后才会出现的目录，运行后会把训练过程的权值记录到该目录下
- `./visuak_logs`: 运行后才会出现的目录，运行后会把可视化数据记录到该目录下
- 'gen_desc.py': 采样生成desc.txt类文件的脚本
- 'sample_16_desc.txt':　按16帧，stride=4采样后数据的描述文件
- 'ChangePreModel.py':　文件中的changeModel函数，可以读取原先的sports1m_finetuning_ucf101.model中的参数，提取shape符合需要的参数，对于不符合的参数，用正态分布随机初设化

## Run
```
运行命令：　/usr/bin/python train.py
不要使用：　python train.py
```
