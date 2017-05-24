# JustAProject

## 文件描述

- `read_sample_data.py`: 读取训练数据
- `sample_data_desc.txt`: 采样后数据的描述文件
- `c3d_model.py`: c3d模型的网络
- `train.py`: 训练文件,使用.npy文件进行训练，推荐使用这个版本来训练
- `train-use-saver.py`: 训练文件，使用tf自带的saver函数加载、保存模型，saver函数保存模型会生成三个文件，所以推荐使用train.py进行训练
	- 训练时记得修改os.environ["CUDA_VISIBLE_DEVICES"]="2,3"来限制选择显卡
- `test.py`: 测试文件,使用.npy文件进行训练，推荐使用这个版本来训练
- `test-use-saver.py`: 测试文件，使用tf自带的saver函数加载，saver函数加载模型需要三个文件，所以推荐使用test.py进行预测
- `./model`: 运行后才会出现的目录，运行后会把训练过程的权值记录到该目录下
- `./visuak_logs`: 运行后才会出现的目录，运行后会把可视化数据记录到该目录下
- 'gen_desc.py': 采样生成desc.txt类文件的脚本
- 'sample_16_desc.txt':　按16帧，stride=4采样后数据的描述文件
- 'evalutation.py':　实现官网提供的准确率计算方法，输入为两个文件，一个为预测结果文件，一个为对应的groundTrue
- 'ChangePreModel.py':　文件中的changeModel函数，可以读取原先的sports1m_finetuning_ucf101.model中的参数，提取shape符合需要的参数，对于不符合的参数，用正态分布随机初设化
	- sports1m_finetuning_ucf101.model（原来C3D代码提供的模型） 文件在 /home/linzibo/JustAProject/models 目录下可以找到

## Run
```
运行命令：　/usr/bin/python train.py
不要使用：　python train.py
```
