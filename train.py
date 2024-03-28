import torch
import torch.nn as nn
import torch.optim as optim
import logging
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import set_seed, resample_data, spike_to_counts2, save_to_excel
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization
from src.model import MLP
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, Subset, DataLoader
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "MLP"
dataFile = "indy_20160627_01.mat"
dataPath = "../data/Makin/"
npy_folder_path = "../data/Makin_processed_npy"
excel_path = 'results/'
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 20
gap_num = 10    # the time slice
seq_size = 128    # the length of the sequence
input_size = 96
hidden_size = 256
out_size = 2   # the output dim
num_layers = 6

# learning rate
lrInit = 6e-4 if modelType == "MLP" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "MLP" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress
dimensions = ['test_r2', 'test_loss', 'train_r2', 'train_loss']

# loading data
print('loading data... ' + npy_folder_path)


class Dataset(Dataset):
    def __init__(self, seq_size, out_size, spike, target):
        print("loading data...", end=' ')
        self.seq_size = seq_size
        self.out_size = out_size
        self.x = spike
        self.y = target

    def __len__(self):
        # 向上取整
        return (len(self.x) + self.seq_size - 1) // self.seq_size

    def __getitem__(self, idx):
        start_idx = idx * self.seq_size
        end_idx = start_idx + self.seq_size

        # 处理最后一个可能不完整的序列
        if end_idx > len(self.x):
            # 对x和y进行填充以达到期望的序列长度
            pad_size_x = end_idx - len(self.x)
            x_padded = np.pad(self.x[start_idx:len(self.x), :], ((0, pad_size_x), (0, 0)), mode='constant',
                              constant_values=0)
            y_padded = np.pad(self.y[start_idx:len(self.y), :], ((0, pad_size_x), (0, 0)), mode='constant',
                              constant_values=0)
            x = torch.tensor(x_padded, dtype=torch.float32)
            y = torch.tensor(y_padded, dtype=torch.float32)
        else:
            x = torch.tensor(self.x[start_idx:end_idx, :], dtype=torch.float32)
            y = torch.tensor(self.y[start_idx:end_idx, :], dtype=torch.float32)
        return x, y

# spike, y, t = load_mat(dataPath+dataFile)
# # y = resample_data(y, 4, 1)
# # new_time = np.linspace(t[0, 0], t[0, -1], len(y))
# # spike, target = spike_to_counts2(spike, y, np.transpose(new_time), gap_num)
# spike, target = spike_to_counts1(spike, y, t[0])

# 获取spike和target子目录的绝对路径
spike_subdir = os.path.join(npy_folder_path, "spike")
target_subdir = os.path.join(npy_folder_path, "target")

# 获取spike和target目录下所有的npy文件名
spike_files = sorted([f for f in os.listdir(spike_subdir) if f.endswith('.npy')])
target_files = sorted([f for f in os.listdir(target_subdir) if f.endswith('.npy')])

# 确保两个目录下的文件一一对应
assert len(spike_files) == len(target_files)
results = []

# 遍历文件并对每一对spike和target文件进行处理
for spike_file, target_file in zip(spike_files, target_files):
    # 提取前缀名以确保对应文件正确
    prefix = spike_file.split('_processed_spike')[0]

    assert prefix in target_file, f"Mismatched prefix: {prefix} vs {target_file}"

    # 加载spike和target的npy文件
    spike = np.load(os.path.join(spike_subdir, spike_file))
    target = np.load(os.path.join(target_subdir, target_file))

    dataset = Dataset(seq_size, out_size, spike, target)

    # # 归一化
    # dataset.x, dataset.y = gaussian_nomalization(dataset.x, dataset.y)
    # # 平滑处理
    # dataset.x = gaussian_filter1d(dataset.x, 3, axis=0)
    # dataset.y = gaussian_filter1d(dataset.y, 3, axis=0)

    src_feature_dim = dataset.x.shape[1]
    trg_feature_dim = dataset.y.shape[1]


    # 按时间连续性划分数据集
    # trainSize = int(0.8 * len(dataset))
    # train_Dataset, test_Dataset = split_dataset(ctxLen, out_dim, dataset, trainSize)
    train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
    test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))
    train_dataloader = DataLoader(train_Dataset, batch_size=batchSize, shuffle=True)
    test_dataloader = DataLoader(test_Dataset, batch_size=len(test_Dataset), shuffle=True)

    num_hidden_layers = num_layers - 1
    step = (hidden_size - out_size) / num_hidden_layers
    # 构建 layerSizes 列表
    layerSizes = [input_size] + [max(int(hidden_size - step * i), out_size) for i in range(num_hidden_layers)] + [out_size]
    model = MLP(layerSizes)
    rawModel = model.module if hasattr(model, "module") else model
    rawModel = rawModel.float()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
          'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)


    tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                          learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                          warmupTokens=0, finalTokens=nEpoch*len(train_Dataset)*seq_size, numWorkers=0,
                          epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                          out_dim=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                          criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
    trainer.train()
    result = trainer.test()
    result['file_name'] = prefix
    results.append(result)
    print('done')
    # torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    #            + '.pth')
save_to_excel(results, excel_path + os.path.basename(npy_folder_path) + '-' + modelType + '-' + 'results.xlsx', modelType, nEpoch, dimensions)
