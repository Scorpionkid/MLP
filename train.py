import torch
import torch.nn as nn
import torch.optim as optim
import logging
import datetime
import numpy as np
from src.utils import *
from src.model import MLP
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, Subset, DataLoader
import os


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
set_seed(42)
# print(os.getcwd())
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "MLP"
dataName = 'Makin'
dataPath = "../Markin/Makin_processed_npy/"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 50
gap_num = 10    # the time slice
seq_size = 128    # the length of the sequence
input_size = 96
hidden_size = 256
out_size = 2   # the output dim
num_layers = 2

# learning rate
lrInit = 6e-4 if modelType == "MLP" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "MLP" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress


# loading data
print('loading data... ' + dataName)


# class Dataset(Dataset):
#     def __init__(self, seq_size, out_size, spike, target):
#         print("loading data...", end=' ')
#         self.seq_size = seq_size
#         self.out_size = out_size
#         self.x = spike
#         self.y = target

#     def __len__(self):
#         # 向上取整
#         return (len(self.x) + self.seq_size - 1) // self.seq_size

#     def __getitem__(self, idx):
#         # i = np.random.randint(0, len(self.x) - self.ctxLen)
#         start_idx = idx * self.seq_size
#         end_idx = start_idx + self.seq_size

#         # 处理最后一个可能不完整的序列
#         if end_idx > len(self.x):
#             # 对x和y进行填充以达到期望的序列长度
#             pad_size_x = end_idx - len(self.x)
#             x_padded = np.pad(self.x[start_idx:len(self.x), :], ((0, pad_size_x), (0, 0)), mode='constant',
#                               constant_values=0)
#             y_padded = np.pad(self.y[start_idx:len(self.y), :], ((0, pad_size_x), (0, 0)), mode='constant',
#                               constant_values=0)
#             x = torch.tensor(x_padded, dtype=torch.float32)
#             y = torch.tensor(y_padded, dtype=torch.float32)
#         else:
#             x = torch.tensor(self.x[start_idx:end_idx, :], dtype=torch.float32)
#             y = torch.tensor(self.y[start_idx:end_idx, :], dtype=torch.float32)
#         return x, y
class Dataset(Dataset):
    def __init__(self, data_path, ctx_len, vocab_size):
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        spike, target = AllDays_split(data_path)
        self.x, self.y = Reshape_ctxLen(spike, target, ctx_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


dataset = Dataset(dataPath, seq_size, out_size)

# 按时间连续性划分数据集
# trainSize = int(0.8 * len(dataset))
# train_Dataset, test_Dataset = split_dataset(ctxLen, out_dim, dataset, trainSize)
train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))
train_dataloader = DataLoader(train_Dataset, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_Dataset, batch_size=1024, shuffle=True)

# setting the model parameters
model = MLP(input_size, hidden_size, out_size)
rawModel = model.module if hasattr(model, "module") else model
rawModel = rawModel.float()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)


print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
      'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)


tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_Dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                      criterion=criterion, optimizer=optimizer)

trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')
