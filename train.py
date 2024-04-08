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
dataPath = "../Makin/Makin_origin_npy/"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 128
nEpoch = 50
gap_num = 10    # the time slice
seq_size = 256    # the length of the sequence
input_size = 96
layerSizes = [256, 256, 256, 256, 256]
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

with open("train.csv", "a", encoding="utf-8") as file:
    file.write(dataPath + " batch size " + str(batchSize) + " epochs " + str(nEpoch) + " sequence len " + str(seq_size) + '\n')


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
train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))

train_dataloader = DataLoader(train_Dataset, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_Dataset, batch_size=len(test_Dataset), shuffle=False)

# setting the model parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(input_size, layerSizes, out_size, device)
rawModel = model.module if hasattr(model, "module") else model
rawModel = rawModel.float()

print("number of parameters: " + str(sum(p.numel() for p in model.parameters())))

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)


print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
      'seq_size', seq_size, 'layerSizes', layerSizes, 'num_layers', num_layers)


tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_Dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_size, seq_size=seq_size, layerSizes=layerSizes, num_layers=num_layers,
                      criterion=criterion, optimizer=optimizer)

trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
with open("train.csv", "a", encoding="utf-8") as file:
    file.write(f"trian loss, train r2 score\n")

trainer.train()

with open("train.csv", "a", encoding="utf-8") as file:
    file.write(f"test loss, test r2 score\n")

trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')
