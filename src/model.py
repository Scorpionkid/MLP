import torch
import torch.nn as nn
from torch.nn import functional as F
# from torcheval.metrics.functional import r2_score

# MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, device = "cuda"):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        N, T, C = src.shape
        src = self.fc1(src)
        src = F.relu(src)
        src = self.fc2(src)
        src = F.relu(src)
        src = self.fc3(src)
        src = F.relu(src)
        src = self.fc4(src)
        src = F.relu(src)
        src = self.fc5(src)
        src = F.relu(src)
        src = self.fc6(src)
        src = F.relu(src)
        src = self.fc7(src)
        return src



# 测试
# if __name__ == '__main__':
