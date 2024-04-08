import torch
import torch.nn as nn
from torch.nn import functional as F
from torcheval.metrics.functional import r2_score
# MLP
class MLP(nn.Module):
    def __init__(self, layerSizes, device = "cuda"):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(prev_layer_size, next_layer_size)
                                     for prev_layer_size, next_layer_size in
                                     zip(layerSizes[:-1], layerSizes[1:])])
        self.device = device


    def forward(self, src):
        for layer in self.layers[:-1]:
            src = F.relu(layer(src))
            # 最后一层输出不使用ReLU激活函数
        src = self.layers[-1](src)
        return src



# 测试
# if __name__ == '__main__':
