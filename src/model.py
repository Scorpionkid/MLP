import torch
import torch.nn as nn
from torch.nn import functional as F
# from torcheval.metrics.functional import r2_score


# MLP
class MLP(nn.Module):
    def __init__(self, input_size, layerSizes, out_size, device):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.device = device
        self.MLP = nn.Sequential()

        for i, (inSize, outSize) in enumerate(zip([input_size] + layerSizes, layerSizes + [out_size])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(inSize, outSize)
            )

            if i < len(layerSizes):
                self.MLP.add_module(
                    name="A{:d}".format(i), module=nn.ReLU()
                )
            # else:
            #     self.MLP.add_module(
            #         name="A{:d}".format(i), module=nn.Softmax()
            #     )

    def forward(self, input):
        output = self.MLP(input)

        return output



# 测试
# if __name__ == '__main__':
