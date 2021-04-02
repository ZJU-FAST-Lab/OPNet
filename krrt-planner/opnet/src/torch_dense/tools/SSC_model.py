import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

class mini_resblock(nn.Module):
    def __init__(self, inputnf, nf):
        nn.Module.__init__(self)
        self.block1_1 = nn.Sequential( nn.Conv3d(inputnf, nf, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(nf),
                                    nn.ReLU(inplace=True))
        self.block1_2 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.by_pass1 = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.block1_3 = nn.Sequential(nn.BatchNorm3d(nf),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        f0 = self.block1_1(x)
        x = self.block1_2(f0)
        f0 = self.by_pass1(f0)
        x = x + f0
        x = self.block1_3(x)
        return x

class mini_resblock_dil(nn.Module):
    def __init__(self, inputnf, nf):
        nn.Module.__init__(self)
        self.block1_1 = nn.Sequential( nn.Conv3d(inputnf, nf, kernel_size=3, stride=1, padding=2, dilation=2),
                                    nn.BatchNorm3d(nf),
                                    nn.ReLU(inplace=True))
        self.block1_2 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=2, dilation=2)

        self.by_pass1 = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.block1_3 = nn.Sequential(nn.BatchNorm3d(nf),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        f0 = self.block1_1(x)
        x = self.block1_2(f0)
        f0 = self.by_pass1(f0)
        x = x + f0
        x = self.block1_3(x)
        return x

class SSCModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.block1_1 = nn.Sequential( #nn.Conv3d(1, 16, kernel_size=7, stride=2, padding=3),
                                    nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(inplace=True))
        self.block1_2 = nn.Sequential(
                                    nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1))

        self.by_pass1 = nn.Conv3d(16, 32, kernel_size=1, stride=1, padding=0)

        self.block1_3 = nn.Sequential(nn.BatchNorm3d(32),
                                    nn.ReLU(inplace=True))

        # self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block2 = mini_resblock(32, 64)

        self.block3 = mini_resblock(64, 64)

        self.block4 = mini_resblock_dil(64, 64)

        self.block5 = mini_resblock_dil(64, 64)

        self.block6 = nn.Sequential(nn.Conv3d(192, 128, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(128, 16, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(16, 2, kernel_size=1, stride=1, padding=0),
                                    )

    def forward(self, x):

        f0 = self.block1_1(x)
        x = self.block1_2(f0)
        f0 = self.by_pass1(f0)
        x = x + f0
        x = self.block1_3(x)

        # x = self.pool1(x)

        x = self.block2(x)
        f1 = self.block3(x)

        f2 = self.block4(f1)
        f3 = self.block5(f2)

        x = torch.cat([f1, f2, f3], dim=1)

        x = self.block6(x)

        return x



if __name__ == '__main__':
    use_cuda = True

    model = SSCModel()
    # model = GenModel_fix(input_dim=(128,128,128), input_nf=1, nf_coarse=16, nf=16, num_hierarchy_levels=2, pass_occ=True, pass_feats=True, use_skip_sparse=1, use_skip_dense=1)
    # exit()
    # batch size 10 -- all batches are identical
    input = torch.rand(1,1,80,80,40)
    print('input', input.shape, torch.min(input).item(), torch.max(input).item())
    if use_cuda:
        model = model.cuda()
        input = input.cuda()
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i in range(100):
            output = model(input)
            # del output
        print("inference time: ", time.time() - start_time)
        print('output', output.shape, torch.min(output).item(), torch.max(output).item())

