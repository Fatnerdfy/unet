from unet import UNet
import utils
import numpy as np
from torch.autograd import Variable
import torch 
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from open3d import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# y_train = utils.load_joints('joint.txt')
# y_train = torch.tensor(y_train).float()

# print(y_train[0][:])

# x_train = utils.load_x_data()
# x_train = torch.tensor(x_train)

# print(x_train.size())
# print(x_train[0][0])

# x = x_train[0][0]

# cnt = 0
# for i in range(240):
#     for j in range(320):
#         if x[i][j] != torch.tensor(0.):
#             cnt += 1
#             print(x[i][j])

# print('{}'.format(cnt))

if __name__ == "__main__":
    depth_raw = utils.load_bin('depth.bin')
    depth = Image(depth_raw)
    plt.subplot(1,2,1)
    plt.imshow(depth)
    plt.show()