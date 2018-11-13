from unet import UNet
import utils
import numpy as np
from torch.autograd import Variable
import torch 
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F 


num_epochs = 10
batch_size = 5
torch.manual_seed(42)
PI = 3.1415927
one_dive_sqrt_twoPI = torch.tensor(1.) / torch.sqrt(torch.tensor(2.*PI))


def get_normal(mu, sigma, y):
    tmp = y - mu
    with open('tt.txt', 'w+') as f:
        f.write(str(y) + '  ' + str(mu))
    result = torch.exp(-tmp*tmp / (2*sigma*sigma)) * one_dive_sqrt_twoPI / sigma
    return result

def myloss_func(parameters, y):
    result = torch.tensor(0.)
    mu = parameters[:, :63]
    sigma = parameters[:, 63:]
    for j in range(mu.size(0)):
        for i in range(63):
            result -= torch.log(get_normal(mu[j][i], sigma[j][i], y[j][i]))
    return result

def train():
    # cuda = torch.cuda.is_available()
    net = UNet(1, 1)
    # print(net)
    # if cuda:
    #     net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_func = myloss_func

    print("preparing traing data ...")
    # x_train = torch.randn(500, 1, 240, 320).float() #x data tensor
    x_train = utils.load_x_data()
    x_train = torch.tensor(x_train)
    # y_train = utils.load_joints('joint.txt')
    y_train = utils.joint_normalization()
    y_train = torch.tensor(y_train).float()

    torch_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    print("done...")

    for epoch in range(num_epochs):
        # net.train()
        for i, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            outputs = net(x)
            sigma = torch.exp(outputs[:, 63:])
            parameters = torch.cat([outputs[:, :63], sigma], 1)
            loss = loss_func(parameters, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch {} : {}/{} Loss : {}".format(epoch+1, i, len(train_loader), loss))
    return net

def get_trained_net():
    net = UNet(1, 1)
    net.load_state_dict(torch.load('net_params.pkl'))
    return net

def predict(x):
    net = get_trained_net()
    return net(x)

if __name__ == "__main__":
    net = train()
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_params.pkl')