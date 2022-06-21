import time
import torch
import os
import random
from torch.autograd import Variable
from alexnet_bulid_network import alexnet
from cnn_bulid_network import cnn
from VGG_bulid_network import vgg
from google_bulid_network import GoogLeNet

import numpy as np
import torch.nn as nn
from torch import optim
from data_preprocess import load_data_cnn,load_data_alex,load_data_vgg,load_data_google

def setup_seed(seed):#随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)
def train():
    start =time.time()
    train_loader, test_loader = load_data_alex()  #alexnet 去掉前面的注释
    #train_loader, test_loader = load_data_cnn()  #cnn 去掉前面的注释
    #train_loader, test_loader = load_data_vgg()  #vgg 去掉前面的注释
    #train_loader, test_loader = load_data_google()  #google 去掉前面的注释

    print('train...')
    epoch_num = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet().to(device)  #alexnet 去掉前面的注释
    #model = cnn().to(device)     #cnn 去掉前面的注释
    #model = vgg(num_classes=2, init_weights=True)  #vgg 去掉前面的注释
    #model.to(device)                               #vgg 去掉前面的注释
    #model = GoogLeNet(num_classes=2, aux_logits=False, init_weights=True)  #google 去掉前面的注释
    #model.to(device)  #google 去掉前面的注释

    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    end = time.time()
    print('Training time: %s Seconds' % (end - start))
    torch.save(model.state_dict(), 'F:/PythonSave/dog_cat_classification/weights/weight_dog_cat_alex.pt')
if __name__ == '__main__':
    train()