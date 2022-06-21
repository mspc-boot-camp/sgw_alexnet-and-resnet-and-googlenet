from data_preprocess import load_test_data_alex,load_test_data_cnn,load_test_data_vgg,load_test_data_google

import torch
from alexnet_bulid_network import alexnet
from cnn_bulid_network import cnn
from VGG_bulid_network import vgg
from google_bulid_network import GoogLeNet
import time
import numpy

def test():
    test_loader = load_test_data_alex()   #alexnet 去掉前面的注释
    #test_loader = load_test_data_cnn()   #cnn 去掉前面的注释
    #test_loader = load_test_data_vgg()   #vgg 去掉前面的注释
    #test_loader = load_test_data_google()  #google 去掉前面的注释

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet().to(device)    #alexnet 去掉前面的注释
    #model = cnn().to(device)       #cnn 去掉前面的注释
    #model = vgg(num_classes=2, init_weights=True)  #vgg 去掉前面的注释
    #model.to(device)                               #vgg 去掉前面的注释
    #model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True)  #google 去掉前面的注释
    #model.to(device)  #google 去掉前面的注释

    model.load_state_dict(torch.load("F:/PythonSave/dog_cat_classification/weights/weight_dog_cat_alex.pt"), False)
    model.eval()
    total = 0
    current = 0
    out_test = []
    for data in test_loader:
        start = time.time()
        images, labels = data

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
        end = time.time()
        time_do = end-start
        out_test.append([labels,time_do])
    numpy.savetxt('F:/PythonSave/dog_cat_classification/output/Result_alex_10.csv', out_test, delimiter=',')
    print('Accuracy:%d%%' % (100 * current / total))
if __name__ == '__main__':
    test()