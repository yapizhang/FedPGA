from typing import Optional

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from scipy.stats import lognorm


random.seed(1)
np.random.seed(1)
num_clients = 10
dir_path = "mnist/"


# Allocate data to users
def generate_mnist(dir_path, num_clients, niid, balance, partition, dir_param):
    dir_path = os.path.join(dir_path, dir_param)
    dir_path = dir_path + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, dir_param=dir_param):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset_image = dataset_image.astype(float) / 255
    # for i in range(num_classes) :
    #     shape, loc, scale = lognorm.fit(dataset_image[dataset_label == i ] .flatten())
    #     dataset_image_lognormal = lognorm.rvs(shape, loc, scale, size=(1000,1,28,28))
    #     dataset_image_imbalanced = np.concatenate((dataset_image, dataset_image_lognormal))
    #     dataset_label_imbalanced = np.concatenate((dataset_label, np.zeros(dataset_image_lognormal.shape[0])))
    #     dataset_image = dataset_image_imbalanced
    #     dataset_label = dataset_label_imbalanced



    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2, dir_param=dir_param)


    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, dir_param=dir_param)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    dir_param: Optional[float] = sys.argv[4] if sys.argv[4] != "-" else None

    generate_mnist(dir_path, num_clients, niid, balance, partition, dir_param)
