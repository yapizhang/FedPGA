from typing import Optional

import numpy as np
import os
import sys
import random
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from PIL import Image

from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 10
dir_path = "weather/"



# Allocate data to users
def generate_meteorology(dir_path, num_clients, niid, balance, partition, dir_param):
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
        
    # Get Cifar10 data

    img_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomVerticalFlip(0.3),
        transforms.Resize((32, 32)),
        transforms.CenterCrop((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = "weather_image"

    dataset = datasets.ImageFolder(root=data_dir, transform=img_transforms)
    datasetloader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset.imgs), shuffle=False)

    for _, data in enumerate(datasetloader, 0):
        dataset.data, dataset.targets = data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(dataset.data.cpu().detach().numpy())
    dataset_label.extend(dataset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

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

    generate_meteorology(dir_path, num_clients, niid, balance, partition, dir_param)
