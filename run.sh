#!/bin/bash

# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.5 --algorithm FedAvg
# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.5 --algorithm FedProx --mu 1
# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.5 --algorithm FedSR --mu 0.01 --lamda 0.01
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm SCAFFOLD
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm FedProx --mu 1 --aggregation=True 
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm FedSR --mu 0.01 --lamda 0.01 --aggregation=True --average weighted

# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.3 --algorithm FedAvg
# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.3 --algorithm FedProx --mu 1
# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.3 --algorithm FedSR --mu 0.01 --lamda 0.01
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm SCAFFOLD
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm FedProx --mu 1 --aggregation=True
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm FedSR --mu 0.01 --lamda 0.01 --aggregation=True --average weighted

# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.7 --algorithm FedAvg
# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.7 --algorithm FedProx --mu 1
# python main.py --data mnist --num_classes 10 -m polynomialRegression --beta 1 --dir 0.7 --algorithm FedSR --mu 0.01 --lamda 0.01
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm SCAFFOLD
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm FedProx --mu 1 --aggregation=True
# python main.py --data mnist --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm FedSR --mu 0.01 --lamda 0.01 --aggregation=True --average weighted

# FashionMNIST
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.3 --algorithm FedAvg
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.3 --algorithm FedProx --mu 1
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.3 --algorithm FedSR --mu 1 --lamda 0.1
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm SCAFFOLD
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm FedProx --mu 1 --aggregation=True
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.3 --algorithm FedSR --mu 1 --lamda 0.1 --aggregation=True --average weighted

# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.5 --algorithm FedAvg
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.5 --algorithm FedProx --mu 1
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.5 --algorithm FedSR --mu 1 --lamda 0.1
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm SCAFFOLD
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm FedProx --mu 1 --aggregation=True
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.5 --algorithm FedSR --mu 1 --lamda 0.1 --aggregation=True --average weighted

# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.7 --algorithm FedAvg
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.7 --algorithm FedProx --mu 1
# python main.py --data FashionMNIST --num_classes 10 -m polynomialRegression --beta 1 --dir 0.7 --algorithm FedSR --mu 1 --lamda 0.1
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm SCAFFOLD
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm FedProx --mu 1 --aggregation=True
# python main.py --data FashionMNIST --num_classes 10 -m mlr --beta 1 --dir 0.7 --algorithm FedSR --mu 1 --lamda 0.1 --aggregation=True --average weighted

#CIFAR10
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm FedAvg
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm FedProx --mu 0.01
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm FedSR --mu 0.1 --lamda 0.01
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm SCAFFOLD
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm FedProx --mu 0.01 --aggregation=True
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.3 --algorithm FedSR --mu 0.1 --lamda 0.01 --aggregation=True --average weighted

# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm FedAvg
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm FedProx --mu 0.01
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm FedSR --mu 0.1 --lamda 0.01
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm SCAFFOLD
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm FedProx --mu 0.01 --aggregation=True
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.5 --algorithm FedSR --mu 0.1 --lamda 0.01 --aggregation=True --average weighted

# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm FedAvg
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm FedProx --mu 0.01
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm FedSR --mu 0.1 --lamda 0.01
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm SCAFFOLD
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm FedProx --mu 0.01 --aggregation=True
# python main.py --data Cifar10 --num_classes 10 -m cnn --beta 1 --dir 0.7 --algorithm FedSR --mu 0.1 --lamda 0.01 --aggregation=True --average weighted
# 
# weather
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm FedAvg
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm FedProx --mu 0.01
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm FedSR --mu 0.1 --lamda 0.01
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm SCAFFOLD
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm FedAvg --aggregation=True --average weighted
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm FedProx --mu 0.01 --aggregation=True
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.3 --algorithm FedSR --mu 0.1 --lamda 0.01 --aggregation=True --average weighted

# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm FedAvg
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm FedProx --mu 0.01
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm FedSR --mu 0.1 --lamda 0.01
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm SCAFFOLD
python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm FedAvg --aggregation=True --average weighted
python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm FedProx --mu 0.01 --aggregation=True
python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.5 --algorithm FedSR --mu 0.1 --lamda 0.01 --aggregation=True --average weighted

# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm FedAvg
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm FedProx --mu 0.01
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm FedSR --mu 0.1 --lamda 0.01
# python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm SCAFFOLD
python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm FedAvg --aggregation=True --average weighted
python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm FedProx --mu 0.01 --aggregation=True
python main.py --data weather --num_classes 11 -m cnn --beta 1 --dir 0.7 --algorithm FedSR --mu 0.1 --lamda 0.01 --aggregation=True --average weighted