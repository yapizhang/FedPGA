import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, dir=None, beta=None):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, dir, beta)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    # print("std for best accurancy:", np.std(max_accurancy))
    # print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, dir=None, beta=None):
    test_acc = []
    algorithms_list = [algorithm] * times

    file_name = dataset + "_dir" + str(dir) + "_" + algorithm + "_beta" + str(beta)
    test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc