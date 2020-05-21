import numpy as np
import os.path
from data import cqr_opve_step, cqr_opve_traj
# import warnings filter
from warnings import simplefilter

if __name__ =="__main__":
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    
    train_path = 'train.npy'
    test_path_list = ['em_test1.npy','em_test2.npy','em_test3.npy','em_test4.npy']
    num_train_list = [5, 10, 20, 50, 75, 100, 200, 500]
    if not os.path.isfile('result/em_1.npy'):
        result = []
        for num_train in num_train_list:
            for test_path in test_path_list:
                result.append(cqr_opve_step(train_path, test_path, num_train))
        np.save('result/em_1.npy', result)
    if not os.path.isfile('result/em_2.npy'):
        result = []
        for num_train in num_train_list:
            for test_path in test_path_list:
                result.append(cqr_opve_traj(train_path, test_path, num_train))
        np.save('result/em_2.npy', result)