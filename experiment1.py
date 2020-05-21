import numpy as np
import os.path

from data import cqr_opve_step, cqr_opve_traj
# import warnings filter
from warnings import simplefilter


if __name__ == "__main__":
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    
    train_path = 'train.npy'
    test_path_list = ['test1.npy','test2.npy','test3.npy','test4.npy']
    num_train_list = [5, 10, 20, 50, 75, 100]
    if not os.path.isfile('result/1.npy'):
        result = []
        for num_train in num_train_list:
            for test_path in test_path_list:
                result.append(cqr_opve_step(train_path, test_path, num_train))
        np.save('result/1.npy', result)
        
    if not os.path.isfile('result/2.npy'):    
        result2 =[]
        for num_train in num_train_list:
            for test_path in test_path_list:
                result2.append(cqr_opve_traj(train_path, test_path, num_train))
        np.save('result/2.npy', result2)
    
    test_path_list = ['test1.npy','test2.npy','test3.npy','test4.npy']
    num_train_list = [200, 500]
    
    if not os.path.isfile('result/3.npy'):
        result = []
        for num_train in num_train_list:
            for test_path in test_path_list:
                result.append(cqr_opve_step(train_path, test_path, num_train))
        np.save('result/3.npy', result)
    
    if not os.path.isfile('result/4.npy'):    
        result4 =[]
        for num_train in num_train_list:
            for test_path in test_path_list:
                result4.append(cqr_opve_traj(train_path, test_path, num_train))
        np.save('result/4.npy', result4)
        
    """mixed policy train set test"""
    train_path_list = ['train_mp_0.npy', 'train_mp_2.npy', 'train_mp_4.npy', 'train_mp_6.npy', 'train_mp_8.npy', 'train_mp_10.npy']
    if not os.path.isfile('result/mp_result1.npy'):   
        mp_result =[]
        for train_path in train_path_list:
            for test_path in test_path_list:    
                mp_result.append(cqr_opve_step(train_path, test_path, num_train = 100))
        np.save('result/mp_result1.npy', mp_result)
        
        mp_result =[]
        for train_path in train_path_list:
            for test_path in test_path_list:    
                mp_result.append(cqr_opve_traj(train_path, test_path, num_train = 1000))
        np.save('result/mp_result2.npy', mp_result)
    
    """train set generated from multiple behviour policies"""
    train_path = 'train_ba.npy'
    test_path_list = ['test1.npy','test2.npy','test3.npy','test4.npy']
    num_train_list = [500]
    if not os.path.isfile('result/ba_result1.npy'):
        ba_result = []
        for num_train in num_train_list:
            for test_path in test_path_list:
                ba_result.append(cqr_opve_step(train_path, test_path, num_train))
        np.save('result/ba_result1.npy', ba_result)
        
        ba_result = []
        for num_train in num_train_list:
            for test_path in test_path_list:
                ba_result.append(cqr_opve_traj(train_path, test_path, num_train))
        np.save('result/ba_result2.npy', ba_result)
        
    """train set with different length of trajectories"""
    longer_train = 'long_train.npy'
    longer_test_list = ['long_test1.npy', 'long_test2.npy', 'long_test3.npy', 'long_test4.npy']
    max_t_list = [50, 100, 200, 500, 1000]
    if not os.path.isfile('result/long_result1.npy'):
        long_result =[]
        for t in max_t_list:
            for test_path in longer_test_list:
                long_result.append(cqr_opve_step(longer_train, test_path,num_train =100, max_t = t))
        np.save('result/long_result1.npy', long_result)        
        long_result =[]
        for t in max_t_list:
            for test_path in longer_test_list:
                long_result.append(cqr_opve_traj(longer_train, test_path,num_train =1000, max_t = t))
        np.save('result/long_result2.npy', long_result)
        