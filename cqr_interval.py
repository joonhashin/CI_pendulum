# import torch
# import random
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.nc import QuantileRegErrFunc

def CQR_rf_interval(x_train, y_train, x_test, y_true, alpha):
    quantiles = [alpha/2, 1-alpha/2]
    
    n_train = x_train.shape[0]
    # in_shape = x_train.shape[1]
    print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" % 
          (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

    # divide the data into proper training set and calibration set
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

    # # zero mean and unit variance scaling 
    # scalerX = StandardScaler()
    # scalerX = scalerX.fit(x_train[idx_train])
    
    # # scale
    # x_train = scalerX.transform(x_train)
    # x_test = scalerX.transform(x_test)
    
    # # scale the labels by dividing each by the mean absolute response
    # mean_y_train = np.mean(np.abs(y_train[idx_train]))
    # y_train = np.squeeze(y_train)/mean_y_train
    # y_true = np.squeeze(y_true)/mean_y_train

    ### Quantile random forests parameters
    # the number of trees in the forest
    n_estimators = 1000
    
    # the minimum number of samples required to be at a leaf node
    # (default skgarden's parameter)
    min_samples_leaf = 1
    
    # the number of features to consider when looking for the best split
    # (default skgarden's parameter)
    max_features = x_train.shape[1]
    
    # target quantile levels
    quantiles_forest = [quantiles[0]*100, quantiles[1]*100]
    
    # use cross-validation to tune the quantile levels?
    cv_qforest = True
    
    # when tuning the two QRF quantile levels one may
    # ask for a prediction band with smaller average coverage
    # to avoid too conservative estimation of the prediction band
    # This would be equal to coverage_factor*(quantiles[1] - quantiles[0])
    coverage_factor = 0.85
    
    # ratio of held-out data, used in cross-validation
    cv_test_ratio = 0.05
    
    # seed for splitting the data in cross-validation.
    # Also used as the seed in quantile random forests function
    cv_random_state = 1
    
    # determines the lowest and highest quantile level parameters.
    # This is used when tuning the quanitle levels by cross-validation.
    # The smallest value is equal to quantiles[0] - range_vals.
    # Similarly, the largest value is equal to quantiles[1] + range_vals.
    cv_range_vals = 30
    
    # sweep over a grid of length num_vals when tuning QRF's quantile parameters                   
    cv_num_vals = 10
    
    # define the QRF's parameters 
    params_qforest = dict()
    params_qforest["n_estimators"] = n_estimators
    params_qforest["min_samples_leaf"] = min_samples_leaf
    params_qforest["max_features"] = max_features
    params_qforest["CV"] = cv_qforest
    params_qforest["coverage_factor"] = coverage_factor
    params_qforest["test_ratio"] = cv_test_ratio
    params_qforest["random_state"] = cv_random_state
    params_qforest["range_vals"] = cv_range_vals
    params_qforest["num_vals"] = cv_num_vals
    

    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=quantiles_forest,
                                                               params=params_qforest)
    print("CQR!")
    # define the CQR object
    nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())
    
    # run CQR procedure
    y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)
    
    coverage_cqr, length_cqr = compute_coverage_len(y_true, y_lower, y_upper)
    print("coverage: ", coverage_cqr)
    print("average length: ", length_cqr)
    return y_lower, y_upper

    
def compute_coverage_len(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals
    Parameters
    ----------
    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    Returns
    -------
    coverage : float, average coverage
    avg_length : float, average length
    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length
