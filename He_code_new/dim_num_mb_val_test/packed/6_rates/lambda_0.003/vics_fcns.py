import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import sympy as sym
from sympy import Matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import os
import torch
import scipy
import matplotlib.pyplot as plt
import platform



def top_split_y(X, Y, percent):
    # Calculate the number of samples for test set
    a = 0
    if isinstance(X, torch.Tensor):
        X = X.numpy()
        a = 1
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    
    print('flag 1')
    print('Y.shape  =', Y.shape)
    print('X.shape  =', Y.shape)
    test_size = int(percent/100 * len(Y))
    print('test_size = ', test_size)
    # Sort Y and get indices of the top 15% values
    sorted_indices = np.argsort(Y[:,0])[::-1][:test_size].flatten()
    print('sorted indices = ', sorted_indices)
    # Create test set
    X_test = X[sorted_indices,:]
    Y_test = Y[sorted_indices]
    print('flag 2')
    print('Y_test.shape  = ', Y_test.shape)
    print('X_test.shape  =', X_test.shape)
    # Create train set
    X_train = np.delete(X, sorted_indices, axis=0)
    Y_train = np.delete(Y, sorted_indices, axis=0)
    print('flag 3')
    print('Y_train.shape  =', Y_train.shape)
    print('X_train.shape  =', X_train.shape)
    if a ==1:
        X_test = torch.tensor(X_test)
        Y_test = torch.tensor(Y_test)
        X_train = torch.tensor(X_train)
        Y_train = torch.tensor(Y_train)
    print('flag 4')
    print('Y_test.shape  =', Y_test.shape)
    print('X_test.shape  =', X_test.shape)
    print('Y_train.shape  =', Y_train.shape)
    print('X_train.shape  =', X_train.shape)
    return X_test, Y_test, X_train, Y_train

def top_split_x(X,Y,percent, index):
# Assuming X is your input array and Y is your output vector


    # Get indices of X[:, 3] sorted in descending order
    sorted_indices = np.argsort(X[:, index])[::-1]

    # Calculate the number of samples for test set
    test_size = int(percent/100 * len(Y))

    # Get the indices for the test set
    test_indices = sorted_indices[:test_size]

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Filter out the test set
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    # Filter out the training set
    X_train = np.delete(X, test_indices, axis=0)
    Y_train = np.delete(Y, test_indices, axis=0)
    return X_test, Y_test, X_train, Y_train