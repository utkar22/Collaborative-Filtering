!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip

import numpy as np
import pickle
import math
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

user_count = 943
item_count = 1682

max_rating = 5
min_rating = 1

def calculate_nmae(y_pred, y_true, normal):
    '''Calculate the Normalised Mean Absolute Error'''
    non_zero_indices = y_true.nonzero()
    y_pred_non_zero = y_pred[non_zero_indices]
    y_true_non_zero = y_true[non_zero_indices]
    
    mae = mean_absolute_error(y_pred_non_zero, y_true_non_zero)
    nmae = mae/normal
    return nmae

def NNM(Y, R, lam):
    '''Function to run Nuclear Norm Minimisation'''
    X = np.zeros((user_count, item_count))
    for i in tqdm(range(100)):
        U, s, V = np.linalg.svd(R * Y, full_matrices=False)
        s = np.maximum(s - lam, 0)
        X = U.dot(np.diag(s)).dot(V)
    return X

#Open the training and testing files
train_file = open(f"ml-100k/u1.base","r") 
test_file = open(f"ml-100k/u1.test","r")

train_lines = train_file.readlines()
test_lines = test_file.readlines()

train_file.close()
test_file.close()

#Creating the user-item rating matrix
table_train = [[0 for x in range(item_count+1)] for y in range(user_count+1)]
table_test = [[0 for x in range(item_count+1)] for y in range(user_count+1)]

for line in train_lines:
    splitted = line.split()
    user_s = int(splitted[0])
    item_s = int(splitted[1])
    rating = int(splitted[2])/5
    table_train[user_s][item_s] = rating

for line in test_lines:
    splitted = line.split()
    user_s = int(splitted[0])
    item_s = int(splitted[1])
    rating = int(splitted[2])
    table_test[user_s][item_s] = rating

np_train = np.array(table_train)
np_test = np.array(table_test)


lam = 0.1
X = NNM(np_train, (np_train != 0), lam)

pred_test = np.nan_to_num(X) * (np_test == 0)
nmae = calculate_nmae(pred_test, np_test, max_rating - min_rating)
print("\nNMAE on testing set:", nmae)
