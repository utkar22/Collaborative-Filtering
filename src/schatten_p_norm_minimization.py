!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip

import numpy as np
import pickle
import math
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from numpy.linalg import svd

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

    nmae = mae

    #nmae = denormalise(nmae)

    return nmae

def denormalise(nmae):
    diff_arr = []
    for i in range(len(min_ratings_i)):
        diff_arr.append(max_ratings_i[i] - min_ratings_i[i])

    tot = sum(diff_arr)/len(diff_arr)

    return nmae*tot

min_ratings_i = []
max_ratings_i = []

def normalise(ratings):
    min_ratings = []
    max_ratings = []

    for j in range(len(ratings[0])):
        min_r = 5
        max_r = 5
        for i in range(len(ratings)):
            x = ratings[i][j]
            if x<min_r:
                min_r = x
            elif x>max_r:
                max_r = x

        min_ratings_i.append(min_r)
        max_ratings_i.append(max_r)

    for i in ratings:
        min_r = 5
        max_r = 1
        for j in i:
            if j!=0:
                if j>max_r:
                    max_r=j
                if j<min_r:
                    min_r=j
        min_ratings.append(min_r)
        max_ratings.append(max_r)

    for i in range(len(ratings)):
        min_r = min_ratings[i]
        max_r = max_ratings[i]
        for j in range(len(ratings[i])):
            if (ratings[i][j]!=0):
                if (min_r!=max_r):
                    ratings[i][j] = (ratings[i][j] - min_r)#/(max_r-min_r)
                    ratings[i][j] = (ratings[i][j])#*(max_ratings_i[j]-min_ratings_i[j])

    return ratings

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
    rating = (int(splitted[2]))
    table_train[user_s][item_s] = rating

for line in test_lines:
    splitted = line.split()
    user_s = int(splitted[0])
    item_s = int(splitted[1])
    rating = (int(splitted[2]))
    table_test[user_s][item_s] = rating

#table_train = normalise(table_train)
#table_test = normalise(table_test)

np_train = np.array(table_train)
np_test = np.array(table_test)

# Define function for Schatten-p norm minimization
def schatten_p_norm_minimization(matrix, p, lmbda, num_iters):
    for i in range(num_iters):
        U, S, Vt = svd(matrix, full_matrices=False)
        S = np.sign(S) * np.maximum(np.abs(S) - lmbda * p, 0)
        matrix = U @ np.diag(S) @ Vt
    return matrix

# Perform Schatten-p norm minimization and calculate NMAE for p = 0.1, p = 0.5, and p = 1
for p in [0.1, 0.5, 1]:
    new_mat = schatten_p_norm_minimization(np_train, p, 0.2, 5)
    nmae = calculate_nmae(np_test, new_mat, max_rating - min_rating)
    print(f'NMAE for p = {p}: {nmae}')
