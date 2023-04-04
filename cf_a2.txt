import numpy as np
import pickle
import math
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

def calculate_mean_absolute_error(y_pred, y_true):
    # Select only the elements of y_pred and y_true that correspond to non-zero elements of y_true
    non_zero_indices = y_true.nonzero()
    y_pred_non_zero = y_pred[non_zero_indices]
    y_true_non_zero = y_true[non_zero_indices]
    
    # Calculate and return the mean absolute error between the predicted and true values
    mae = mean_absolute_error(y_pred_non_zero, y_true_non_zero)
    return mae

max_rating = 5
min_rating = 1
normal = max_rating - min_rating

def normalise(ratings):
    min_ratings = []
    max_ratings = []

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
                    ratings[i][j] = (ratings[i][j] - min_r)#/(max_r - min_r)

    return ratings

def predict(user_vecs, item_vecs):
    preds = np.zeros((user_vecs.shape[0], item_vecs.shape[0]))
    u = 0
    while u < user_vecs.shape[0]:
        i = 0
        while i < item_vecs.shape[0]:
            preds[u, i] = user_vecs[u, :].dot(item_vecs[i, :].T)
            i += 1
        u += 1
    return preds

def ALS(lat_vectors, fixed_vecs, ratings, mu, how):

    A = fixed_vecs.T.dot(fixed_vecs)
    muI = np.eye(A.shape[0]) * mu

    B = A+muI

    shp = lat_vectors.shape[0]
    
    if how == 'user':
        u = 0
        while u < shp:
            lat_vectors[u, :] = np.linalg.solve(B, ratings[u, :].dot(fixed_vecs))
            u += 1
    elif how == 'item':    
        i = 0
        while i < shp:
            lat_vectors[i, :] = np.linalg.solve(B, ratings[:, i].T.dot(fixed_vecs))
            i += 1
    
    return lat_vectors



def partial_train(iter_count, user_vecs, item_vecs, ratings, user_reg, item_reg):
    
    i = 0
    while i < iter_count:
        user_vecs = ALS(user_vecs, item_vecs, ratings, user_reg, 'user')
        item_vecs = ALS(item_vecs, user_vecs, ratings, item_reg, 'item')
        i += 1
    
    return user_vecs, item_vecs



def train(iter_count, ratings, factor_count, user_reg, item_reg):
    """ Train model for iter_count iterations from scratch."""
    # initialize latent vectors
    n_users, n_items = ratings.shape

    user_vecs = np.random.random((n_users, factor_count))
    item_vecs = np.random.random((n_items, factor_count))
    
    user_vecs, item_vecs = partial_train(iter_count, user_vecs, item_vecs, ratings, user_reg, item_reg)
    return user_vecs, item_vecs


def find_mae(iter_arr, rating_test, rating_train, factor_count, user_reg, item_reg):
    iter_arr.sort()
    train_mae =[]
    test_mae = []
    iter_diff = 0

    i = 0
    user_vecs, item_vecs = train(iter_arr[i] - iter_diff, rating_train, factor_count, user_reg, item_reg)

    while i < len(iter_arr):
        preds = predict(user_vecs, item_vecs)
        train_mae.append(calculate_mean_absolute_error(preds, rating_train))
        test_mae.append(calculate_mean_absolute_error(preds, rating_test))

        if i < len(iter_arr) - 1:
            iter_diff = iter_arr[i+1] - iter_arr[i]
            user_vecs, item_vecs = partial_train(iter_diff, user_vecs, item_vecs, rating_train, user_reg, item_reg)

        i += 1

    return (min(train_mae) / normal , min(test_mae) / normal)

iter_arr = [1,]
iter_arr.append(2)
iter_arr.append(5)
iter_arr.append(10)

reg_arr = [0.1, 1, 10]
LF = [5, 10, 20]

total_folds = 5
curr_fold = 1

train_nmae_arr = []
test_nmae_arr = []

while (curr_fold<=total_folds):
    print(f"Fold {curr_fold}")

    train_file = open(f"ml-100k/u{curr_fold}.base","r")
    test_file = open(f"ml-100k/u{curr_fold}.test","r")

    train_lines = train_file.readlines()
    test_lines = test_file.readlines()

    train_file.close()
    test_file.close()

    user_count = 943
    item_count = 1682

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

    table_train = normalise(table_train)
    table_test = normalise(table_test)

    np_train = np.array(table_train)
    np_test = np.array(table_test)


    min_train_nmae = 100
    min_test_nmae = 100

    i = 0
    while i < len(LF):
        j = 0
        while j < len(reg_arr):

            nmae = find_mae(iter_arr=iter_arr, rating_test=np_test, rating_train=np_train, factor_count=LF[i], user_reg=reg_arr[j], item_reg=reg_arr[j])
            if (nmae and nmae[0] < min_train_nmae):
                min_train_nmae = nmae[0]
            if (nmae and nmae[1] < min_test_nmae):
                min_test_nmae = nmae[1]

            j += 1
        i += 1




    #print(f"Train nmae: {min_train_nmae}")
    print(f"Test nmae: {min_test_nmae}")
    print()

    train_nmae_arr.append(min_train_nmae)
    test_nmae_arr.append(min_test_nmae)


    curr_fold+=1

avg_train = 0
avg_test = 0
for i in range(total_folds):
    avg_train += train_nmae_arr[i]
    avg_test += test_nmae_arr[i]

avg_train = avg_train/total_folds
avg_test = avg_test/total_folds

print(f"Average test nmae: {avg_test}")
