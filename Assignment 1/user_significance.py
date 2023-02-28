import numpy
import pickle
import math
from tqdm import tqdm

def average(x):
    return float(sum(x)) / len(x)

def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity between two vectors v1 and v2.
    The formula for cosine similarity is (v1 dot v2) / (||v1||*||v2||).
    """
    dot_product = sum([x*y for x,y in zip(v1, v2)])
    magnitude_v1 = math.sqrt(sum([x*x for x in v1]))
    magnitude_v2 = math.sqrt(sum([y*y for y in v2]))

    if (magnitude_v1 == 0 or magnitude_v2 == 0):
        return 0
    
    return dot_product / (magnitude_v1 * magnitude_v2)


curr_file_user = 1

user_1_base_file = open(f"ml-100k/u{curr_file_user}.base","r")
user_1_test_file = open(f"ml-100k/u{curr_file_user}.test","r")

user_1_base = user_1_base_file.readlines()
user_1_test = user_1_test_file.readlines()

user_1_base_file.close()
user_1_test_file.close()


user = 943
item = 1682

full_table = [[0 for x in range(item+1)] for y in range(user+1)]
avg_rating = [0 for x in range(user+1)] ## for storing the average of ratings


for line in user_1_base:
    splitted_arr = line.split()
    full_table[int(splitted_arr[0])][int(splitted_arr[1])] = int(splitted_arr[2])
	
for j in range(1,user+1):
    total_sum = sum(full_table[j])
    total_ratings = item - (full_table[j].count(0)-1)
    avg_rating[j] = float(total_sum)/total_ratings


user_top_neighbours=[]


##### Creating Similarity Matrix for users ####
user_sim = [[0 for x in range(user+1)] for y in range(user+1)]
for u1 in range(1,user+1):
    curr = []
    for u2 in range(u1+1,user+1):
        user1_arr = []
        user2_arr = []
        n=0
        for i in range(1,item+1):
            if full_table[u1][i] != 0 and full_table[u2][i] != 0:
                user1_arr.append(full_table[u1][i])
                user2_arr.append(full_table[u2][i])

                n+=1
        
        curr_sim = cosine_similarity(user1_arr,user2_arr)

        #For 50 and less co-rated items, weight is penalised as n/50. Else weight is 1.
        if (n<50):
            curr_sim = curr_sim * n/50

        user_sim[u1][u2] = curr_sim
        user_sim[u2][u1] = curr_sim
            
        curr.append([u2,curr_sim])
        
    curr = sorted(curr,key=lambda l:l[1], reverse=True)
    user_top_neighbours.append(curr)
            


K = 10

totalerror = 0
testsetlen = len(user_1_test)

for line in user_1_test:
    splitted_arr = line.split()
    user_1_id = int(splitted_arr[0])
    item_id = int(splitted_arr[1])
    curr_rating = int(splitted_arr[2])

    score = 0
    tot_sim = 0

    top_n = user_top_neighbours[user_1_id-1]

    j = 0
    t = 0
    while j<K and t<len(top_n):
        curr = top_n[t]
        
        user_2 = curr[0]
        similarity = curr[1]

        if full_table[user_2][item_id]!=0:
            score += (full_table[user_2][item_id]-avg_rating[user_2]) * similarity
            tot_sim += similarity

            j+=1

        t+=1

            
    
    if tot_sim == 0:
        testsetlen -= 1
        continue
    
    pred_rating = int(round(float(score)/tot_sim +avg_rating[user_1_id]))
    totalerror += abs(curr_rating - pred_rating)
    
mae = float(totalerror)/testsetlen
print (f"MAE = {mae}")

