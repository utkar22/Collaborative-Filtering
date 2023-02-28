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

full_table = [[0 for x in range(user+1)] for y in range(item+1)]
avg_rating = [0 for x in range(item+1)] ## for storing the average of ratings


for line in user_1_base:
    splitted_arr = line.split()
    full_table[int(splitted_arr[1])][int(splitted_arr[0])] = int(splitted_arr[2])
	
for j in range(1,item+1):
    total_sum = sum(full_table[j])
    total_ratings = user - (full_table[j].count(0)-1)
    try:
        avg_rating[j] = float(total_sum)/total_ratings
    except:
        avg_rating[j] = 0


item_top_neighbours=[]


##### Creating Similarity Matrix for items ####
item_sim = [[0 for x in range(item+1)] for y in range(item+1)]
for i1 in range(1,item+1):
    curr = []
    for i2 in range(i1+1,item+1):
        item1_arr = []
        item2_arr = []
        for u in range(1,user+1):
            if full_table[i1][u] != 0 and full_table[i2][u] != 0:
                item1_arr.append(full_table[i1][u])
                item2_arr.append(full_table[i2][u])
        
        curr_sim = cosine_similarity(item1_arr,item2_arr)

        item_sim[i1][i2] = curr_sim
        item_sim[i2][i1] = curr_sim
            
        curr.append([i2,curr_sim])
        
    curr = sorted(curr,key=lambda l:l[1], reverse=True)
    item_top_neighbours.append(curr)
            

#pickle.dump(item_sim, open(f"u{curr_file_user}.p","wb"))
#item_sim = pickle.load(open(f"u{curr_file_user}.p","rb"))


K = 10

totalerror = 0
testsetlen = len(user_1_test)

for line in user_1_test:
    splitted_arr = line.split()
    user_id = int(splitted_arr[0])
    item_1_id = int(splitted_arr[1])
    curr_rating = int(splitted_arr[2])

    score = 0
    tot_sim = 0

    top_n = item_top_neighbours[item_1_id-1]

    j = 0
    t = 0
    while j<K and t<len(top_n):
        curr = top_n[t]
        
        item_2 = curr[0]
        similarity = curr[1]

        if full_table[item_2][user_id]!=0:
            score += (full_table[item_2][user_id]-avg_rating[item_2]) * similarity
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

