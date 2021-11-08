#!/usr/bin/env python
# coding: utf-8

# In[26]:


# ---IMPORT PACKAGES---

import numpy as np     #package to work with large arrays and mathematical functions
import itertools       #package for a number of iterator building blocks
from time import time  #package to handle time-related tasks
import random          #package to generate pseudo-random numbers for various distributions
import matplotlib.pyplot as plt #package to creating static, animated, and interactive visualizations
from os.path import expanduser  # new

# In[2]:


# ---START THE CLOCK TO MEASURE THE EXECUTION SPEED---

start = time() 


# In[3]:


# ---MODEL PARAMETERS---
# --NK landscape parameters--

N = 3                  #Number of decisions
i = 1                #Generate number of NK landscapes to begin with
which_imatrix = 1      #Interaction matrix, Choose "1" for random, "2" for custom
K = 2                  #Number of interdependencies


# In[4]:


# ---GENERATE INTERACTION MATRIX---

def imatrix_rand():
    '''
    This function takes the number of N elements and K interdependencies to create a random interaction matrix.
    '''
    Int_matrix_rand = np.zeros((N, N))
    for aa1 in np.arange(N):
        Indexes_1 = list(range(N))
        Indexes_1.remove(aa1)              # remove self
        np.random.shuffle(Indexes_1)
        Indexes_1.append(aa1)
        Chosen_ones = Indexes_1[-(K+1):]   # this takes the last K+1 indexes
        for aa2 in Chosen_ones:
            Int_matrix_rand[aa1, aa2] = 1  # we turn on the interactions with K other variables
    return(Int_matrix_rand)


# In[5]:


# ---CUSTOM MATRIX---

if which_imatrix == 2:  #This value is custom choosen
    K = 1               #Set to the average value
    Int_matrix = np.array([
                 [1, 0, 1],
                 [1, 1, 0],
                 [1, 0, 1]
                 ])


# In[6]:


# ---FUNCTIONS TO GENERATE NK LANDSCAPE---

def calc_fit(NK_land_, inter_m, Current_position, Power_key_):
    '''
    Takes the landscape and a given combination and returns a vector of fitness values for the vector of the N decision variables.
    '''
    Fit_vector = np.zeros(N)
    for ad1 in np.arange(N):
        Fit_vector[ad1] = NK_land_[np.sum(Current_position * inter_m[ad1]
                                          * Power_key_), ad1]
    return(Fit_vector)


def comb_and_values(NK_land_, Power_key_, inter_m):
    '''
    Calculates values for all combinations on the landscape. The resulting array contains:
    - the first columns indexed from 0 to N-1 are for each of the combinations
    - columns indexed from N to 2*N-1 are for the fit value (vector) of those combinations
    - the column indexed 2N is for the total fit (average of the entire vector)
    - column indexed 2N+1 is a dummy, with 1 indicating a local peak
    - the last column is a dummy, with 1 indicating the global peak
    '''
    Comb_and_value = np.zeros((2**N, N*2+3))  # to capture the results
    c1 = 0  # starting counter for location
    for c2 in itertools.product(range(2), repeat=N):
        # this takes time so be carefull with landscapes of bigger size
        Combination1 = np.array(c2)  # taking each combination
        fit_1 = calc_fit(NK_land_, inter_m, Combination1, Power_key_)
        Comb_and_value[c1, :N] = Combination1  # combination and values
        Comb_and_value[c1, N:2*N] = fit_1
        Comb_and_value[c1, 2*N] = np.mean(fit_1)
        c1 = c1 + 1
    for c3 in np.arange(2**N):  # now let's see if it is a local peak
        loc_p = 1  # first, assume it is
        for c4 in np.arange(N):  # check the local neighbourhood
            new_comb = Comb_and_value[c3, :N].copy().astype(int)
            new_comb[c4] = abs(new_comb[c4] - 1)
            if ((Comb_and_value[c3, 2*N] <
                 Comb_and_value[np.sum(new_comb*Power_key_), 2*N])):
                loc_p = 0  # if smaller than the neighbour, then it is not peak
        Comb_and_value[c3, 2*N+1] = loc_p
    max_ind = np.argmax(Comb_and_value[:, 2*N])
    Comb_and_value[max_ind, 2*N+2] = 1
    return(Comb_and_value)


# In[7]:


# ---GENERATING THE NK LANDSCAPES---

Power_key = np.power(2, np.arange(N - 1, -1, -1))  #Used to find addresses on the landscape
Landscape_data = np.zeros((i, 2**N, N*2+3))        #We prepare an array to receive the data

for i_1 in np.arange(i):
    '''
    Now we create the landscapes
    '''
    if which_imatrix==1:
        Int_matrix = imatrix_rand().astype(int)
    elif which_imatrix==4:                         #Diagonal
        '''
        The code below serves to poke three holes in the diagonal IM so that K=2. It is a little bit cumbersome but does the job  :-)
        Note that it only works with N=6
        '''
        Int_matrix = Int_matrix4.copy()
        id_change = random.sample(range(15), 3)
        for index in id_change:
            if index == 0:
                Int_matrix[1,0] = 0
            elif index == 1:
                Int_matrix[2,0] = 0
            elif index == 2:
                Int_matrix[2,1] = 0
            elif index == 3:
                Int_matrix[3,0] = 0
            elif index == 4:
                Int_matrix[3,1] = 0
            elif index == 5:
                Int_matrix[3,2] = 0
            elif index == 6:
                Int_matrix[4,0] = 0
            elif index == 7:
                Int_matrix[4,1] = 0
            elif index == 8:
                Int_matrix[4,2] = 0
            elif index == 9:
                Int_matrix[4,3] = 0
            elif index == 10:
                Int_matrix[5,0] = 0
            elif index == 11:
                Int_matrix[5,1] = 0
            elif index == 12:
                Int_matrix[5,2] = 0
            elif index == 13:
                Int_matrix[5,3] = 0
            elif index == 14:
                Int_matrix[5,4] = 0
    
    NK_land = np.random.rand(2**N, N)              #This is a table of random U(0,1) numbers
                                                   #Now it is time to survey the topography of our NK landscape
    Landscape_data[i_1] = comb_and_values(NK_land, Power_key, Int_matrix)


# In[8]:


# ---CALCULATING SUMMARY STATISTICS---

number_of_peaks = np.zeros(i)
max_values = np.zeros(i)
min_values = np.zeros(i)

for i_2 in np.arange(i):
    number_of_peaks[i_2] = np.sum(Landscape_data[i_2, :, 2*N+1])
    max_values[i_2] = np.max(Landscape_data[i_2, :, 2*N])
    min_values[i_2] = np.min(Landscape_data[i_2, :, 2*N])

# --Let's print some summary statistics of our sample of NK landscapes--

print('Summary statistics for IMatrix: ' + str(which_imatrix) + ' K=' + str(K))
print('average number of peaks: ' + str(np.mean(number_of_peaks)))
print('maximum number of peaks: ' + str(np.max(number_of_peaks)))
print('minimum number of peaks: ' + str(np.min(number_of_peaks)))
print('average maximum value: ' + str(np.mean(max_values)))
print('average minimum value: ' + str(np.mean(min_values)))

# --Plot histogram of the number of local peaks in our sample--

plt.figure(1, facecolor='white', figsize=(4, 2), dpi=150)  # for screens with

# --Higher resolution change dpi to 150 or 200. For normal use 75--

plt.hist(number_of_peaks, bins=20, range=(1, 20), color='dodgerblue', edgecolor='black')     #Adjust if necessary
plt.title('Distribution of the number of peaks', size=12)
plt.xlabel('number of peaks', size=10)
plt.ylabel('frequency', size=10)


# In[9]:


# ---SAVING THE LANDSCAPES AS A BINARY FILE FOR FUTURE RETRIEVAL---

np.save('N_' + str (N) + '_Matrix_' + str(which_imatrix) + '_K_' + str(K) + '_i_' + str(i) + '.npy', Landscape_data)

elapsed_time = time() - start
print('time: ' + str("%.2f" % elapsed_time) + ' sec')


# In[10]:

print ("Data type is: ", type(Landscape_data))
print ("Array Dimensions are: ", Landscape_data.ndim)
print ("The shape of the array is: ", Landscape_data.shape)



# *** 1. LOAD THE NK LANDSCAPE FILE *****************************************

NK_landscape = np.load('N_3_Matrix_1_K_2_i_1.npy')

#%%
################################################################################################################################################################## 
print("####### Start of the new appended code ########")
# Notes
# 1. Code below works for any number of landscapes (i) num_landscapes, for any value of N, and any value of N1  

# importing additonal libraries
from itertools import combinations, product # library for computing combinatorials

# Defining utility functions
# This function returns all possible combinations ( and corresponding complimentary combinations ) for selecting num_r out of num_total locations 
# Each location is represented by an index. For num_total locations, index values vary from [0 to num_total-1].
# Example : think of combinatorial (num_total)C(num_r) 
#         : for num_total = 3, num_r = 2 -> location indices are 0,1,2 ;
#           (3)C(2) combinations are (0,1) , (1,2) , (0,2) and corresponding complimentary combinations are (2),(0),(1) 
def get_nCr_combinations(num_total, num_r) :
  index_combinations = combinations(range(0,num_total), num_r) # returns a Combination Object
  index_combinations = list(index_combinations) 
  index_combinations = [list(elem) for elem in index_combinations] # each of the elements of index_combinations are a tuple , converting each element to a list
  index_combinations_comp = []
  for index_combination in index_combinations :
    temp = [elem for elem in range(0,num_total) if (elem not in index_combination)]
    index_combinations_comp.append(temp)
  return index_combinations, index_combinations_comp  

# This returns all Nbit long binary numbers in order. Each binary number is represented as a list , for example [0,0,0]
def get_Nbit_bin_nums(Nbit) :
  nums = product([0,1],repeat=Nbit) # returns a Product object
  nums = list(nums)
  nums = [list(elem) for elem in nums]
  return nums

# This function computes the decimal value corresponding to binary digits in input numpy array digital_arr 
# Eg : if digit_arr == [0 , 0 , 1] , then function returns value as 1*2^0 + 0*2^1 + 0*2^2 
def compute_dec_val(digit_arr) :
  num = digit_arr.shape[0]
  val = 0
  for i in range(num):
    val = val + digit_arr[i]*(2**(num-1-i))
  return val

# This function computes a multi-dimensional index array corresponding to N, N1 values
def compute_multidim_index_array(N,N1) :
  index_combinations_N1, index_combinations_new = get_nCr_combinations(N,N1)
  num_nCn1_combinations = len(index_combinations_N1)
  binary_nums_N1 = get_Nbit_bin_nums(N1)
  binary_nums_new = get_Nbit_bin_nums(N-N1)
  num_N1_bit_digits = 2**N1
  num_NmN1_bit_digits = 2**(N-N1)
  mdim_index_array = np.zeros((num_N1_bit_digits,num_nCn1_combinations,num_NmN1_bit_digits),dtype="uint8") # Assuming N < 2^8=256 
  for i in range(num_N1_bit_digits) :
    for j in range(num_nCn1_combinations) :
      for k in range(num_NmN1_bit_digits) :
        temp_row = np.zeros( (N,) , dtype="uint8") # Assuming N < 2^8=256 
        temp_row[index_combinations_N1[j]] = binary_nums_N1[i]
        temp_row[index_combinations_new[j]] = binary_nums_new[k]
        mdim_index_array[i,j,k] = compute_dec_val(temp_row)
  return mdim_index_array   

# This functions returns the binary representation of the input number with nbit number of binary digits
def nbit_binary_rep(num,nbit) :
  str_nbit=""
  for i in range(nbit) :
    rem = num % 2
    str_nbit = str(rem) + str_nbit 
    num = num // 2
  return str_nbit 

# This function prints out the calculated average fitness values 
# Input 'arr' numpy contains fitness values, N1_val is the corresponding value of N1 
def print_fitness_values(arr, N1_val):
    rows, cols = arr.shape
    mdim_index_array = compute_multidim_index_array(N,N1_val)
    for i in range(rows):
        for j in range(cols) :
            indices_from_array = mdim_index_array[i,j]
            indices_str = [ nbit_binary_rep(elem,N) for elem in indices_from_array ]
            indices_str = " & ".join(indices_str)
            fitness_val = arr[i,j]
            print_str = "{} \t {} \t {} ".format(nbit_binary_rep(i,N1_val),indices_str,fitness_val)
            print(print_str)


# Calculating the results for NK_landscape
N = int( (NK_landscape.shape[2] - 3) / 2 )
num_landscapes = NK_landscape.shape[0]
print("Num of landscapes(i) : {} , N : {} ".format(num_landscapes,N))
N1_vals = list(range(1,N)) # N1 can take values from 1 to N-1
avg_fitness_vals_N = NK_landscape[:,:,2*N]
# Just for testing purposes and matching with values provided in the excel sheet for N=3 only. Do not Uncomment.
#avg_fitness_vals_N = np.array([[0.3635633686, 0.6738775208, 0.6083782294, 0.5940397713, 0.3959950414, 0.7282253333, 0.4898850854, 0.8165185328 ]])
print("Average fitness values used for N={} \n {}".format(N, avg_fitness_vals_N))
avg_fitness_vals_all_N1_allL = []
for N1_val in N1_vals : # 1 and 2
    mdim_index_array = compute_multidim_index_array(N,N1_val) 
    arr_shape = mdim_index_array.shape
    avg_fitness_vals_N1_allL = []
    for L_ind in range(num_landscapes) : # range 0 to 1
        avg_fitness_vals_N1_L = avg_fitness_vals_N[L_ind, mdim_index_array.flatten()].reshape(arr_shape)
        avg_fitness_vals_N1_L = np.mean(avg_fitness_vals_N1_L,axis=(2))
        avg_fitness_vals_N1_allL.append(avg_fitness_vals_N1_L)
    avg_fitness_vals_all_N1_allL.append(avg_fitness_vals_N1_allL)


# Printing calculated values
for i_landscape in range(num_landscapes):
  for N1_ind , N1_val in enumerate(N1_vals) :
    print("Landscape(i):{} , N1:{}".format(i_landscape, N1_val))
    print_fitness_values(avg_fitness_vals_all_N1_allL[N1_ind][i_landscape],N1_val)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


## no variable is changed and everything is original as per March's model original conceptualization. 
## change the variables as per NK model.

import pandas as pd
import collections
get_ipython().magic(u'matplotlib inline')


# In[2]:


# number of dimensions
# same as variable N in the NK model
m = 3

# number of people
n = 3

# times of iteration
t = 10


# In[3]:


# socialization rate
p1 = 0.1

# learning by the code
p2 = 0.9




# <b>Environmental (External) Reality</b>

# In[4]:


# # seed the random number generator. re-write this function accordingly.
# np.random.seed(0)

# In[5]:


# reality will change according to the global peak in the given NK landscape. re-write this function accordingly.
#reality = np.random.choice([-1, 1], size = m)
# finding global peak from lookup table
N = int( (NK_landscape.shape[2] - 3) / 2 )
num_landscapes = NK_landscape.shape[0]
gp_ind = np.argmax(NK_landscape[:,:,-1] == 1.0)
reality = NK_landscape[:,gp_ind,-3][0]

# In[6]:


reality


# **Beliefs held by individuals about reality and an organization - initial condition**

# In[7]:


# org will initialize as sequence of "?" instead of "0". re-write this function accordingly.
org = ['?','?','?']
org_score = np.average(NK_landscape[:,:,2*N])

# In[8]:


org_score


# In[9]:

def generate_miss_bit(N1_ind,ind):
    lst = [list(i) for i in itertools.product([0, 1], repeat=N1_ind)]
    miss_bit_str = []
    
    if ind < (2**N1_ind)*(N1_ind+1):
        miss_bit_int = lst[int(ind/(N1_ind+1))]  
        miss_bit_int.insert(ind%(N1_ind+1),2) # 2 instead of ?
        miss_bit_str = []
        for val in miss_bit_int:
            if val == 2:
                miss_bit_str.append("?")
            else:
                miss_bit_str.append(str(val))
    else:
        print("index out of range in 'generate_miss_bit'")
    return miss_bit_str    

def generate_index(N1_ind,miss_bit_str):
    miss_bit_int = []
    lst = [list(i) for i in itertools.product([0, 1], repeat=N1_ind)]
    rem = miss_bit_str.index('?')
    miss_bit_str.remove('?')
    for i in miss_bit_str:
        miss_bit_int.append(int(i))
    com = lst.index(miss_bit_int)
    return (com*(N1_ind+1)+rem)

# ind will initialize with 0, ?, 1. re-write this function accordingly.
#ind = np.random.choice([-1, 0, 1], size = (n, m))
N1_ind = N-1
i_landscape = 0
N1_ind_len = (2**N1_ind)*(N1_ind+1)
ind = []

#random indices from gavetti Model N1 = 2 
data  =list(range(N1_ind_len))
index = []
for i in range(n): 
    rand = random.choice(data)
    index.append(rand)    
for i in index:
    ind.append(generate_miss_bit(N1_ind,i))
print("randomized individuals")
print(ind)
ind_score = list(avg_fitness_vals_all_N1_allL[N1_ind-1][i_landscape].flatten()[index])
print(ind_score)
# In[10]:


ind_score

# In[12]:


# # knowledge level of the code
# # org_score will be calculated from the lookup table of NK landscape. re-write this function accordingly. 
# org_score = sum( org == reality )

# # knowledge level of individuals
# # ind_score will be calculated from the lookup table. re-write this function accordingly.
# ind_score = []
# for i in range(n):
#     ind_score.append( sum( ind[i] == reality ) )


# # In[13]:


print( "Knowledge Level of the Code: " + str(org_score) )
print( "Knowledge Level of the Individuals: " + str(sum(ind_score) / len(ind_score) ) )


# <b>Knowledge level (Individual Score) of each individual</b>

# In[14]:


for i in range(len(ind_score)): 
    print ("Knowledge level (ind_score) of Individual " + str(i) + ":", ind_score[i])


# **Socialization**

# In[15]:


for i in range(n):
    for k in range(m):
        if org[k] != '?':
            if ind[i][k] != org[k]:
                if np.random.random() < p1:
                    ind[i][k] = org[k]


#**Learning by the code**

# In[16]:


superior_group = []
for i in range(n):
    if ind_score[i] > org_score:
        superior_group.append(i)
print(superior_group)

# In[17]:


# the counter values of -1, 0, 1 will change with 0, ?, 1 respectively. re-write this function accordingly.

dominant_belief = []
ind = np.array(ind)

for k in range(m):
    counter = collections.Counter( ind[superior_group][:, k] )
    
    if counter['1'] >= counter['?'] and counter['1'] >= counter['0']:
        dominant_belief.append('1')
    elif counter['0'] >= counter['?'] and counter['0'] >= counter['1']:
        dominant_belief.append('0')
    else:
        dominant_belief.append('?')

# for k in range(m):
#     if np.random.random() < p2:
#         org[k] = dominant_belief[k]
org = dominant_belief

# In[35]:


print("updated dominant belief:") 
print(dominant_belief)



# In[19]:


reality


# In[20]:

print("updated org:") 
print(org)



# In[21]:


ind


# In[22]:

# ind_score = []
# for i in range(n):
#     ind_score.append( sum( ind[i] == reality ) )


# In[23]:


print( "Knowledge Level of the Code: " + str(org_score) )
print( "Knowledge Level of the Individuals: " + str(np.mean(ind_score)) )


# # Multiple Iterations

# **Defining the iteration function**

# # In[24]:


# # re-write this function accordingly.

# def iteration(m, n, t, p1, p2):
    
#     global ind_score_list
    
#     reality = np.random.choice([-1, 1], size = m)
#     org = np.random.choice([0], size = m)
#     ind = np.random.choice([-1, 0, 1], size = (n, m))
    
#     org_score_list = []
#     ind_score_list = []
    
#     for time in range(t):
        
#         org_score = sum( org == reality )
#         ind_score = []
#         for i in range(n):
#             ind_score.append( sum( ind[i] == reality ) )

#         # socialization
#         for i in range(n):
#             for k in range(m):
#                 if org[k] != 0:
#                     if ind[i][k] != org[k]:
#                         if np.random.random() < p1:
#                             ind[i][k] = org[k]

#         # learning by the code
#         superior_group = []
#         for i in range(n):
#             if ind_score[i] > org_score:
#                 superior_group.append(i)

#         dominant_belief = []
#         for k in range(m):
#             counter = collections.Counter( ind[superior_group][:, k] )

#             if counter[1] >= counter[0] and counter[1] >= counter[-1]:
#                 dominant_belief.append(1)
#             elif counter[-1] >= counter[0] and counter[-1] >= counter[1]:
#                 dominant_belief.append(-1)
#             else:
#                 dominant_belief.append(0)

#         for k in range(m):
#             if np.random.random() < p2:
#                 org[k] = dominant_belief[k]

#         # post-iteration score
#         org_score = sum( org == reality )

#         ind_score = []
#         for i in range(n):
#             ind_score.append( sum( ind[i] == reality ) )
            
#         org_score_list.append( org_score )
#         ind_score_list.append( np.mean(ind_score) )
        
#     print( "Knowledge Level of the Code: " + str(org_score) )
#     print( "Knowledge Level of the Individuals: " + str(np.mean(ind_score)) )


# # **Knowledge level comparison (p1 = 0.1 ~ 0.9)**

# # In[25]:


# plt.figure( figsize = (15, 6) )

# for soc_rate in np.arange(0.1, 1, 0.1):
#     iteration(30, 50, 80, soc_rate, 0.5)
#     plt.plot( ind_score_list, label = 'p1 = %s' % round(soc_rate, 1) )

# plt.xlabel('Iterations (t)')
# plt.ylabel('Knowledge Level of the Individuals')
# plt.legend()


# # In[26]:


# df = pd.DataFrame()

# # average of 20 iterations
# for i in range(20):
#     knowledge_list = []
#     for soc_rate in np.arange(0.1, 1, 0.1):
#         iteration(30, 50, 80, soc_rate, 0.5)
#         knowledge_list.append( ind_score_list[-1] )

#     df[i] = knowledge_list


# # In[27]:


# df['normalized'] = df.mean(axis = 1)
# df.set_index( np.arange(0.1, 1, 0.1), inplace = True )
# df


# # In[28]:


# plt.plot( df['normalized'] )
# plt.xlabel( 'Socialization Rate (p1)' )
# plt.ylabel( 'Average Equilibrium Knowledge' )


# # **Knowledge level comparison (p2 = 0.1, 0.5, 0.9)**

# # In[29]:


# dict = {}
# for learn_rate in [0.1, 0.5, 0.9]:
#     dict['p2_%s' % learn_rate] = pd.DataFrame()
    
# for key in dict.keys():
#     for soc_rate in np.arange(0.1, 1, 0.1):
#         iteration(30, 50, 200, soc_rate, float( key.split('_')[-1] ))
#         dict[key][soc_rate] = ind_score_list
#     dict[key].columns = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# # In[30]:


# dict['p2_0.1'].tail()


# # In[31]:


# dict['p2_0.5'].tail()


# # In[32]:


# dict['p2_0.9'].tail()


# # In[33]:


# plt.plot(dict['p2_0.1'].iloc[-1], label = 'p2 = 0.1')
# plt.plot(dict['p2_0.5'].iloc[-1], label = 'p2 = 0.5')
# plt.plot(dict['p2_0.9'].iloc[-1], label = 'p2 = 0.9')
# plt.xlabel( 'Socialization Rate (p1)' )
# plt.ylabel( 'Average Equilibrium Knowledge' )
# plt.legend()
# plt.axis([0, 1, 0, 30])
    
    



    
    


# In[ ]:


ind = [['1', '0', '?'], ['0', '?', '1'], ['?', '1', '0']]
ind = np.array(ind)
for k in range(3):
    counter = collections.Counter( ind[superior_group][:, k] )
    print(counter)


# In[12]:


miss_bit_str = ['?', '1', '1']
N1_ind = 2
miss_bit_int = []
lst = [list(i) for i in itertools.product([0, 1], repeat=N1_ind)]
rem = miss_bit_str.index('?')
miss_bit_str.remove('?')
for i in miss_bit_str:
    miss_bit_int.append(int(i))
com = lst.index(miss_bit_int)
print(com*(N1_ind+1)+rem)


# In[ ]:


if '?' in org:
    #updating org_score
    org_score = avg_fitness_vals_all_N1_allL[N1_ind-1][i_landscape].flatten()[generate_index(N1_ind,org)]
    #new ind
    N1_ind_len = (2**N1_ind)*(N1_ind+1)
    ind = []

    #random indices from gavetti Model N1 = 2 
    data  =list(range(N1_ind_len))
    index = []
    # getting range of acceptable indices from N1 = 2 model
    comb = []
    comb.append(org)
    org_copy = []

    org_copy[unknown_ind] = '0'
    comb.append(org_copy)
    org_copy = org
    org_copy[unknown_ind] = '1'

    print("possible individuals")
    print("comb")
    ind = random.choices(comb,k=3)
    print("randomized individuals")
    print(ind)
    for i in ind:
        index = generate_index(N1_ind,i)
        ind_score.append(list(avg_fitness_vals_all_N1_allL[N1_ind-1][i_landscape].flatten()[index]))
    print("updated ind_scores:" )
    print(ind_score)

else:
    print("find in landscape")

