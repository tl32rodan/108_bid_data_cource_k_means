#!/usr/bin/env python
# coding: utf-8

# In[1]:



import random
import math
import numpy as np
import pandas as pd
from sklearn import cluster, datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# # Data Preprocessing

# ## Abalone dataset

# In[2]:


#data_x : features (Dataframe)
#data_y : labels   (np.array)
'''
column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]
abalone = pd.read_csv("abalone.data",names=column_names)
data_x = abalone.drop(columns=["rings"])
#0,1,2 labeling sex
#data_x = data_x.replace(['M','F','I'],[0,1,2])

#one-hot encoding
dfDummies = pd.get_dummies(data_x['sex'], prefix = 'sex')
data_x = pd.DataFrame(preprocessing.scale(pd.concat([data_x.drop(columns=['sex']), dfDummies], axis=1)))

data_y = np.array(abalone["rings"])
data_y[data_y<11]    = 0
data_y[data_y>=11]   = 1
Simulate_time = 10
data_x
'''


# ## Iris Dataset

# In[3]:


#data_x : features (Dataframe)
#data_y : labels   (np.array)

iris = datasets.load_iris()
data_x = pd.DataFrame(iris.data)

data_y = iris.target

Simulate_time = 5


# In[4]:


class our_k_means:
    def __init__(self,data_x,data_y,m = 2,epsilon = 1.0e-6):
        self.data_x = pd.DataFrame(data_x)
        self.data_y = data_y
        self.label  = pd.DataFrame(data_y)
        self.acc    = 0
        
        #資料特性
        self.DCNT = len(self.data_x)               #資料個數
        self.DIM  = len(self.data_x.columns)       #資料維度
        self.K    = len(np.unique(data_y))         #叢聚個數
        #self.K    = np.amax(self.data_y)+1        #叢聚個數
        self.MAX_ITER = 100                        #最大迭代 
        self.MIN_PT = 0                            #最小變動點
        
        #k-means過程的參數
        self.m    = m                       #m              :hyper parameter,控制fuzzy程度的變數,通常為2
        self.epsilon = epsilon              #epsilon        :收斂的閾值
        self.data =[]                       #data[DCNT][DIM]:資料
        self.cent =[]                       #cent[K][DIM]   :各centroid的座標
        self.table=[]                       #table[DCNT][K] :各資料對各cluster的membership values matrix
        self.dis_k=[]                       #dis_k[K][DIM]  :各cluster的座標和
        self.cent_c=[]                      #cent_c[K]      :各cluster的擁有資料數和
        self.nearest=[]                     #nearest[DNST]  :各資料最可能屬於的cluster
        self.iterl = 0 
        self.obj_value = 0
        self.prev_obj_value = 0
        
        #計算acc時的參數
        self.origin_mass = []
        self.cent_name = []

        
    #run k-means    
    def run(self):
        
        #initialize tables
        self.kmeans_init()   #初始化centroid
                
        #first iteration 
        self.iterl = 0
        self.update_table()
        self.obj_value = self.cal_obj_func()
        self.prev_obj_value = self.obj_value*2
        
        #update centroid & data clustering
        while self.iterl<self.MAX_ITER and abs(self.prev_obj_value-self.obj_value)>=self.epsilon :
            self.prev_obj_value = self.obj_value
            self.iterl+=1
            self.update_cent()
            self.update_table()
            self.obj_value = self.cal_obj_func()
        
        #self.print_result()    
        
    #Calculate average accuracy    
    def calculate_acc(self,iterate_times):
        self.acc = 0
        i = 0
        while( i < iterate_times):
            self.run()
            self.calculate_origin_mass()
            self.cent_name = self.centroid_names()
            # Avoid the rare situations that some cluster are gone
            #if len(np.unique(self.cent_name)) != self.K:
            #    continue
                
            self.nearest_cluster()
            i += 1
            self.acc += accuracy_score(self.data_y,self.nearest)
            
            #self.print_result()
        
        if iterate_times is not 0:
            self.acc /= iterate_times
        
        print("Average accuracy for ",iterate_times," times : ",self.acc)
        return self.acc
#---------------------------------------------------------------------------------
#----------------Subfunctions of calculate_acc(iterate_times)---------------------
#---------------------------------------------------------------------------------
    def centroid_names(self):
        cent_name = np.zeros(self.K)
        
        for i in range(self.K):
            min_dist=float("inf")
            name = 0
            for j in range(self.K):
                dist = np.linalg.norm(self.cent[i] - self.origin_mass[j])
                if dist < min_dist:
                    min_dist = dist
                    name = j
            cent_name[i] = name
            
        return cent_name
    
    def calculate_origin_mass(self):
        self.origin_mass = np.zeros((self.K,self.DIM))
        
        counter = np.zeros(self.K)
        for i in range(self.K):
            counter[i] = len(self.data_y[self.data_y==i])
            
        
        for j in range(self.DIM):
            for i in range(self.DCNT):
                a = self.data_y[i]
                self.origin_mass[a][j] += self.data_x.iloc[i,j]
            for i in range(self.K):  
                if counter[i] is not 0:
                    self.origin_mass[i][j] /= counter[i]
    
    def nearest_cluster(self):
        self.nearest = np.zeros(self.DCNT)
        
        for i in range(len(self.data_x.index)):
            self.nearest[i] = self.cent_name[np.argmax(self.table[i])]
        
#---------------------------------------------------------------------------------
#------------------------------Subfunctions of run()------------------------------
#---------------------------------------------------------------------------------    
    def kmeans_init(self):
        
        self.data = self.data_x.values
        self.cent = np.zeros((self.K,self.DIM))
        self.table= np.zeros((self.DCNT,self.K))
        self.dis_k= np.zeros((self.K,self.DIM))
        self.cent_c=np.zeros(self.K)
        self.U    = np.zeros((self.DCNT,self.K))
                
        pick = []
        counter = 0
        while(counter<self.K):
            rnd = random.randint(0,self.DCNT-1)
            if(rnd not in pick):
                pick.append(rnd)
                counter=counter+1
                
        for i in range(self.K):
            for j in range(self.DIM):
                self.cent[i][j] = self.data[pick[i]][j] 
      

    def update_cent(self):
        
        for k in range(self.K):
            down = 0
            for i in range(self.DCNT):
                down += self.table[i][k]
                
            for i in range(self.DCNT):    
                for j in range(self.DIM):
                    self.cent[k][j] += self.data_x.iloc[i,j]*self.table[i][k]
                    
            for j in range(self.DIM):
                self.cent[k][j] /= down
                    
    def cal_w(self,i,j):
        w = 0
        dis = np.linalg.norm(self.data_x.iloc[i].values-self.cent[j])
        for c in range(self.K):
            dis_c = np.linalg.norm(self.data_x.iloc[i].values-self.cent[c])
            if dis_c != 0:
                w += math.pow((dis/dis_c),2/(self.m-1))
        
        if(w != 0):
            w = 1/w
            
        return w
            
    def update_table(self):
        for i in range(self.DCNT):
            for j in range(self.K):
                self.table[i][j] = self.cal_w(i,j)
                
    def cal_obj_func(self):
        obj_value = 0
        for i in range(self.DCNT):
            for j in range(self.K):
                obj_value += self.table[i][j]*math.pow(np.linalg.norm(self.data_x.iloc[i].values-self.cent[j]),2)
        return obj_value

    def print_cent(self):
        print("Centroids:")
        print(self.cent)

    def print_result(self):
        print("FCM:")
        print(self.table)
        print("Object function value = ",end='')
        print(self.obj_value)
        print("Previous Object function value = ",end='')
        print(self.prev_obj_value)
        print("iter = ",end='')
        print(self.iterl)     
    


# ## Here comes our FCM
# ### Let's run it!

# In[5]:


result = our_k_means(data_x,data_y)
result.calculate_acc(Simulate_time)


# In[6]:


result.print_cent()


# In[7]:


result.print_result()

