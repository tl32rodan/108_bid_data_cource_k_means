#!/usr/bin/env python
# coding: utf-8

# In[1]:



import random
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


#data_x : features (np.array)
#data_y : labels   (np.array)

iris = datasets.load_iris()
data_x = iris.data
data_y = iris.target

Simulate_time = 100


# In[4]:


class our_k_means:
    def __init__(self,data_x,data_y):
        self.data_x = pd.DataFrame(data_x)
        self.data_y = data_y
        self.label  = pd.DataFrame(data_y)
        self.acc    = 0
        
        #資料特性
        self.DCNT = len(self.data_x)               #資料個數
        self.DIM  = len(self.data_x.columns)       #資料維度
        self.K    = len(np.unique(self.data_y))    #叢聚個數
        #self.K    = np.amax(self.data_y)+1        #叢聚個數
        self.MAX_ITER = 30                         #最大迭代 
        self.MIN_PT = 0                            #最小變動點
        
        #k-means過程的參數
        self.data =[]                       #data[DCNT][DIM]:資料
        self.cent =[]                       #cent[K][DIM]   :各centroid的座標
        self.table=[]                       #table[DCNT]    :各資料的所屬cluster
        self.dis_k=[]                       #dis_k[K][DIM]  :各cluster的座標和
        self.cent_c=[]                      #cent_c[K]      :各cluster的擁有資料數和
        self.ch_pt = 0                      #ch_pt          :紀錄變動點個數
        self.iterl = 0 
        self.sse2 = 0
        
        #計算acc時的參數
        self.origin_mass = []
        self.cent_name = []
        
    
    #run k-means    
    def run(self):
        
        #initialize tables
        self.kmeans_init()   #初始化centroid
                
        #first iteration 
        self.ch_pt = 0          
        self.iterl = 0
        self.sse2 = self.update_table()
        sse1 = self.sse2-1
        
        
        #update centroid & data clustering
        while self.iterl<self.MAX_ITER and sse1!=self.sse2 and self.ch_pt >self.MIN_PT  :
            sse1 = self.sse2
            self.iterl+=1
            self.update_cent()
            self.sse2 = self.update_table()
        
        self.table = self.table.astype(int)
        
        
        
    #Calculate average accuracy    
    def calculate_acc(self,iterate_times):
        self.acc = 0
        i = 0
        while( i < iterate_times):
            self.run()
            self.calculate_origin_mass()
            self.cent_name = self.centroid_names()
            # Avoid the rare situations that some cluster are gone
            if len(np.unique(self.cent_name)) != self.K:
                continue
                
            self.nearest_cluster()
            i += 1
            self.acc += accuracy_score(self.data_y,self.table)
            
            
            
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
        for i in range(len(self.data_x.index)):
            min_dist=float("inf")
            name = 0
            for j in range(self.K):
                dist = np.linalg.norm(self.data_x.iloc[i]-self.cent[j])
                if dist < min_dist:
                    min_dist = dist
                    name = j
                    
            self.table[i] = self.cent_name[name]
            
#---------------------------------------------------------------------------------
#------------------------------Subfunctions of run()------------------------------
#---------------------------------------------------------------------------------    
    def kmeans_init(self):
        
        self.data = self.data_x.values
        self.cent = np.zeros((self.K,self.DIM))
        self.table= np.zeros(self.DCNT)
        self.dis_k= np.zeros((self.K,self.DIM))
        self.cent_c=np.zeros(self.K)
                
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
        
    
    def cal_distance(self,x,y):
        sum = 0
        for i in range(self.DIM):
            sum = sum + (self.data[x][i]-self.cent[y][i])*( self.data[x][i]-self.cent[y][i])
        return sum

            
    def update_table(self):
        t_sse = 0
        self.ch_pt = 0 
        
        for i in range(self.DCNT):
            min_dis = self.cal_distance(i,0)
            min_k=0
            for j in range(1,self.K):
                dis = self.cal_distance(i,j)
                if(dis<min_dis):
                    min_dis = dis
                    min_k = j
            self.ch_pt+=(self.table[i]!=min_k)
            self.table[i] = min_k
            self.cent_c[min_k] +=1
            t_sse+=min_dis
            for j in range(self.DIM):
                self.dis_k[min_k][j]+=self.data[i][j]
                
        return t_sse

    def update_cent(self):
        for i in range(self.K):
            for j in range(self.DIM):
                if self.cent_c[i] != 0:
                    self.cent[i][j] = self.dis_k[i][j]/self.cent_c[i]
                else:
                    self.cent[i][j] = self.dis_k[i][j]

    def print_cent(self):
        print("Centroids:")
        print(self.cent)

    def print_result(self):
        print("K means:")
        print(self.table)
        print("sse = ",end='')
        print(self.sse2)
        print("ch_pt = ",end='')
        print(self.ch_pt)
        print("iter = ",end='')
        print(self.iterl)     
    


# ## Here comes our k-means
# ### Let's run for 1 time and check the performance

# In[5]:


result = our_k_means(data_x,data_y)
result.calculate_acc(1)


# ## Print out the center's coordinates

# In[6]:


result.print_cent()


# ## Then , we run it for (Simulate_time) times

# In[7]:


# Calculate
result.calculate_acc(Simulate_time)
result.acc


# # Let's run the k-means provides by sklearn
# ### -Then we can estimate how good we've done

# In[8]:


data_x = pd.DataFrame(data_x)
DIM  = len(data_x.columns)       #資料維度
K    = len(np.unique(data_y))    #叢聚個數
label= pd.DataFrame(data_y)


def k_means_sklearn(x):
    
    # KMeans 演算法
    kmeans_fit = cluster.KMeans(n_clusters = K).fit(x)

    # 測試分群結果
    cluster_labels = kmeans_fit.predict(x)
    
    return cluster_labels


# In[9]:


#all the clusters should be 1-D DataFrame which contains the same labels
def find_mass(k,dim,table,data):
    mass = np.zeros((k,dim))
    num = np.zeros(k)
    row_count = 0

    for i in table.values:
        for j in range(dim):
            mass[i][j] += data.iloc[row_count][j]
        row_count += 1
        num[i] += 1
        
    for i in range(k):
        for j in range(dim):
            mass[i][j] /= num[i]
    
    return mass


# In[10]:


def calculate_closest(k,origin,after_clustering):
    closest = np.zeros(k)
    for i in range(k):
        min_dist=float("inf")
        for j in range(k):
            dist = np.linalg.norm(after_clustering[i]-origin[j])
            if dist < min_dist:
                min_dist = dist
                closest[i]=j
    return closest 


# In[11]:


def relabel(origin_table,rename_table,target):
    target = target.replace(origin_table,rename_table)
    return target


# In[12]:



cluster_labels = k_means_sklearn(data_x) # sklearn.cluster.k_means_sklearn


# In[13]:


#A list that can be used to compared the order of label
temp = np.arange(K)
#Turn the labels trained by skilearn into DataFrame format
cluster_labels = pd.DataFrame(cluster_labels)

#Find the mass of data with trained labels
mass_sklean_kmeans = find_mass(K,DIM,cluster_labels.iloc[0:,0],data_x)
#Find tha mass of data with original labels
mass_origin        = find_mass(K,DIM,label.iloc[0:,0],data_x)
#Fine the correct cluster names & Relabel
closest_sklearn    = calculate_closest(K,mass_origin,mass_sklean_kmeans).astype(int)
cluster_labels     = relabel(temp,closest_sklearn ,cluster_labels)


# In[14]:


#Valid accuracy
sklearn_acc = accuracy_score(data_y,cluster_labels)
sklearn_acc

