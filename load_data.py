# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 03:17:33 2019

@author: KETAN
"""


from scipy.io import wavfile as w
import numpy as np
import os
from MFCC import mfcc_2 
import math

no_files_per_person=5
no_mfcc_coef=49

person_no=np.array([241,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271])

l=len(person_no)

#the last one is removed though for label
person=['']
test_1=np.zeros((int(no_files_per_person*l),no_mfcc_coef))
for j in range(1,l+1):
    address=r"C:\Users\KETAN\Project btech\Speaker Recognition\Speaker_Recognition_VCTK\Speaker_Recognition_VCTK\Test\p"+str(person_no[j-1])
    os.chdir(address)
    n=0
    person=['']
    for root, dirs, files in os.walk("."):  
        for filename in files:
            person.append(str(filename))
            n+=1
            
    
    
    
        for i in range(1,no_files_per_person+1):
            test_1[i-1+(j-1)*no_files_per_person,:]=(mfcc_2(person[i]).T)
            test_1[i-1+(j-1)*no_files_per_person,-1]=j
            
#randomly arrange these input rows to feed to NN
test_1=test_1[np.random.permutation(len(test_1[:,0])),:]

items=len(test_1)
split=60
testt=math.floor(items*split/100)
train=test_1[0:testt,:]
test=test_1[testt:,:]    

#data preprosessing

temp=train[:,0:no_mfcc_coef-1]
mean=np.mean(temp,0)
std=np.std(temp,0)

#zero mean
train[:,0:no_mfcc_coef-1]=train[:,0:no_mfcc_coef-1]-mean
test[:,0:no_mfcc_coef-1]=test[:,0:no_mfcc_coef-1]-mean

#unit variance
train[:,0:no_mfcc_coef-1]=train[:,0:no_mfcc_coef-1]/std
test[:,0:no_mfcc_coef-1]=test[:,0:no_mfcc_coef-1]/std

'''
temp=temp-np.mean(temp,0)#zero centered
temp=temp/np.std(temp,0)#unit variance
#we do this because all input dimentions are equally important
test_1[:,0:48]=temp

'''
'''
#PCA
temp=train[:,0:no_mfcc_coef-1]
cov=(temp.T@temp)/temp.shape[0]
[U,d,V]=np.linalg.svd(cov)
index=np.argsort(d)
#d=d[index]
V=V[:,index]
keep_coef=48

#VV=np.delete(V,np.arange(no_mfcc_coef-1-keep_eig),axis=1)
train[:,0:no_mfcc_coef-1]=(train[:,0:no_mfcc_coef-1]@U)/np.sqrt(d)
train=np.delete(train,np.arange(no_mfcc_coef-keep_coef-1),axis=1)

test[:,0:no_mfcc_coef-1]=(test[:,0:no_mfcc_coef-1]@U)/np.sqrt(d)
test=np.delete(test,np.arange(no_mfcc_coef-keep_coef-1),axis=1)

'''
