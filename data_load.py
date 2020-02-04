# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:09:28 2019

@author: KETAN
"""
import os
import numpy as np
import librosa
import torch
from data_preProcess import preProcess

dur=2
samplingR=16000
samples=samplingR*dur
noMfcc=50
trainPercent=.65
filesPerPerson=10
noPerson=220
personCount=0
fileCount=0
trainLabel,testLabel,train,test=[],[],[],[]


address=r"C:\Users\KETAN\Project btech\Speaker Recognition\Speaker_Recognition_VCTK\Speaker_Recognition_VCTK\Train"
os.chdir(address)
noise, sampling=librosa.load(r"C:\Users\KETAN\Downloads\noise7.wav",sr=sampling)
for root, subdirs, files in os.walk('.'):
    
    if subdirs==[]:
        personCount+=1
        print(personCount,'HAV')
        print(root)
        fileCount=0
        for file in files:
            address=root+'\\'+file
            if (os.path.splitext(file)[1]=='.wav')&(file[0]!='.'):
                
           
                audio, sampling = librosa.load(address, sr = samplingR)
                
                randd=np.random.uniform(0,.1)
                
                audio=np.fft.fft(audio)
                audio=audio*100/np.max(abs(audio))#fft normalizing
                audio=np.real(np.fft.ifft(audio))
                #audio+=noise[:len(audio)]*randd
                fileCount+=1  
                
                if len(audio)<(samples+sampling):
                    while len(audio)<(samples+sampling):
                        audio = np.concatenate((audio,audio), axis=0)
                    audio = audio[int(.5*sampling):int(samples+.5*sampling)]
                    window=np.hamming(len(audio))#check if this works!!
                    audio=audio*window
                
                data=librosa.feature.mfcc(audio,sr=sampling,n_mfcc=noMfcc )
                
                
                data=np.sum(data,axis=1).reshape(noMfcc,1)
                #data=data/abs(data[0])
                label=personCount-1
                
                #train-test split
                die=np.random.uniform(0,1)
                if die<trainPercent:
                    train.append(data)
                    trainLabel.append(label)
                else:
                    test.append(data)
                    testLabel.append(label)
                  
                    
                if fileCount>=filesPerPerson:
                    break
    if personCount>=noPerson:
                    break     
                
train,test=preProcess(train,test)   
trainLabel=torch.FloatTensor(trainLabel).view(len(trainLabel))   
testLabel=torch.FloatTensor(testLabel).view(len(testLabel))                         
print(train.shape)
print(test.shape)                                        
                    
                        
