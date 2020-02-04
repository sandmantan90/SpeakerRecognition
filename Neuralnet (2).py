import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter as SM

input_size=noMfcc
output_size=211
def get_correct_num(pred,lables):
    return pred.argmax(dim=1).eq(lables).sum().item()


def get_error(pred,lables):
    if (pred.argmax(dim=1).eq(lables)) :
        print(pred.argmax(dim=1),lables)
    
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.linear1=nn.Linear(input_size,100)
        self.bn1=nn.BatchNorm1d(100)
        self.dpout=nn.Dropout(.25)
        self.linear2=nn.Linear(100,100)
        self.bn2=nn.BatchNorm1d(100)
        self.linear3=nn.Linear(100,100)
        self.bn3=nn.BatchNorm1d(100)
        '''
        self.linear2=nn.Linear(1000,1000)
        self.linear3=nn.Linear(1000,1000)
        self.linear4=nn.Linear(1000,1000)
        self.linear5=nn.Linear(1000,1000)
        '''
        self.linear6=nn.Linear(100,output_size)
        
    def forward(self,x):
        y_pred=self.bn1(self.linear1(x))
        y_pred=self.dpout(F.tanh(y_pred))
        
        
        '''
        y_pred=self.dpout(self.bn2(self.linear2(y_pred)))
        y_pred=((F.tanh(y_pred)))
        
        y_pred=self.bn3(self.linear3(y_pred))
        y_pred=self.dpout(F.tanh(y_pred))
       
        y_pred=self.linear4(y_pred)
        y_pred=F.tanh(y_pred)
        y_pred=self.linear5(y_pred)
        y_pred=F.tanh(y_pred)
        '''
        y_pred=self.linear6(y_pred)
        
        
        return( y_pred)      
        
model=Network()
#model.cuda()
criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(), lr=.001)   
b=1
train=torch.tensor(train)
i=train
o=trainLabel
i,o=Variable(i),Variable(o)
#i.o=i.cuda(),o.cuda()

test=torch.tensor(test)
p=test
q=testLabel

trainAcc=[]
testAcc=[]
for epoch in range(5000):
    
    total_loss=0
    total_correct=0
    model=model.train()
    pred=model(train.float())
        
    loss=F.cross_entropy(pred,trainLabel.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  
        
    total_loss =total_loss+loss.item()
    total_correct=total_correct+get_correct_num(pred,trainLabel.long())
    with torch.no_grad():
        model=model.eval()
        pred2=model(test)
        tot=get_correct_num(pred2,testLabel.long())
    trainAcc.append(total_correct/len(train[:,1]))
    testAcc.append(tot/len(test[:,1]))
    print("epoch: {}  total correct: {} = {}%  loss: {} total correct test: {} = {}%".format(epoch,total_correct,total_correct/len(train[:,1]),loss,tot,tot/len(test[:,1])))    

plt.close('all')
plt.figure()
plt.plot(trainAcc,label='train')
plt.plot(testAcc,label='test')
plt.legend()
plt.show()
#get_error(yy_pred,q.long()-1)   