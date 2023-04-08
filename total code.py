# 저장된 데이터 파일 불러오기
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import scipy.io

mat_file_name="C:/Users/user/Documents/sEMG-TASK/sEMG.mat"
mat_file=scipy.io.loadmat(mat_file_name)
emg=mat_file['emg']
label=mat_file['label']
rep=mat_file['repetition']

# data_first 빈 딕셔너리 만들기
data_first={}
for num_label in range(1,18):                                                          
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        data_first[key_name]=[]


# data 딕셔너리의 value에 입력값을 추가하는 함수 설정
def dicplus(dictionary, key, value):                                            
    value_list=dictionary[key]
    value_list.append(value)
    dictionary[key]=value_list


# data_first 딕셔너리의 각 key에 value 입력
for i in range(emg.shape[0]):                                                  
    for num_label in range(1,18):
        for num_rep in range(1,7):
            if num_label==label[i] and num_rep==rep[i]:
                key_name="L"+str(num_label)+"-"+str(num_rep)       
                dicplus(data_first, key_name, emg[i,:])


# data_first 딕셔너리의 각 valve를 numpy형태로 설정                
import numpy as np                  
for num_label in range(1,18):                                                   
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        data_first[key_name]=np.array(data_first[key_name])
        
# data_second 딕셔너리 생성 & 입력        
data_second={}
for num_label in range(1,18):                                                          
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        array=data_first[key_name]
        len_array=(array.shape[0]//400)*400
        array=array[:len_array,:]
        array=array.reshape((-1,400,12))
        data_second[key_name]=array
        
# data_MAV 딕셔너리 생성 & 입력
data_MAV={}
for num_label in range(1,18):                                                   
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        mav=data_second[key_name]
        mav=np.absolute(mav)
        mav=np.mean(mav, axis=1) 
        data_MAV[key_name]=mav 
        
        
# data_VAR 딕셔너리 생성 & 입력
data_VAR={}
for num_label in range(1,18):                                                   
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        var=data_second[key_name]
        var=var**2
        var=np.sum(var, axis=1)
        var=var/(400-1)
        data_VAR[key_name]=var
        
        
# data_WL 딕셔너리 생성 & 입력
data_WL={}
for num_label in range(1,18):                                                   
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        wl=data_second[key_name]
        wl1=wl[:,0:-1,:]
        wl2=wl[:,1:,:]
        wl=wl1-wl2
        wl=np.absolute(wl)
        wl=np.sum(wl, axis=1)        
        data_WL[key_name]=wl 
        
        
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
torch.manual_seed(1)


# x_train, y_train, y_train_one_hot data 만들기
x_train=torch.Tensor([])
y_train=torch.Tensor([])
for num_label in range(1,18):                                                   
    for num_rep in range(1,5):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        mav=torch.tensor(data_MAV[key_name])
        var=torch.tensor(data_VAR[key_name])
        wl=torch.tensor(data_WL[key_name])
        x_train_t1=torch.cat((mav,var,wl), dim=1)
        x_train=torch.cat((x_train, x_train_t1), dim=0)
        y_train_t1=torch.ones(x_train_t1.shape[0],1)*(num_label-1)
        y_train=torch.cat((y_train, y_train_t1), dim=0)
y_train=y_train.reshape((-1,))
y_train=np.array(y_train)
y_train=torch.LongTensor(y_train)
y_train_one_hot=torch.zeros(y_train.shape[0],17)
y_train_one_hot.scatter_(1,y_train.unsqueeze(1), 1) 



# x_test, y_test, y_test_one_hot data 만들기
x_test=torch.Tensor([])
y_test=torch.Tensor([])
for num_label in range(1,18):                                                   
    for num_rep in range(5,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        mav=torch.tensor(data_MAV[key_name])
        var=torch.tensor(data_VAR[key_name])
        wl=torch.tensor(data_WL[key_name])
        x_test_t1=torch.cat((mav,var,wl), dim=1)
        x_test=torch.cat((x_test, x_test_t1), dim=0)
        y_test_t1=torch.ones(x_test_t1.shape[0],1)*(num_label-1)
        y_test=torch.cat((y_test, y_test_t1), dim=0) 
y_test=y_test.reshape((-1,))
y_test=np.array(y_test)
y_test=torch.LongTensor(y_test)
y_test_one_hot=torch.zeros(y_test.shape[0],17)
y_test_one_hot.scatter_(1,y_test.unsqueeze(1), 1) 


# 단층 퍼셉트론으로 분류
model=nn.Sequential(nn.Linear(36,17))                                           
optimizer=optim.SGD(model.parameters(), lr=1e-1)
losses_36_17=[]
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=model(x_train)
    cost=F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    losses_36_17.append(cost.item())
    if epoch%1000==0:
        print('Epoch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

# train 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_train)
    correct_prediction = torch.argmax(hypothesis, 1) == y_train
    accuracy = correct_prediction.float().mean()
    print('Train Accuracy:', accuracy.item()*100)
    
# test 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_test)
    correct_prediction = torch.argmax(hypothesis, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print('Test Accuracy:', accuracy.item()*100)
    
print("")


# 다층 퍼셉트론으로 분류
model=nn.Sequential(nn.Linear(36,50),nn.ReLU(), nn.Linear(50,50),nn.ReLU(),
                    nn.Linear(50,50),nn.ReLU(), nn.Linear(50,17))              
optimizer=optim.SGD(model.parameters(), lr=1e-1)
losses_36_50_50_17=[]
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=model(x_train)
    cost=F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    losses_36_50_50_17.append(cost.item())
    if epoch%1000==0:
        print('Epoch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        
# train 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_train)
    correct_prediction = torch.argmax(hypothesis, 1) == y_train
    accuracy = correct_prediction.float().mean()
    print('Train Accuracy:', accuracy.item()*100)
    
# test 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_test)
    correct_prediction = torch.argmax(hypothesis, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print('Test Accuracy:', accuracy.item()*100)  
    