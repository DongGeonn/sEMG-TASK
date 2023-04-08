import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
torch.manual_seed(1)


print("lr=1")
model=nn.Sequential(nn.Linear(36,17))                                           
optimizer=optim.SGD(model.parameters(), lr=1)
losses0=[]
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=model(x_train)
    cost=F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    losses0.append(cost.item())
    if epoch%1000==0:
        print('Epoch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        
# train 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_train)
    correct_prediction = torch.argmax(hypothesis, 1) == y_train
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Train Accuracy:', accuracy.item()*100)
    
# test 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_test)
    correct_prediction = torch.argmax(hypothesis, 1) == y_test
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Test Accuracy:', accuracy.item()*100)   
        
print("")


print("lr=0.1")
model=nn.Sequential(nn.Linear(36,17))                                           
optimizer=optim.SGD(model.parameters(), lr=1e-1)
losses1=[]
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=model(x_train)
    cost=F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    losses1.append(cost.item())
    if epoch%1000==0:
        print('Epoch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

# train 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_train)
    correct_prediction = torch.argmax(hypothesis, 1) == y_train
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Train Accuracy:', accuracy.item()*100)
    
# test 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_test)
    correct_prediction = torch.argmax(hypothesis, 1) == y_test
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Test Accuracy:', accuracy.item()*100)   
        
print("")   
      
    
print("lr=0.01")
model=nn.Sequential(nn.Linear(36,17))                                           
optimizer=optim.SGD(model.parameters(), lr=1e-2)
losses2=[]
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=model(x_train)
    cost=F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    losses2.append(cost.item())
    if epoch%1000==0:
        print('Epoch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

# train 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_train)
    correct_prediction = torch.argmax(hypothesis, 1) == y_train
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Train Accuracy:', accuracy.item()*100)
    
# test 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_test)
    correct_prediction = torch.argmax(hypothesis, 1) == y_test
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Test Accuracy:', accuracy.item()*100)   
print("") 
       
        
print("lr=0.001")
model=nn.Sequential(nn.Linear(36,17))                                           
optimizer=optim.SGD(model.parameters(), lr=1e-3)
losses3=[]
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=model(x_train)
    cost=F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    losses3.append(cost.item())
    if epoch%1000==0:
        print('Epoch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        
# train 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_train)
    correct_prediction = torch.argmax(hypothesis, 1) == y_train
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Train Accuracy:', accuracy.item()*100)
    
# test 정확도 측정
with torch.no_grad():         
    hypothesis=model(x_test)
    correct_prediction = torch.argmax(hypothesis, 1) == y_test
    accuracy = correct_prediction.float().mean()
    np_hypothesis=hypothesis.detach().numpy()
    np_y_train=y_train.detach().numpy()
    np_y_train_one_hot=y_train.detach().numpy()
    print('Test Accuracy:', accuracy.item()*100)   