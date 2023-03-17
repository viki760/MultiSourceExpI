'''
training of source models
with normalization layer in feature extractor
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# class LeNet(nn.Module): 					
#     def __init__(self):						
#         super(LeNet, self).__init__()    	
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):			 
#         x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
#         x = self.pool1(x)            # output(16, 14, 14)
#         x = F.relu(self.conv2(x))    # output(32, 10, 10)
#         x = self.pool2(x)            # output(32, 5, 5)
#         x = x.view(-1, 32*5*5)       # output(32*5*5)
#         x = F.relu(self.fc1(x))      # output(120)
#         x = F.relu(self.fc2(x))      # output(84)
#         x = self.fc3(x)              # output(10)
#         return x


DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
# DATA_PATH = r"D:\task\research\codes\MultiSource\wsl\2\multi-source\data_set_2\\"
N_TASK = 21

class Net_f(nn.Module):
    def __init__(self):
        super(Net_f, self).__init__()
        googlenet = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        self.feature=torch.nn.Sequential(*list(googlenet.children())[0:18])
        self.fc1 = nn.Linear(1024,32)
        self.fc2 = nn.Linear(32,10)
        self.BN = nn.BatchNorm1d(10)

    def forward(self,x):
        out=self.feature(x)
        out=out.view(-1,1024)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        out=self.BN(out)

        return out     

class Net_g(nn.Module):
    def __init__(self,num_class=2, dim=10):
        super(Net_g, self).__init__()

        self.fc=nn.Linear(num_class, dim)

    def forward(self,x):
        out=self.fc(x)

        return out

def corr(f,g):
    k = torch.mean(torch.sum(f*g,1))
    return k
    
def cov_trace(f,g):
    cov_f = torch.mm(torch.t(f),f) / (f.size()[0]-1.)
    cov_g = torch.mm(torch.t(g),g) / (g.size()[0]-1.)
    return torch.trace(torch.mm(cov_f, cov_g))


def load_data(id, batch_size=100, t=0):
    '''
    task_id    range(21)
    batch_size 100
    t          0(train)/1(test)
    '''
    if t==0:
        x = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_train.npy").transpose((0,3,1,2))).to(torch.float32)
        y = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_train.npy"))
    else:
        x = torch.from_numpy(np.load(DATA_PATH+"x"+str(id)+"_test.npy").transpose((0,3,1,2))).to(torch.float32)
        y = torch.from_numpy(np.load(DATA_PATH+"y"+str(id)+"_test.npy"))
    data = torch.utils.data.DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return data

# id = sys.argv[1]
batch_size = 10
num_epochs = 20
lr = 0.0001

acc_list = np.zeros(N_TASK)

for id in range(N_TASK):
    train_loader = load_data(id, batch_size, 0)
    test_loader = load_data(id, batch_size, 1)


    model_f = Net_f().to(device)
    model_g = Net_g().to(device)					  				# 定义训练的网络模型
    # loss_function = nn.CrossEntropyLoss() 				

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer_fg = torch.optim.Adam(list(model_f.parameters())+list(model_g.parameters()),lr=lr)

    # Train the model
    total_step = len(train_loader)
    print(total_step)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            model_f.train()
            model_g.train()
            labels_one_hot = torch.zeros(len(labels), 2).scatter_(1, labels.view(-1,1), 1)
            # Forward pass
            optimizer_fg.zero_grad()
            f = model_f(Variable(images).to(device))
            g = model_g(Variable(labels_one_hot).to(device))
            
            # Backward and optimize
            # loss = (-2)*corr(f,g)
            # loss += 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum()
            # loss += cov_trace(f,g) 

            loss = (-2)*corr(f,g) + 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum() + cov_trace(f,g)

            # loss.retain_grad()
            loss.backward()

            # for name, parms in model_f.named_parameters():
            #     print(name, parms.requires_grad)
            #     print(name, parms.grad)
            #     break 

            optimizer_fg.step()

            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            # print(loss.grad)

    # Test the model
    model_f.eval()
    model_g.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        acc = 0
        total = 0

        for images, labels in test_loader:

            labels= labels.numpy()
            fc=model_f(Variable(images).to(device)).data.cpu().numpy()
            f_mean=np.sum(fc,axis=0)/fc.shape[0]
            fcp=fc-f_mean
            
            labellist=torch.Tensor([[1,0],[0,1]])
            gc=model_g(Variable(labellist).to(device)).data.cpu().numpy()
            gce=np.sum(gc,axis=0)/gc.shape[0]
            gcp=gc-gce
            fgp=np.dot(fcp,gcp.T)
            acc += (np.argmax(fgp, axis = 1) == labels).sum()
            total += len(images)

        acc = float(acc) / total

        print('Test Accuracy of the model on the 1000 test images: {} %'.format(100 * acc))
        
        acc_list[id] = acc

    # # Save the model checkpoint
    # torch.save(model.state_dict(), 'model.ckpt')

    print('Finished Training')

    save_path_f = 'weight/f_task_n_'+str(id)+'.pth'
    torch.save(model_f.state_dict(), save_path_f)
    save_path_g = 'weight/g_task_n_'+str(id)+'.pth'
    torch.save(model_g.state_dict(), save_path_g)

    
np.savetxt('results/acc_n_0308.npy', acc_list)