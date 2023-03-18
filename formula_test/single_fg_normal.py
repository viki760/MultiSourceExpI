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

DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/formula_test/weight/"
SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/formula_test/results/"
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
ACC_ALL = []

for id in range(N_TASK):
    for id_s in range(N_TASK):
        train_loader_s = load_data(id_s, batch_size, 0)
        train_loader_t = load_data(id, batch_size, 0)
        test_loader = load_data(0, batch_size, 1)


        model_f = Net_f().to(device)
        model_g = Net_g().to(device)					  			
        # loss_function = nn.CrossEntropyLoss() 				

        # Loss and optimizer
        # criterion = nn.CrossEntropyLoss()
        optimizer_fg = torch.optim.Adam(list(model_f.parameters())+list(model_g.parameters()),lr=lr)
        alpha = 0.4
        ACC = []

        # Train the model
        total_step = len(train_loader_s)
        print(total_step)

        for epoch in range(num_epochs):
            for i, (data_s, data_t) in enumerate(zip(train_loader_s, train_loader_t)):
                image_s, label_s = data_s[0], torch.zeros(len(data_s[1]), 2).scatter_(1, data_s[1].view(-1,1), 1)
                image_t, label_t = data_t[0], torch.zeros(len(data_t[1]), 2).scatter_(1, data_t[1].view(-1,1), 1)
                model_f.train()
                model_g.train()
                # labels_one_hot = torch.zeros(len(labels), 2).scatter_(1, labels.view(-1,1), 1)
                # Forward pass
                optimizer_fg.zero_grad()
                fs = model_f(Variable(image_s).to(device))
                gs = model_g(Variable(label_s).to(device))
                ft = model_f(Variable(image_t).to(device))
                gt = model_g(Variable(label_t).to(device))
                # Backward and optimize
                # loss = (-2)*corr(f,g)
                # loss += 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum()
                # loss += cov_trace(f,g) 

                # print(np.array([alpha*(-2)*corr(fs,gs).item(), (1-alpha)*(-2)*corr(ft,gt).item(), 2*((torch.sum(ft,0)/ft.size()[0])*(torch.sum(gt,0)/gt.size()[0])).sum().item(), cov_trace(ft,gt).item()]))
                loss = alpha*(-2)*corr(fs,gs) + (1-alpha)*(-2)*corr(ft,gt) + 2*((torch.sum(ft,0)/ft.size()[0])*(torch.sum(gt,0)/gt.size()[0])).sum() + cov_trace(ft,gt)

                # loss.retain_grad()
                loss.backward()

                # for name, parms in model_f.named_parameters():
                #     print(name, parms.requires_grad)
                #     print(name, parms.grad)
                #     break 

                optimizer_fg.step()

                
                if (i+1) % 10 == 0:
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

                ACC.append(acc)

                print('Test Accuracy of the model on the 1000 test images: {} %'.format(100 * acc))

        # # Save the model checkpoint
        # torch.save(model.state_dict(), 'model.ckpt')

        print('Finished Training')
        print(ACC)
        ACC_ALL.append(ACC)


        save_path_f = MODEL_PATH + 'f_task_t='+str(id)+'_s='+str(id_s)+'_alpha='+str(alpha)+'.pth'
        torch.save(model_f.state_dict(), save_path_f)
        save_path_g = MODEL_PATH + 'g_task_t='+str(id)+'_s='+str(id_s)+'_alpha='+str(alpha)+'.pth'
        torch.save(model_g.state_dict(), save_path_g)

print(ACC_ALL)
np.savetxt(ACC_ALL, (SAVE_PATH+'acc_transfer.npy'))