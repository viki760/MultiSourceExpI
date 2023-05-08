import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import CIFAR100
import torch.nn.functional as F
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import cvxpy as cvx
import scipy.io as scio
from torch.utils.data import Dataset,DataLoader,TensorDataset
import sys
import logging
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# N = str(sys.argv[1])
# print('\n---------------------------circle:N='+N+'--------------------------------\n')

DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
ALL_TASK = 10
# TYPE = 'c'
TYPE = sys.argv[1]

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

# transform
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
    ])

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
    if id==0 and t == 0:
        data = torch.utils.data.DataLoader(TensorDataset(x[:10], y[:10]), batch_size=batch_size, shuffle=True)
    else:
        data = torch.utils.data.DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    return data

def train(siter, id):
    sample, label = next(siter[id - 1])
    labels_one_hot = torch.zeros(len(label), 2).scatter_(1, label.view(-1,1), 1)
    f = model_f(Variable(sample).to(device))
    g = model_g(Variable(labels_one_hot).to(device))
    f = f - torch.mean(f,0)
    g = g - torch.mean(g,0)

    loss = alpha0[id].item()*(-2) * corr(f, g)
    return loss

if __name__ == "__main__":

    
    log_path = './log/'

    logtime = time.strftime('%m%d_%H%M_%S_')

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(log_path, logtime + 'train.log'), level=logging.INFO)

    target = load_data(0)
    testset = load_data(0, t=1)
    targetiter = iter(target)
    samplest, labelst=next(targetiter)
    labels_one_hot_t = torch.zeros(len(labelst), 2).scatter_(1, labelst.view(-1,1), 1)


    lr = 0.0001
    epoch = 15

    if TYPE == "c":
        # OTCE
        id_list = np.array([9, 4, 5, 2, 7, 8, 6, 1, 3])
    elif TYPE == "r":
        # random
        id_list = np.array([8, 3, 5, 4, 7, 6, 1, 2, 9])
    elif TYPE == "a":
        # alpha
        id_list = np.array([3, 2, 9, 7, 4, 6, 1, 5, 8])
    else:
        id_list = np.arange(1,10)
    
    for N_TASK in range(2, ALL_TASK+1):
        time_start=time.time()

        alpha0=torch.rand(N_TASK)
        
        # alpha0 = torch.tensor([0.050390117, 0.047139142, 0.045838752, 0.051040312, 0.046163849, 0.047789337, 0.046814044, 0.045838752, 0.047789337, 0.045513654, 0.048439532, 0.048114434, 0.049089727, 0.047789337, 0.046814044, 0.045838752, 0.046488947, 0.047464239, 0.048114434, 0.049089727, 0.048439532])
        # alpha0 = torch.tensor([0.050390117, 0.047139142, 0.045838752, 0.051040312, 0.046163849, 0.047789337, 0.046814044, 0.045838752, 0.047789337, 0.045513654, 0.048439532])
        # alpha0 = torch.tensor([0.050390117, 0.047139142, 0.045838752, 0.051040312, 0.046163849, 0.047789337, 0.046814044, 0.045838752])
        # alpha0 = torch.tensor([0.12660375, 0.12478026, 0.12478542, 0.12478752, 0.12478631, 0.12480189, 0.12469672, 0.12475813])
        # alpha0 = torch.tensor([0.4415618, 0.08049484, 0.06581235, 0.07855702, 0.10422341, 0.0798626, 0.07626385, 0.07322414])
        # alpha0 = torch.tensor([0.16561725, 0.10530306, 0.1235456 , 0.11943303, 0.11991479, 0.12380397, 0.12111729, 0.12126502])
        # alpha0 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
        alpha0 = alpha0 / alpha0.sum()
        print(alpha0)


        ids = id_list[:N_TASK-1]
        print(ids)

        model_f = Net_f().to(device)
        model_g = Net_g().to(device)
        optimizer_fg = torch.optim.Adam(list(model_f.parameters()) + list(model_g.parameters()), lr = lr)
        
        losslist = []
        acclist = [0]
        
        
        

        for i in range(epoch):
            
            sourceiter = []
            for id in ids:
                source = load_data(id, batch_size=25)
                sourceiter.append(iter(source))

            losscc=[]
            for k in range(len(source)): 

                model_f.train()
                model_g.train()
                optimizer_fg.zero_grad()

                ft = model_f(Variable(samplest).to(device))
                gt = model_g(Variable(labels_one_hot_t).to(device))
                ft = ft - torch.mean(ft,0)
                gt = gt - torch.mean(gt,0)
                loss = alpha0[0].item()*(-2) * corr(ft, gt)

                for i in range(1, N_TASK):
                    loss += train(sourceiter, i)

                # loss += 2 * (torch.mean(ft,0) * torch.mean(gt,0)).sum()
                loss += cov_trace(ft, gt)
                loss += torch.sum(abs(alpha0))

                losscc.append(loss.item())
                loss.backward()
                optimizer_fg.step()


                model_f.eval()
                model_g.eval()

                acc=0
                total=0

                fc = model_f(Variable(samplest).to(device)).data.cpu().numpy()
                f_mean = np.sum(fc, axis = 0) / fc.shape[0]
                labellist = torch.Tensor([[1, 0], [0, 1]])
                gc = model_g(Variable(labellist).to(device)).data.cpu().numpy()
                gce = np.sum(gc,axis = 0) / gc.shape[0]
                gcp = gc - gce

                for k, data in enumerate(testset, 0):
                    samples, labels = data
                    labels = labels.numpy()
                    fc = model_f(Variable(samples).to(device)).data.cpu().numpy()
                    fcp = fc-f_mean
                    fgp = np.dot(fcp,gcp.T)
                    acc += (np.argmax(fgp, axis = 1) == labels).sum()
                    total += len(samples)

                acc = float(acc) / total
                print(acc)
                # if acc > 0.7:
                if acc > (max(acclist)):
                    print('changepara')
                    finalacc = acc
                    paraf = model_f.state_dict()
                    parag = model_g.state_dict()
                acclist.append(acc)


            losslist.append(sum(losscc) / len(losscc))
            print(sum(losscc) / len(losscc))
        #-------------------renewalpha
            

        #-----start loop------------
        print(losslist)
        print(acclist)
        print(finalacc)
        print(alpha0)
        logging.info('type:{},\n N:{},\n losslist: {},\n acclist: {},\n finalacc: {},\n alpha: {}'.format(TYPE, N_TASK, losslist, acclist, finalacc, alpha0))
        torch.save(paraf, 'mpara/cifar100f_set_3_alpha_'+str(N_TASK)+'_'+TYPE+'.pth')
        torch.save(parag, 'mpara/cifar100g_set_3_alpha_'+str(N_TASK)+'_'+TYPE+'.pth')

        time_end=time.time()
        print(time_end-time_start)
