'''
functions for loading data and model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset,DataLoader,TensorDataset


# definition of f-g network
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


# load data
def load_data(path, batch_size=None, id=0, t=0):
    '''
    load data as tensor dataset with given task id from DATA_PATH
    
    batch_size  default:200
    task_id    range(21); default:0
    t          0(train)/1(test); default:train
    '''
    if t==0:
        x = torch.from_numpy(np.load(path+"x"+str(id)+"_train.npy").transpose((0,3,1,2))).to(torch.float32)
        y = torch.from_numpy(np.load(path+"y"+str(id)+"_train.npy"))
    else:
        x = torch.from_numpy(np.load(path+"x"+str(id)+"_test.npy").transpose((0,3,1,2))).to(torch.float32)
        y = torch.from_numpy(np.load(path+"y"+str(id)+"_test.npy"))
    
    # in case no input
    if batch_size == None:
        batch_size = x.__len__()
        # print(batch_size)

    data = torch.utils.data.DataLoader(TensorDataset(x, y), batch_size, shuffle=True)

    return data


# load model
def load_model(path=None, id=0, t=1):
    '''
    load model with given model id from MODEL_PATH; if no source selection return initialized model

    model_id    range(21)   0 for target model, others for source models
    t          0(train)/1(eval)
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    model_f = Net_f().to(device)
    model_g = Net_g().to(device)

    if not path == None:
        model_f.load_state_dict(torch.load(path+'f_task_n_'+str(id)+'.pth', map_location=device))
        model_g.load_state_dict(torch.load(path+'g_task_n_'+str(id)+'.pth', map_location=device))

    if t==0:
        model_f.train()
        model_g.train()
    else:
        model_f.eval()
        model_g.eval()

    return model_f, model_g

def load_multi_model(path=None, id_list=0, t=1):

    res = [load_model(path=path, id=i, t=t) for i in id_list]
    
    class Net_multiple(nn.Module):
        def __init__(self, model_list):
            super(Net_multiple, self).__init__()

            # ((1, 2), (5, 6), (10, 11))
            # ->
            #  [(1, 5, 10), (2, 6, 11)]
            tmp = zip(*model_list)
            self.f_model_list, self.g_model_list = [nn.ModuleList(x) for x in tmp]

        def forward(self, x, y):
            feat_f_list = [f(x) for f in self.f_model_list]
            feat_g_list = [g(y) for g in self.g_model_list]

            # out = (torch.cat(feat_f_list, dim=1), torch.cat(feat_g_list, dim=1))  # (bs, n * dim)

            out = (torch.stack(feat_f_list, dim=1).mean(dim=1), torch.stack(feat_g_list, dim=1).mean(dim=1))

            return out   
    return Net_multiple(res)
