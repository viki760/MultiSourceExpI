# calculate optimized g with fixed f
# calculate h score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
import time
import loading
class vanilla_fg():
    '''
    calculation with fixed feature extractor w/o transfer
    '''
    def __init__(self, data_path, model_path, t_id=0, batch_size=None):
        self.data_path = data_path
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.load(id=t_id, batch_size=batch_size)
        self.test_data = self.load(id=t_id, batch_size=batch_size, t=1)
        self.model_f, self.model_g = loading.load_model(path = model_path, id = t_id)
        self.n_label= int(next(iter(self.data))[1].max()+1)
    def load(self, id, batch_size = None, t=0):
        
        data = loading.load_data(path = self.data_path, id=id, batch_size=batch_size, t=t)
        return data
    # estimate the distribution of labels using given data samples
    def get_distribution_y(self, data_y):
        '''
        calculate the distribution of labels given data_y
        '''
        px = np.zeros(self.n_label)
        for i in range(self.n_label):
            for j in data_y:
                if j == i:
                    px[i] += 1
        return px / data_y.size
    
    # expectation of fx
    def get_exp(self, fx):
        return np.mean(fx, axis=1)
    # conditional expectation of fx
    def get_conditional_exp(self, fx, x, y):
        # calculate conditional expectation of fx
        ce_f = np.zeros((self.n_label, fx.shape[1]))
        for i in range(self.n_label):
            x_i = x[np.where(y==i)]
            fx_i = self.model_f(Variable(x_i).to(self.device)).cpu().detach().numpy() - fx.mean(0)
            ce_f[i] = fx_i.mean(axis=0)
        
        return ce_f
        
    def get_g(self):
        images, labels = next(iter(self.data))
        # take the first batch as input data
        labels_one_hot = torch.zeros(len(labels), self.n_label).scatter_(1, labels.view(-1,1), 1)
        f = self.model_f(Variable(images).to(self.device)).cpu().detach().numpy()
        g = self.model_g(Variable(labels_one_hot).to(self.device)).cpu().detach().numpy()
        # expectation and normalization of f and g
        e_f = f.mean(0)
        n_f = f - e_f
        # e_g = g.mean(0)
        # n_g = g - e_g
        gamma_f = n_f.T.dot(n_f) / n_f.shape[0]
        ce_f = self. get_conditional_exp(f, images, labels)
        g_y_hat = np.linalg.inv(gamma_f).dot(ce_f.T).T
        
        g_y = np.array([g[torch.where(labels == i)][0] for i in range(labels.max()+1)])
        
        g_rand = np.random.random(g_y.shape)
        return g_rand, g_y, g_y_hat
    # classification accuracy with different gy
    def get_accuracy(self, gc):
        
        acc = 0
        total = 0
        for images, labels in self.test_data:
            labels= labels.numpy()
            fc=self.model_f(Variable(images).to(self.device)).data.cpu().numpy()
            f_mean=np.sum(fc,axis=0)/fc.shape[0]
            fcp=fc-f_mean
            
            gce=np.sum(gc,axis=0)/self.n_label
            gcp=gc-gce
            fgp=np.dot(fcp,gcp.T)
            acc += (np.argmax(fgp, axis = 1) == labels).sum()
            # print(np.where(np.argmax(fgp, axis = 1) != labels))
            total += len(images)
        acc = float(acc) / total
        return acc

if __name__ == '__main__':
    DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/weight/"
    SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/results/"
    N_TASK = 21

    acc = np.zeros((N_TASK,3))
    for i in range(N_TASK):
        cal = vanilla_fg(DATA_PATH, MODEL_PATH,i)
        
        g_r, g, g_hat = cal.get_g()
        rand = cal.get_accuracy(gc=g_r)
        org = cal.get_accuracy(gc=g)
        hat = cal.get_accuracy(gc=g_hat)
        print("-------------task_id:{:d}-------------".format(i))
        print("random:{:.1%}\noriginal:{:.1%}\ncalculated:{:.1%}\n".format(rand, org, hat))
        acc[i] = rand, org, hat
    np.savetxt(SAVE_PATH+'vanilla_acc_table_'+time.strftime("%m%d", time.localtime())+'.npy', acc)