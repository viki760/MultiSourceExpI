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
import sys
sys.path.append("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp")
import fg_train.util.loading as loading
import hydra
from omegaconf import DictConfig
import time
import os
import json
# from tqdm import tqdm

class fg:
    '''
    calculation with fixed feature extractor
    '''
    def __init__(self, cfg, t_id=0) -> None:

        self.data_path = cfg.path.data
        self.model_path = cfg.path.wd+"fg_train/weight/"
        self.load_path = cfg.path.wd+"fg_train/load/"
        self.save_path = cfg.path.wd+"fg_train/results/"
        self.log_path = cfg.path.wd+"fg_train/log/"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise Warning('Cuda unavailable. Now running on CPU.')

        self.batch_size = cfg.setting.batch_size
        self.lr = cfg.setting.lr
        self.num_epochs = cfg.setting.num_epochs

        self.t_id = [t_id] if isinstance(t_id, int) else t_id


    def load(self, id) -> None:

        self.data = loading.load_data(path = self.data_path, id = id, batch_size = self.batch_size, t = 0)
        self.test_data = loading.load_data(path = self.data_path, id = id, batch_size = self.batch_size, t = 1)
        self.model_f, self.model_g = loading.load_model(path = self.model_path, id = id)
        self.n_label= int(next(iter(self.data))[1].max()+1)
        # self.n_label= [int(d[1].max()+1) for d in self.data][0]
        # self.data = self.load(id=self.t_id, batch_size=self.batch_size)
        

        self.images, self.labels = next(iter(self.data))
        labels_one_hot = torch.zeros(len(self.labels), self.n_label).scatter_(1, self.labels.view(-1,1), 1)
        self.f = self.model_f(Variable(self.images).to(self.device)).cpu().detach().numpy()
        self.g = self.model_g(Variable(labels_one_hot).to(self.device)).cpu().detach().numpy()

        torch.save(
            {
                "id": id,
                "x": self.images,
                "y": self.labels,
                "f": self.f,
                "g": self.g,
                "n_label": self.n_label,
                "test_data": self.test_data,
            }, f"{self.load_path}test{id}.pt"
        )

        # res = {}
        # for i in range(10):
        # res[f"task{i}"] = {}

    def read_from_load(self, id) -> None:
        data = torch.load(f"{self.load_path}test{id}.pt")
        self.images, self.labels, self.f, self.g, self.n_label, self.test_data = data["x"], data["y"], data["f"], data["g"], data["n_label"], data["test_data"]


    # estimate the distribution of labels using given data samples
    def get_distribution_y(self, data_y):
        "calculate the distribution of labels given data_y"
        px = np.zeros(self.n_label)
        for i in range(self.n_label):
            for j in data_y:
                if j == i:
                    px[i] += 1
        return px / data_y.size
    
    
    def get_exp(self, fx):
        "expectation of fx"
        return np.mean(fx, axis=1)
    
    def get_conditional_exp(self, x=None, y=None, f=None):
        "calculate conditional expectation of fx"
        if x is None:
            x, y, f = self.images, self.labels, self.f

        ce_f = np.zeros((self.n_label, f.shape[1]))
        for i in range(self.n_label):
            x_i = x[np.where(y==i)]
            fx_i = self.model_f(Variable(x_i).to(self.device)).cpu().detach().numpy() - f.mean(0)
            ce_f[i] = fx_i.mean(axis=0)
        
        return ce_f

    def normalize(self, f):
        e_f = f.mean(axis=0)
        n_f = f - e_f
        return n_f

    def load_for_id(self, id):

        try:
            self.read_from_load(id)
            self.model_f, self.model_g = loading.load_model(path = self.model_path, id = id)
        except:
            self.load(id)

    def get_g(self):

        
        g_y = np.array([self.g[torch.where(self.labels == i)][0] for i in range(self.n_label)])
        
        g_rand = np.random.random(g_y.shape)

        return g_rand, g_y

    
    def get_accuracy(self, gc):
        "classification accuracy with different gy"
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

    def acc(self):
        "output accuracy dict for all g and target tasks"
        acc_all = {}
        for id in self.t_id:
            self.load_for_id(id)
            acc = [self.get_accuracy(g) for g in self.get_g()]
            acc_list = {
                "g_rand": acc[0],
                "g_net": acc[1],
            }
            acc_all[id] = acc_list     
        return acc_all
    
    def logging(self):
        # logging
        pass

    def save(self, obj, filename:str) -> None:
        '''
        save object as file with given filename
        only npy files supported
        '''
        try:
            np.save(f"{self.save_path}{os.path.basename(sys.argv[0]).strip('.py')}_{filename}_{time.strftime('%m%d', time.localtime())}.npy", obj)
        except:
            raise TypeError("unexpected object type")


if __name__ == '__main__':

    # DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    # MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/fg_train/weight/"
    # SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/fg_train/results/"
    

    N_TASK = 21
    TASK_LIST = range(N_TASK)

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def run(cfg : DictConfig)->None:    
        cal = fg(cfg, TASK_LIST)
        acc = cal.acc()
        # json.dumps(acc, indent=4, sort_keys=True)
        print(acc)
        cal.save(acc, "accuracy_dict")


    run()


    # np.load("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/fg_train/results/accuracy_dict_0410.npy", allow_pickle=True).item()
    # cfg = yaml.load(open("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/conf/config.yaml","r"), Loader = yaml.Loader)  
 
    # acc = np.zeros((N_TASK,3))
    # for i in range(N_TASK):
    #     cal = fg(cfg=cfg, i)
        
    #     g_r, g, g_hat = cal.get_g()
    #     rand = cal.get_accuracy(gc=g_r)
    #     org = cal.get_accuracy(gc=g)
    #     hat = cal.get_accuracy(gc=g_hat)
    #     print("-------------task_id:{:d}-------------".format(i))
    #     print("random:{:.1%}\noriginal:{:.1%}\ncalculated:{:.1%}\n".format(rand, org, hat))
    #     acc[i] = rand, org, hat
    # np.savetxt(SAVE_PATH+'vanilla_acc_table_'+time.strftime("%m%d", time.localtime())+'.npy', acc)