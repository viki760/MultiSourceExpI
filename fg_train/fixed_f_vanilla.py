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
import util.loading as loading
from fixed_f import fg


def load():
    pass

class vanilla_fg(fg):
    '''
    calculation with fixed feature extractor w/o transfer
    '''
    def __init__(self, cfg, t_id=0):
        super(vanilla_fg, self).__init__(cfg = cfg, t_id=t_id)
        
    def get_g(self):
        # expectation and normalization of f and g
        n_f = self.normalize(self.f)
        # n_g = self.normalize(self.g)

        gamma_f = n_f.T.dot(n_f) / n_f.shape[0]
        ce_f = self. get_conditional_exp()
        g_y_hat = np.linalg.inv(gamma_f).dot(ce_f.T).T
        
        g_y = np.array([self.g[torch.where(self.labels == i)][0] for i in range(self.n_label)])
        
        g_rand = np.random.random(g_y.shape)

        return g_rand, g_y, g_y_hat


    def acc(self):
        acc_all = {}
        for id in self.t_id:
            self.load_for_id(id)
            acc = [self.get_accuracy(g) for g in self.get_g()]
            acc_list = {
                "g_rand": acc[0],
                "g_net": acc[1],
                "g_cal": acc[2],
            }
            acc_all[id] = acc_list     
        return acc_all

if __name__ == '__main__':

    import hydra 
    from omegaconf import DictConfig

    N_TASK = 21
    TASK_LIST = range(N_TASK)

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def run(cfg : DictConfig)->None:    
        cal = vanilla_fg(cfg, TASK_LIST)
        acc = cal.acc()
        print(acc)
        cal.save(acc, "accuracy_dict")
    
    run()

    # DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    # MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/weight/"
    # SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/results/"
    # N_TASK = 21

    # acc = np.zeros((N_TASK,3))
    # for i in range(N_TASK):
    #     cal = vanilla_fg(DATA_PATH, MODEL_PATH,i)
        
    #     g_r, g, g_hat = cal.get_g()
    #     rand = cal.get_accuracy(gc=g_r)
    #     org = cal.get_accuracy(gc=g)
    #     hat = cal.get_accuracy(gc=g_hat)
    #     print("-------------task_id:{:d}-------------".format(i))
    #     print("random:{:.1%}\noriginal:{:.1%}\ncalculated:{:.1%}\n".format(rand, org, hat))
    #     acc[i] = rand, org, hat
    # np.savetxt(SAVE_PATH+'vanilla_acc_table_'+time.strftime("%m%d", time.localtime())+'.npy', acc)
    
        