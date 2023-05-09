import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
import json
import sys
sys.path.append("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp")
import util.loading as loading
from fg_train.fixed_f_transfer import transfer_fg
# from fg_train.fixed_f import fg
# from metrics.OTCE import OTCE
from metrics.H_score import Hscore

class multi_fg(transfer_fg):
    def __init__(self, cfg, t_ids, s_ids, alpha):
        super(multi_fg, self).__init__(cfg = cfg, t_ids=t_ids, s_ids=s_ids, alpha=alpha)
        # self.alpha_given = alpha
        

    def empirical(self):
        pass

    def finetune(self):
        pass

    def get_Hscore_multi(self, t_id):
        return Hscore(id_t = t_id, id_s = self.s_ids, alpha = self.alpha, include_target = True)
    
    def get_g(self, alpha_type='given'):

        if alpha_type == 'rand':
            self.alpha = self.rand_alpha()

        # expectation and normalization of f and g
        n_f = self.normalize(self.f)
        # n_g = self.normalize(self.g)

        gamma_f = n_f.T.dot(n_f) / n_f.shape[0]

        ce_f = self.get_conditional_exp()
        
        ce_f_s = np.array([self.get_conditional_exp(self.s_x_list[i], self.s_y_list[i], self.s_f_list[i]) for i in range(self.n_source)])

        g_y_hat = np.linalg.inv(gamma_f).dot(((1-np.sum(self.alpha)) * ce_f + ce_f_s.transpose(1,2,0).dot(self.alpha)).T).T        
        
        g_rand = np.random.random(g_y_hat.shape)

        return g_rand, g_y_hat
    

    def grid_alpha(self):
        pass

    def optimize_alpha(self):
        pass

    def rand_alpha(self):
        a = np.random.random(self.n_source + 1)
        a /= a.sum()
        return a[1:]
    



if __name__ == '__main__':
    import time
    import hydra
    from omegaconf import DictConfig

    N_TASK = 21
    TASK_LIST = range(N_TASK)

    s_l = [i for i in range(1, N_TASK)]
    alpha = np.ones(20) / 40
    

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def run(cfg : DictConfig)->None: 
        
        cal = multi_fg(cfg, t_ids=0, s_ids=s_l, alpha=alpha)

        acc = cal.acc(empirical=False, finetune = False)
        print(acc)
        cal.save(acc, f"accuracy_dict_source={s_l}_")

    run()