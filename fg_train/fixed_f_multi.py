import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
from fixed_f import fg
from OTCE import OTCE
import json
import sys
sys.path.append("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp")
import trainer.loading as loading
from fg_train.fixed_f_transfer import transfer_fg

class multi_fg(transfer_fg):
    def __init__(self, cfg, t_ids, s_ids, alpha):
        super(multi_fg, self).__init__(cfg = cfg, t_ids=t_ids, s_ids=s_ids, alpha=alpha)

    
    def get_g(self):
        # expectation and normalization of f and g
        n_f = self.normalize(self.f)
        # n_g = self.normalize(self.g)

        gamma_f = n_f.T.dot(n_f) / n_f.shape[0]

        ce_f = self.get_conditional_exp()
        
        ce_f_s = np.array([self.get_conditional_exp(self.s_x_list[i], self.s_y_list[i], self.s_f_list[i]) for i in range(self.n_source)])

        g_y_hat = np.linalg.inv(gamma_f).dot(((1-np.sum(self.alpha)) * ce_f + ce_f_s.transpose(1,2,0).dot(self.alpha)).T).T        
        
        g_rand = np.random.random(g_y_hat.shape)

        return g_rand, g_y_hat