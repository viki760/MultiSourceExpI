# calculate optimized g with fixed f
# calculate h score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
import loading
from fixed_f_vanilla import vanilla_fg


class single_fg(vanilla_fg):

    '''
    calculation with fixed feature extractor with single source transfer only
    '''

    def __init__(self, data_path, model_path, t_id, batch_size=None, s_id=0, alpha = 0.4):
        
        super(single_fg, self).__init__(data_path=data_path, model_path=model_path, t_id=t_id, batch_size=batch_size)   
        self.data_s = loading.load_data(path = data_path, id = s_id)

        self.model_f_tr = self.model_f.load_state_dict(torch.load(model_path+'f_task_t='+str(t_id)+'_s='+str(s_id)+'_alpha='+str(alpha)+'.pth', map_location=self.device))
        self.model_g_tr = self.model_g.load_state_dict(torch.load(model_path+'f_task_t='+str(t_id)+'_s='+str(s_id)+'_alpha='+str(alpha)+'.pth'+'.pth', map_location=self.device))

        self.alpha = alpha

    
    def s_tr_g_train(self):
        
        _, labels = next(iter(self.data))
        labels_one_hot = torch.zeros(len(labels), self.n_label).scatter_(1, labels.view(-1,1), 1)
        # f = self.model_f_tr(Variable(images).to(self.device)).cpu().detach().numpy()
        g = self.model_g_tr(Variable(labels_one_hot).to(self.device)).cpu().detach().numpy()
        

        g_y = np.array([g[torch.where(labels == i)][0] for i in range(labels.max()+1)])

        return g_y


    def s_tr_g_cal(self):

        images, labels = next(iter(self.data))
        # take the first batch as input data
        labels_one_hot = torch.zeros(len(labels), self.n_label).scatter_(1, labels.view(-1,1), 1)
        f = self.model_f(Variable(images).to(self.device)).cpu().detach().numpy()
        g = self.model_g(Variable(labels_one_hot).to(self.device)).cpu().detach().numpy()

        images_s, labels_s = next(iter(self.data_s))
        labels_one_hot_s = torch.zeros(len(labels_s), self.n_label).scatter_(1, labels_s.view(-1,1), 1)
        f_s = self.model_f(Variable(images_s).to(self.device)).cpu().detach().numpy()
        g_s = self.model_g(Variable(labels_one_hot_s).to(self.device)).cpu().detach().numpy()

        # g = (1-alpha) * g + alpha * g_s

        # expectation and normalization of f and g
        e_f = f.mean(0)
        n_f = f - e_f
        # e_g = g.mean(0)
        # n_g = g - e_g

        gamma_f = n_f.T.dot(n_f) / n_f.shape[0]

        ce_f = self. get_conditional_exp(f, images, labels)
        ce_f_s = self. get_conditional_exp(f_s, images_s, labels_s)
        g_y_hat = np.linalg.inv(gamma_f).dot(((1-self.alpha) * ce_f + self.alpha * ce_f_s).T).T        
        
        g_rand = np.random.random(g_y_hat.shape)

        return g_rand, g_y_hat



if __name__ == '__main__':
    import time

    DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/formula_test/weight/"
    SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/formula_test/results/"
    N_TASK = 21
    alpha = 0.4

    for t_id in range(21):

        acc = np.zeros((N_TASK,3))
        
        for id in range(21):
            cal = single_fg(DATA_PATH, MODEL_PATH, t_id=t_id, s_id=id, alpha=alpha)
            
            g = cal.s_tr_g_train()
            g_r, g_hat = cal.s_tr_g_cal()
            rand = cal.get_accuracy(gc=g_r)
            org = cal.get_accuracy(gc=g)
            hat = cal.get_accuracy(gc=g_hat)
            print("-------------source_task_id:{:d}-------------".format(id))
            print("random:{:.1%}\noriginal:{:.1%}\ncalculated:{:.1%}\n".format(rand, org, hat))
            acc[id] = rand, org, hat
            print("-------------end-------------")

        np.savetxt(SAVE_PATH+'single_acc_table_'+time.strftime("%m%d", time.localtime())+'_alpha='+str(alpha)+'_t='+str(t_id)+'.npy', acc)