# file: display.py
# visualizing results

import numpy as np
from matplotlib import pyplot as plt


def read_accuracy(t_id, res_path, type = 'fixed_f_transfer', filedates = ['0414', '0412']):

    data = []
    for i in range(21):
        data_i = {}
        for date in filedates:
            data_i.update(np.load(f"{res_path}{type}_accuracy_dict_source={i}__{date}.npy", allow_pickle=True).item()[t_id])
        data.append(data_i)

    g_rand, g_cal, g_net, W, ce, finetune = [], [], [], [], [], []
    for d in data:
        g_rand.append(d['g_rand'])
        g_cal.append(d['g_cal'])
        g_net.append(d['g_net'])
        W.append(d['otce'][0][0])
        ce.append(d['otce'][0][1])
        finetune.append(d['finetune'][0])
    
    return g_rand, g_cal, g_net, W, ce, finetune

if __name__ == '__main__':

    t_id = 0
    res_path = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/fg_train/results/"
    type = 'fixed_f_transfer'


    g_rand, g_cal, g_net, W, ce, finetune = read_accuracy(t_id, res_path)
    plt.plot(g_rand,'.',color='black')
    plt.plot(g_net,'^', color='lightgreen')
    plt.plot(finetune, 'v', color='lightblue')
    plt.plot(g_cal,'r<')
    plt.plot(ce, 'o', color='orange')
    plt.xlabel('source task id')
    plt.xticks(np.arange(21))
    plt.ylim(-0.1,1)
    plt.legend(['g_rand','g_net','finetune','g_cal','ce'])
    plt.title(f'Accuracy Comparison (Target = {t_id})')
    plt.show()

    plt.savefig(f'{res_path}fig_acc_{type}_t={t_id}.png')

