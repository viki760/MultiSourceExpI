import numpy as np
import sys
import copy
from tqdm import tqdm
sys.path.append("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp")
import util.loading as loading
from fg_train.fixed_f_transfer import transfer_fg
# from fg_trai  n.fixed_f import fg
# from metrics.OTCE import OTCE
from metrics.CDNV import CDNV, get_transfer_feature


def maximize_f(f, alpha, lr=0.001, epsilon = 0.0001, num_iters = 100):
    score = np.zeros(num_iters)
    for i in tqdm(range(num_iters)):
        grad = np.zeros_like(alpha)
        for j in range(len(alpha)):
            a_plus, a_minus = np.copy(alpha), np.copy(alpha)
            a_plus[j] += epsilon
            a_minus[j] -= epsilon
            grad[j] = (f(a_plus)-f(a_minus))/(2*epsilon)
        alpha += lr*grad
        alpha[alpha < 0] = 0
        alpha = alpha / alpha.sum() if alpha.sum() > 1 else alpha
        print(alpha)
        score[i] = f(alpha)

    return alpha, score

def update_alpha(t_id, s_id, lr, include_target=True):
    #! 这里 include_target 为 False 时的 alpha 和为 1 约束条件没有解决
    n_s_tasks = len(s_id)
    alpha = np.ones(n_s_tasks) / (n_s_tasks + 1)
    label, feature = get_transfer_feature(t_id, s_id, for_optim=True)
    f = lambda a: CDNV(id_t = t_id, id_s = copy.deepcopy(s_id), alpha = a, include_target = include_target, for_optim=True, features=feature, labels=label)
    alpha_opt, score_curve = maximize_f(f, alpha, lr=lr)
    return alpha_opt, score_curve

if __name__ == '__main__':
    lr_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    save_path = '/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/alpha/'
    for lr in lr_list:
        a,s = update_alpha(0, list(range(1,21)), lr=lr, include_target=True)
        np.savetxt(save_path+'cdnv_grad_lr='+str(lr)+'_alpha.npy', a)
        np.savetxt(save_path+'cdnv_grad_lr='+str(lr)+'_scorelist.npy', s)
    # from matplotlib import pyplot as plt
    # plt.plot(s)