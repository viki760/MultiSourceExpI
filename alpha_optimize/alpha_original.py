import sys 
sys.path.append("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/") 

from fixed_f_vanilla import vanilla_fg
from fixed_f_single import single_fg
import loading
import cvxpy as cvx
import numpy as np
import torch
from torch.autograd import Variable

class alpha_vanilla():
    '''
    Original version of calculating alpha via convex optimization
    '''

    def __init__(self, model_path, data_path, task_list = None) -> None:
        '''
        task_list is the list of task ids where the first id is the target / the others are the sources
        '''

        self.task_list = list(range(21)) if task_list == None else task_list
        self.dim = len(task_list)
        self.model_path = model_path
        self.data_path = data_path
        self.feature_list = self.load_feature(task_list)
        self.data = loading.load_data(task_list[0])
        self.model_f, self.model_g = loading.load_model(task_list[0])


    def load_source_feature(self):
        
        f_list = []

        # images, labels = next(iter(self.data))
        # # take the first batch as input data
        # labels_one_hot = torch.zeros(len(labels), self.n_label).scatter_(1, labels.view(-1,1), 1)
        # f_t = self.model_f(Variable(images).to(self.device)).cpu().detach().numpy()
        

        for id in self.task_list[1:]:
            model_f_i, _ = loading.load_model(id)
            data_i = loading.load_data(id)
            images_i, _ = next(iter(data_i))
            f = model_f_i(Variable(images_i).to(self.device)).cpu().detach().numpy()
            f_list.append(f)
        
        return f_list





    def get_A(self):
        A = np.zeros([self.dim, self.dim])
        source_f = self.load_source_feature()
    
        return A


    # def regularize(self, A, type):
    #     "adding regularization"
    #     if type == 'l1':
    #         # regularize with l1-norm
    #         pass
    #     elif type == 'l2':
    #         # regularize with l2-norm
    #         pass
    #     elif type == None:
    #         pass

    #     return A

    def optimize(self, A, type=None):
        "solution to alpha convex optimization given matrix A"
        alphav=cvx.Variable(self.dim)
        # adding regularization
        if type == 'l1':
            # regularize with l1-norm
            obj = cvx.Minimize(cvx.quad_form(alphav, A)+cvx.norm(alphav, 1))

        elif type == 'l2':
            # regularize with l2-norm
            obj = cvx.Minimize(cvx.quad_form(alphav, A)+cvx.norm(alphav, 2))
            
        elif type == None:
            obj = cvx.Minimize(cvx.quad_form(alphav, A))
        
        constraint = [np.ones([1,self.dim]) @ alphav == 1., np.eye(self.dim) @ alphav >= np.zeros(self.dim)]
        prob = cvx.Problem(obj, constraint)
        prob.solve() 

        return alphav.value

if __name__ == "__main__":
    
    DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/weight/"
    SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/results/"
    get_alpha = alpha_vanilla(model_path = MODEL_PATH, data_path = DATA_PATH)

    A = get_alpha.get_A()

    alpha_0 = get_alpha.optimize(A)
    alpha_1 = get_alpha.optimize(A, type='l1')
    alpha_2 = get_alpha.optimize(A, type='l2')

