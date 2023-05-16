from torch.autograd import Variable
import torch
import numpy as np
import cvxpy as cvx
import loading
from fixed_f_single import single_fg
from fixed_f_vanilla import vanilla_fg
import sys
sys.path.append(
    "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/")


# def load(type_, args):
#     if type_ == "single_fg":
#                 self.model_f_tr, self.model_g_tr = loading.load_model()
#         # self.model_f_tr.load_state_dict(torch.load(model_path+'f_task_t='+str(t_id)+'_s='+str(s_id)+'_alpha='+str(alpha)+'.pth', map_location=self.device))
#         self.model_f_tr.load_state_dict(torch.load(
#             f"{model_path}_f_task_t= {}+str(t_id)+'_s='+str(s_id)+'_alpha='+str(alpha)+'.pth'", map_location=self.device
#             ))

#         self.model_g_tr.load_state_dict(torch.load(model_path+'g_task_t='+str(t_id)+'_s='+str(s_id)+'_alpha='+str(alpha)+'.pth', map_location=self.device))
#     elif type_ == ""


# @dataclass
# class SimpleTypes:
#     num: int = 10
#     pi: float = 3.1415
#     is_awesome: bool = True
#     height: Height = Height.SHORT
#     description: str = "text"
#     data: bytes = b"bin_data"
#     path: pathlib.Path = pathlib.Path("hello.txt")

# args = SimpleTypes()

# args.path = xxx
# args["path"] =


class alpha_vanilla(single_fg):
    '''
    Original version of calculating alpha via convex optimization
    '''

    def __init__(self, model_path, data_path, task_list=None) -> None:
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
            f = model_f_i(Variable(images_i).to(
                self.device)).cpu().detach().numpy()
            f_list.append(f)

        return f_list

    def get_conditional_exp_ft(self, fx, x, y):
        #!
        ce_f_ft = np.zeros((self.n_label, fx.shape[1], fx.shape[1]))
        for i in range(self.n_label):
            x_i = x[np.where(y == i)]
            fx_i = self.model_f(Variable(x_i).to(
                self.device)).cpu().detach().numpy() - fx.mean(0)
            ce_f_ft[i] = fx_i.mean(axis=0)

        return ce_f_ft

    def get_V(args, id):
        f_id = self.f_list[args.id]
        data_id = self.data_list[id]
        x_id, y_id = next(iter(data_id))

        # expectation and normalization of f and g
        e_f_id = f_id.mean(0)
        n_f_id = f_id - e_f_id

        gamma_f_id = n_f_id.T.dot(n_f_id) / n_f_id.shape[0]

        ce_f_id = self. get_conditional_exp(f_id, x_id, y_id)
        #!
        ce_f_ft_id = self. get_conditional_exp_ft(f_id.dot(f_id.T), x_id, y_id)

        py = self.get_distribution_y(y_id)

        V = 0
        for i in range(len(py)):
            V += py[i] / self.py_t[i] * \
                np.trace(np.linalg.inv(gamma_f_id).dot(ce_f_ft_id[i].T).T)
            #!
            V -= py[i]*2 / self.py_t[i] * \
                ((gamma_f_id)**(-0.5).dot(ce_f_id[i].T))**2
        return V

    def get_h(self, id, l):
        ce_f_id = self.get_conditional_exp(self.f_list[id])
        _, y_id = next(iter(self.data_list[id]))
        h = self.get_distribution_y(y_id)[l] * ce_f_id[l].T
        return h

    def get_A(self):
        A = np.zeros([self.dim, self.dim])
        self.f_list = self.load_source_feature()
        self.data_list = [loading.load_data(
            path=self.data_path, id=i, batch_size=None) for i in range(self.dim)]

        torch.save(
            {
                "features": self.f_list,
                "targets":}
        )

        data = torch.load()

        data["features"]

        _, y_t = next(iter(self.data_list[0]))
        self.py_t = self.get_distribution_y(y_t)

        for i in range(self.dim):
            A[i][i] = self.get_V(i) / len(self.data_list[i])

        for i in range(self.dim):
            for j in range(self.dim):
                dij = 0
                for l_y in range(self.n_label):
                    dij += (self.get_h(0, l_y) - self.get_h(i, l_y)
                            ).dot(self.get_h(0, l_y) - self.get_h(j, l_y)).T / self.py_t[l_y]
                A[i][j] += np.trace(np.linalg.inv(self.gamma_f).dot(dij))
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
        alphav = cvx.Variable(self.dim)
        # adding regularization
        if type == 'l1':
            # regularize with l1-norm
            obj = cvx.Minimize(cvx.quad_form(alphav, A)+cvx.norm(alphav, 1))

        elif type == 'l2':
            # regularize with l2-norm
            obj = cvx.Minimize(cvx.quad_form(alphav, A)+cvx.norm(alphav, 2))

        elif type == None:
            obj = cvx.Minimize(cvx.quad_form(alphav, A))

        constraint = [np.ones([1, self.dim]) @ alphav == 1.,
                      np.eye(self.dim) @ alphav >= np.zeros(self.dim)]
        prob = cvx.Problem(obj, constraint)
        prob.solve()

        return alphav.value


if __name__ == "__main__":

    DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
    MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/weight/"
    SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/formula_test/results/"
    get_alpha = alpha_vanilla(model_path=MODEL_PATH, data_path=DATA_PATH)

    A = get_alpha.get_A()

    alpha_0 = get_alpha.optimize(A)
    alpha_1 = get_alpha.optimize(A, type='l1')
    alpha_2 = get_alpha.optimize(A, type='l2')
