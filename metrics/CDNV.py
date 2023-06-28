'''
calculate h score
'''

from tqdm import tqdm
import numpy as np
import gc
import sys
import torch
sys.path.append("/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp")
import util.loading as loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
# DATA_PATH = r"D:\task\research\codes\MultiSource\wsl\2\multi-source\data_set_2\\"
PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/"
MODEL_PATH = PATH + "fg_train/weight/"


def getCov(X):
    X_mean = X-np.mean(X, axis=0, keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1)
    return cov


def getDiffNN(f, Z):
    #Z=np.argmax(Z, axis=1)
    Covf = getCov(f)
    alphabetZ = list(set(Z))
    g = np.zeros_like(f)
    for z in alphabetZ:
        l = Z == z
        fl = f[Z == z, :]
        # conditional expectation
        Ef_z = np.mean(fl, axis=0)
        g[Z == z] = Ef_z

    Covg = getCov(g)
    dif = np.trace(np.dot(np.linalg.pinv(Covf, rcond=1e-15), Covg))
    return dif


def getCDNV(f, Z):
    #Z=np.argmax(Z, axis=1)

    alphabetZ = list(set(Z))
    varf = 0
    Ef = np.zeros([len(alphabetZ), len(f[0])])

    get_norm2 = lambda f, ef: ((f - ef)*(f - ef)).sum(axis = 1)
    for z in alphabetZ:
        f_z = f[Z == z, :]
        Ef_z = np.mean(f_z, axis=0)

        varf_z = np.mean(get_norm2(f_z, Ef_z) , axis=0)

        varf += varf_z
        Ef[z] = Ef_z

    if len(alphabetZ) == 2:
        class_dist = ((Ef[0] - Ef[1]) * (Ef[0] - Ef[1])).sum()
    else:
        class_dist = np.var(Ef, axis=0).sum()
    
    return varf / 2 / class_dist

def get_transfer_feature(id_t, id_s, for_optim=False):
    
    #! only for with target
    if for_optim == True:
        id_s.append(id_t)
    data = loading.load_data(path = DATA_PATH, id = id_t, batch_size = 15, t = 0)
    images, labels = next(iter(data))
    f = []
    for i in range(len(id_s)):
        model_f, _ = loading.load_model(path = MODEL_PATH, id = id_s[i])
        f_i = model_f(images.to(device)).cpu().detach().numpy()
        f.append(f_i)
    features = np.array(f)
    # # feature = f.sum(axis = 0)

        
    return labels, features

def CDNV(id_t, id_s, alpha=1.0, include_target=False, for_optim=False, features=None, labels=None):
    '''
    given target and source list, return h score
    alpha: feature weights (should be an array)
    include_target: whether to insert target feature extractor in linear combination
    for_optim: if true, store features in 
    '''
    
    
    # data_test = loading.load_data(path = data_path, id = id_t, batch_size = 100, t = 1)

    # in case of single transferability
    id_s = [id_s] if isinstance(id_s, int) else id_s
    alpha = np.array([alpha]) if isinstance(alpha, float) else alpha

    if include_target == True:
        id_s.append(id_t)
        alpha = np.append(alpha, 1 - alpha.sum())
    else:
        alpha = alpha / alpha.sum()

    if for_optim == False:
        labels, features = get_transfer_feature(id_t, id_s)

    feature = np.array([alpha[i]* features[i] for i in range(len(alpha))]).sum(axis=0)

    cdnv = getCDNV(feature, labels.cpu().detach().numpy())
    gc.collect()

    return cdnv




if __name__ == "__main__":
    cdnv = CDNV(0, [1,2], np.array([0.2, 0.6]), True)
    print(cdnv)
