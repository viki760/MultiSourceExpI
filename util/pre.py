import numpy as np
import pandas as pd

DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data/cifar-100-python/"
SAVE_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/data/"

# read dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# reconstruct img into 32(w)*32(h)*3(channel)
def reconstruct(img):
    r = img[:1024].reshape(32,32,1)
    g = img[1024:2048].reshape(32,32,1)
    b = img[2048:].reshape(32,32,1)
    img = np.concatenate((r,g,b),axis = 2) / 255
    return img

# derive 2-class task with given categories i,j
# k indicating fine(0)/coarse(1) label
def make_task(i, j, iscoarse = False, istest = False):

    key = 'test' if istest else 'train'
    label = np.load(SAVE_PATH + 'yc_'+key+'.npy') if iscoarse else np.load(SAVE_PATH + 'yf_'+key+'.npy')
    x = np.load(SAVE_PATH + 'x_'+key+'.npy')

    label = np.where(label==i, -1, 0) + np.where(label==j, 1, 0)
    print(np.where(label!=0))
    x_tr, y_tr = x[np.where(label != 0)], label[np.where(label != 0)]
    return x_tr, np.array(y_tr > 0, dtype = int)

def make(pairs, dir, target_id = 0, shot = 10):
    for i in range(len(pairs)):
        set = pairs[i]
        x, y = make_task(set[0]-1, set[1]-1, bool(set[2]))
        if i == target_id:
            sample = np.random.choice(len(x), shot, replace=False)
            x, y = x[sample], y[sample]
        np.save(SAVE_PATH + str(dir)+ "/x"+str(i)+"_train.npy", x)
        np.save(SAVE_PATH + str(dir)+ "/y"+str(i)+"_train.npy", y)

        xt, yt = make_task(set[0]-1, set[1]-1, bool(set[2]), istest=True)
        np.save(SAVE_PATH + str(dir)+ "/x"+str(i)+"_test.npy", xt)
        np.save(SAVE_PATH + str(dir)+ "/y"+str(i)+"_test.npy", yt)

def task_info(pairs, dir):
    for i in range(len(pairs)):
        pass


if __name__ == "__main__":

    ## process train
    # get img
    train = unpickle(DATA_PATH+'train')
    train_X_init = train[b'data']
    train_Y1, train_Y2 = train[b'fine_labels'], train[b'coarse_labels']
    train_X_img = np.array([reconstruct(x) for x in train_X_init])

    # get label matching dict
    meta = unpickle(DATA_PATH+'meta')
    meta[b'fine_label_names']
    label_dict = {meta[b'fine_label_names'][i].decode('utf-8'):i for i in range(len(meta[b'fine_label_names']))}

    # save reconstructed data
    np.save(SAVE_PATH + 'x_train.npy', train_X_img)
    np.save(SAVE_PATH + 'yf_train.npy', np.array(train_Y1))
    np.save(SAVE_PATH + 'yc_train.npy', np.array(train_Y2))

    
    
    ## process test
    # get img
    test = unpickle(DATA_PATH+'test')
    test_X_init = test[b'data']
    test_Y1, test_Y2 = test[b'fine_labels'], test[b'coarse_labels']
    test_X_img = np.array([reconstruct(x) for x in test_X_init])

    # save reconstructed data
    np.save(SAVE_PATH + 'x_test.npy', test_X_img)
    np.save(SAVE_PATH + 'yf_test.npy', np.array(test_Y1))
    np.save(SAVE_PATH + 'yc_test.npy', np.array(test_Y2))


    pair = [[4, 22, 0], [1, 58, 0], [6, 21, 0], [9, 14, 0], [10, 17, 0], [98, 35, 0], [8, 16, 0], [7, 15, 0], 
                [3, 12, 0], [31, 96, 0], [13, 69, 0], [48, 53, 0], [77, 38, 0], [50, 72, 0], [28, 29, 0], 
                [49, 50, 0], [36, 37, 0], [61, 62, 0], [4, 44, 0], [4, 70, 0], [4, 24, 0]]
    # 0 bear, chimpanzee  
    # 1 apple, pear   
    # 2 bed, chair  
    # 3 bicycle, bus 
    # 4 bottle, can    
    # 5 wolf, fox
    # 6 beetle, camel   
    # 7 bee, butterfly  
    # 8 baby, boy   
    # 9 dolphin, whale
    # 10 bridge, road
    # 11 maple_tree, oak_tree
    # 12 skyscraper, house
    # 13 mountain, sea
    # 14 crocodile, cup
    # 15 motorcycle, mountain
    # 16 girl, hamster
    # 17 plain, plate
    # 18 bear, lion
    # 19 bear, rocket
    # 20 bear, cloud

    make(pair, dir = 'data_set_1')



