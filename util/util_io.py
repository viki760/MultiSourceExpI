import numpy as np
import scipy.io as sio
from scipy import stats
import pickle
import sys, getopt,os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder,normalize
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize

## read Office+Caltech benchmark data
def readOfficeCaltech(src, oneHotLabel=False):
    path = './data/' + src + '_SURF_L10.mat'
    data = sio.loadmat(path)
    fts = data['fts']
    fts = np.divide(fts, np.sum(fts, axis=1, keepdims=True)) 
    X = stats.zscore(fts, axis=0) ## preprocessing of Geodesic Flow Kernel
    X = X.astype('float32')
    Y = data['labels'].ravel()
    if oneHotLabel:
        Y = to_categorical(Y, num_classes=10)
    return X, Y

## read Bookmarks data
def readBookmarks():
    path = './data/bookmarksXY.mat'
    data = sio.loadmat(path)
    X = data['tb_bm100_X']
    Y = data['tb_bm100_Y']
    tags = data['top100Names']
    tagnames =[]
    for t in tags[0]:
        tagnames.append(t[0][4:])
    
    print('reading %d samples with %d dim feature and  %d dim labels' %\
           (len(X), X.shape[1],Y.shape[1]))
    return X,Y,np.array(tagnames)

## load data stored in a pickle binary file
def loadData(fname):
    print("reading from",fname,"...")
    with open(fname,'rb') as f:
        return pickle.load(f)

## dump data to in a pickle binary file
def dumpData(var,fname):
    with open(fname,'wb') as f:
        pickle.dump(var,f)
    print("data written to",fname)
""" 
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

def dumpList(var,fname):
    with open(fname,'wb') as f:
        d = dict()
        for v in var:
            d[namestr(v,globals())] = v
        pickle.dump(d,f)
    print("Dictionary of %d items has been written to %s" % (len(d),fname))
"""

def printHelp():
    print("run_ace.py -s <src_prefix> -t <target_prefix> -a <tag>")
    sys.exit(2)

    
def parseArg(argv): 
    if len(argv)<3:
        printHelp() 
    src_prefix = ''
    tgt_prefix = ''
    tag=''
    import sys,getopt
    try:
        opts,args = getopt.getopt(argv,"hs:t:a:",["src_prefix=","tgt_prefix=",
                                                "tag="])
    except getopt.GetoptError:
        printHelp() 
    for opt,arg in opts:
        if opt=="-h":
            printHelp() 
        elif opt in ("-s","--src_prefix"):
            src_prefix = arg
        elif opt in ("-t","--tgt_prefix"):
            tgt_prefix = arg
        elif opt in ("-a","--tag"):
            tag = arg
    print("Transfer learning from",src_prefix,"to",tgt_prefix,
          "with tag name",tag)

    return src_prefix, tgt_prefix,tag


## copied from Keras library
## https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y[y==10] = 0
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def split_samples(X, Y, split=0.9,train_indices = None):
    if train_indices is None:
        train_indices = np.random.choice(len(X), round(len(X)*split), replace=False)
    test_indices = np.array(list(set(range(len(X))) - set(train_indices)))    
    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    print( "Split ratio:",split ,len(train_indices) ,"training samples,",
           len(test_indices),"testing samples")
    return X_train, Y_train, X_test, Y_test, train_indices


def onehot(data,transform=True):
    if transform:
        le = LabelEncoder()
        le.fit(list(set(data)))
        data = le.transform(data)
    alp = list(set(data))
    seed = np.diag(np.ones(len(alp)))
    thedict = {alp[j]:list(seed[j]) for j in range(len(alp))} 
    result = np.zeros([len(data),len(alp)])
    for i in alp:
        result[data==i,:] = thedict[i]
    return result

def genRandomY(X,ny):
    
    Xhot = onehot(X,transform=False).transpose()
    nx = Xhot.shape[0]
    nsamples = Xhot.shape[1]

    Pyx = normalize(np.abs(np.random.randn(ny,nx)) ,norm='l1',axis=0)
    
    Py = np.dot(Pyx, Xhot)

    Y = np.zeros(nsamples)
    for i in range(nsamples):
        Y[i] = np.random.choice(ny, 1, p = Py[:,i] )
    
    return Y,Pyx

def genMarkovSamples(nsamples, nx=20,ny=15,nz=5):    
    # generate X- Y -Z data according to random stochastic 
    # matrix P(y|x) , and a random mapping F(y) =z
    # F(y) is created from a random stochastic matrix P(z|y)
    # i.e. F(y) = z = argmax_z(P(z|y)) 
    X = np.random.randint(0,nx,size=nsamples);
    Xhot = onehot(X,transform=False).transpose()
    # generate column stochastic matrices
    Pyx = normalize(np.abs(np.random.randn(ny,nx)) ,norm='l1',axis=0)
    print(Pyx)
    Pzy = normalize(np.abs(np.random.randn(nz,ny)), norm='l1',axis = 0)
    F = np.argmax(Pzy,axis=0)
    FF = np.zeros((nz,ny))
    FF[F,range(ny)]=1
    print(FF)
    # generate Y
    Py = np.dot(Pyx, Xhot)

    Y = np.zeros(nsamples)
    for i in range(nsamples):
        Y[i] = np.random.choice(range(0,ny), 1, p = Py[:,i] )

    Yhot = onehot(Y,transform=False).transpose()

    # generate Z
    Pz = np.dot( Pzy ,Yhot)
    Z = np.argmax(Pz, axis=0) 
    return X,Y,Z 
    

def genMarkovSamplesZRange(nsamples, nx=20,ny=15,nz_list=[5]):
    X = np.random.randint(0,nx,size=nsamples);
    Xhot = onehot(X,transform=False).transpose()
    # generate column stochastic matrix P(Y|X)
    Pyx = normalize(np.abs(np.random.randn(ny,nx)) ,norm='l1',axis=0)
    print(Pyx)

    # generate Y
    Py = np.dot(Pyx, Xhot)
    Y = np.zeros(nsamples)
    for i in range(nsamples):
        Y[i] = np.random.choice(range(0,ny), 1, p = Py[:,i], replace= False)
    Yhot = onehot(Y,transform=False).transpose()

    # generate Zs
    ZList=np.zeros( (len(nz_list), nsamples));
    for i,nz in enumerate(nz_list):
        F =np.zeros((nz,ny))
        F[np.mod(range(ny),nz), range(ny)]  = 1
        Pz = np.dot( F ,Yhot)
        ZList[i,:] = np.argmax(Pz, axis=0)    
    return X,Y,ZList 

def genMarkovSamplesFPattern(nsamples, nx,ny,nz):
    # generate X
    X = np.random.randint(0,nx,size=nsamples);
    Xhot = onehot(X,transform=False).transpose()
    # generate Y
    Pyx = normalize(np.abs(np.random.randn(ny,nx)) ,norm='l1',axis=0)
    print(Pyx)
    
    Py = np.dot(Pyx, Xhot)
    Y = np.zeros(nsamples)
    for i in range(nsamples):
        Y[i] = np.random.choice(range(0,ny), 1, p = Py[:,i], replace= False)
    Yhot = onehot(Y,transform=False).transpose()
    # generate Z
    cutoff= ny-nz+1 
    
    nPtn =nz-1
    ZList=np.zeros( (nPtn, nsamples));
    FyzList=[]
    for k in range(1,nPtn+1):
        F =np.zeros((nz,ny))        
        # k: the number of unique values that take majority
        blocksize = round(cutoff/k)
        for j in range(k):            
            F[j,j*blocksize:min(cutoff+1,(j+1)*blocksize)] = 1
        
        nOthers = ny -blocksize*k #cutoff
        drange =np.array([i for i in  range(nOthers)],dtype=int)

        F[np.mod(drange,nz-k)+k, drange+blocksize*k]  = 1
        FyzList.append(F)
        print("F: k=",k,"=====================")
        print(F)
        Pz = np.dot( F ,Yhot)
        ZList[k-1,:] = np.argmax(Pz, axis=0)  
    return X,Y,ZList,(Pyx,FyzList)
        
def makeTimestampDirectory():
    timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outDir="output/"+timestamp
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        return outDir
    
def generateSeparableData(nx,ny,nsamples):
    V = np.random.rand(ny, nx)
    #V = np.random.choice(np.delete(np.arange(9), 0), (ny,nx)).astype(float)
    p_YCondX = normalize(V ,norm='l1',axis=0)
    index = np.argmax(p_YCondX,axis=0)
    for i in range(nx):
        p_YCondX[index[i], i] += 0.5#np.divide(1,ny)
        
    p_YCondX = normalize(p_YCondX ,norm='l1',axis=0)
    X, Y, Px = generateSamples(p_YCondX, nx, ny, nsamples)
    return X, Y, Px, p_YCondX

def generateSamples(p_YCondX, nx, ny, nsamples):
    X = np.random.randint(0,nx,size=nsamples)
    samples = np.zeros((nsamples,2))
    Px = np.zeros((nx,))
    samples[:,0] = X
    for i in range(nx):
        y = np.random.choice(ny, size=np.sum(X==i), p=p_YCondX[:,i])
        samples[X==i,1] = y
        Px[i] = np.divide(np.sum(X==i), nsamples)       
    return samples[:, 0], samples[:, 1], Px

def generateSamplesCond(p_YCondX, X):
    nsamples = len(X)
    nx = p_YCondX.shape[1]
    ny = p_YCondX.shape[0]
    print('ny:',ny)
    Y = np.zeros_like(X) #not one-hot!
    '''
    for i in range(nsamples):
        x = X[i]
        Y[i] = np.random.choice(ny, size = 1, p = p_YCondX[:,x])
    '''    
    for i in range(nx):
        Y[X == i] = np.random.choice(ny, size = sum(X==i), p = p_YCondX[:,i])
    return Y

def genSeparableRandomY(X,ny,delta=1):    
    Xhot = onehot(X,transform=False).transpose()
    nx = Xhot.shape[0]
    nsamples = Xhot.shape[1]

    Pyx = normalize(np.abs(np.random.randn(ny,nx)) ,norm='l1',axis=0)
    index = np.argmax(Pyx,axis=0)
    for i in range(nx):
        Pyx[index[i], i] += delta  #np.divide(1,ny)
    Pyx = normalize(Pyx ,norm='l1',axis=0)    
    Py = np.dot(Pyx, Xhot)

    Y = np.zeros(nsamples)
    for i in range(nsamples):
        Y[i] = np.random.choice(range(0,ny), 1, p = Py[:,i] )
    #plt.imshow(Pyx)
    #plt.colorbar()
    #print(Pyx)
    #plt.show()
    return Y,Pyx

def generate2DSamples(xCard, yCard, nSamples):
    
    # randomly pick joint distribution, normalize
    Pxy = np.random.random([xCard, yCard])
    Pxy = Pxy / sum(sum(Pxy))

    # compute marginals
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0)    
    lp = np.reshape(Pxy, xCard*yCard)
        
    data = np.random.choice(range(xCard*yCard), nSamples, p=lp)
    
    X = (data/yCard).astype(np.int) 
    Y = data % yCard
    
    return([Pxy, Px, Py, X, Y])

def generatePerturbed(p_ZCondX0, amp=0.5):
    p_ZCondX = np.zeros(p_ZCondX0.shape,dtype='float64')
    np.copyto(p_ZCondX, p_ZCondX0)
    #the greatest probability 
    index = np.argmax(p_ZCondX,axis=0)
    for i in range(p_ZCondX.shape[1]):
        p_ZCondX[index[i], i] += amp
        
    p_ZCondX = normalize(p_ZCondX ,norm='l1',axis=0)
    return p_ZCondX

