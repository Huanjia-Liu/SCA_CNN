# from get_labels import *
import torch 
import torch.nn.functional as F
import numpy as np

class Data():
    def __init__(self, samples=None, labels=None):
        if ( (samples==None).all() ):
            self.samples = []
        else:
            self.samples = samples
        if ( (labels==None).all() ):
            self.labels = []
        else:
            self.labels = labels

        self.tempsamples = None
        self.templabels = None

        self.pairs = None
        self.train = None
        self.train_labels = None
        self.vali = None
        self.vali_labels = None
        self.test = None
        self.test_labels = None

        self.mean = None
        self.var = None
        self.max = None

        # self.pairs = np.concatenate( (self.samples, self.labels), axis=1 )
        self.train_pairs = None
        self.vali_pairs = None
        self.test_pairs = None

        self.train_size = None
        self.vali_size = None
        self.test_size = 0

    def binary_resample_balanced(self, key_guess, ratio):
        num_class0 = len(self.labels) - self.labels[:,key_guess].sum()
        print("number of zeros:", num_class0)
        index0 = np.where( self.labels[:,key_guess]==0 )[0]
        samples0 = self.samples[index0]
        labels0 = self.labels[index0, key_guess]
        print( "No. of 0's", len(samples0))
        index1 = np.nonzero( self.labels[:,key_guess] )[0]
        samples1 = self.samples[index1]
        labels1 = self.labels[index1, key_guess]
        print( "No. of 1's", len(samples1))

        # shuffle data1
        idx1 = np.random.randint(len(index1), size=ratio*num_class0)
        samples1 = samples1[idx1]
        labels1 = labels1[idx1]

        # concatenate class0 and class1
        self.tempsamples = np.concatenate( (samples0, samples1), axis=0)
        self.templabels = np.concatenate( (labels0, labels1), axis=0)

        # assign data size
        data_size = (ratio+1)*num_class0 
        self.train_size = int( data_size*0.7 )
        self.vali_size = data_size - self.train_size
        print( "used data size is:", data_size )
    
    def no_resample(self, key_guess, hp):
        self.tempsamples = self.samples
        self.templabels = self.labels[:, key_guess]
        
        self.train_size = hp.train_size
        self.vali_size =  hp.vali_size
        self.test_size =  hp.test_size
        
        # return (ratio+1)*num_class0

    def data_randomize(self):
        random_seed = 0
        self.pairs = np.concatenate( (self.tempsamples, np.expand_dims(self.templabels, axis=1)), axis=1 ).astype('float32')
        # print( self.pairs[0:5] )
        # print()
        np.random.seed(random_seed)
        np.random.shuffle(self.pairs)
        # print( self.pairs[0:5] )
        self.tempsamples = self.pairs[:,:-1]
        self.templabels = self.pairs[:,-1]

        # for i in range(5):
        #     print( self.samples.shape, self.labels.shape )

    def data_spilt(self):
        self.train = self.tempsamples[ : self.train_size]
        self.vali = self.tempsamples[ self.train_size : self.train_size + self.vali_size ]
        self.test = self.tempsamples[ self.train_size + self.vali_size : self.train_size + self.vali_size + self.test_size ]

        self.train_labels = self.templabels[ : self.train_size]
        self.vali_labels = self.templabels[ self.train_size : self.train_size + self.vali_size ]
        self.test_labels = self.templabels[ self.train_size + self.vali_size : self.train_size + self.vali_size + self.test_size ]

        print( 
                "train set shape:", self.train.shape, self.train_labels.shape,
                "vali set shape:", self.vali.shape, self.vali_labels.shape,
                "test set shape:", self.test.shape, self.test_labels.shape
            )

    # def pair_up(self, key_guess, POIs, POIs_en):
    #     if (POIs_en):
    #         self.train_pairs = np.concatenate( ( np.take(self.train, POIs, axis=1), np.expand_dims(self.train_labels[:,key_guess], axis=1) ), axis=1 )
    #         self.vali_pairs = np.concatenate( ( np.take(self.vali, POIs, axis=1), np.expand_dims(self.vali_labels[:,key_guess], axis=1) ), axis=1 )
    #         self.test_pairs = np.concatenate( ( np.take(self.test, POIs, axis=1), np.expand_dims(self.test_labels[:,key_guess], axis=1) ), axis=1 )
    #     else:
    #         self.train_pairs = np.concatenate( (self.train, np.expand_dims(self.train_labels[:,key_guess], axis=1)), axis=1 )
    #         self.vali_pairs  = np.concatenate( (self.vali,  np.expand_dims(self.vali_labels[:,key_guess],  axis=1)), axis=1 )
    #         self.test_pairs  = np.concatenate( (self.test,  np.expand_dims(self.test_labels[:,key_guess],  axis=1)), axis=1 )
    #     print( "training set no. of 1:", self.train_labels[:,key_guess].sum(), " vali set no. of 1:", self.vali_labels[:,key_guess].sum() )
    
    def features_normal_db(self):
        self.mean = np.array(np.mean(self.train))
        self.var = np.array(np.var(self.train))
    
    def max_normal(self):
        self.max = np.amax( self.train )
        self.train = self.train / self.max
        self.vali = self.vali / self.max
        self.test = self.test / self.max

    def to_torch(self):
        # self.train_pairs = torch.from_numpy(self.train_pairs)
        # self.vali_pairs = torch.from_numpy(self.vali_pairs)
        # self.test_pairs = torch.from_numpy(self.test_pairs)
        self.train = torch.from_numpy(self.train)
        self.vali = torch.from_numpy(self.vali)
        self.test = torch.from_numpy(self.test)
        self.train_labels = torch.from_numpy(self.train_labels)
        self.vali_labels = torch.from_numpy(self.vali_labels)
        self.test_labels = torch.from_numpy(self.test_labels)
 

    
    # add other normalization algos

        

# test code
import time
from lib.get_labels import get_MSB
from hyperparam import hyperparams
from lib.function_initialization import read_plaintextANDmask
from lib.hdf5_files_import import load_ascad

def main():

    path_trace = r'Data\ASCAD\ASCAD.h5'
    hp = hyperparams()

    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad( path_trace, True )
    plt, masks = read_plaintextANDmask(Metadata_profiling)

    labels = get_MSB( atk_round=hp.atk_round, byte_pos=hp.byte_pos, plt=plt, cpt=None )
    Data1 = Data( X_profiling, labels.T )
    for key_guess in range(1):
        key_guess = 1
        Data1.no_resample(key_guess, hp)
        Data1.data_randomize()
        Data1.data_spilt()
        Data1.features_normal_db()
        Data1.to_torch()


# test
import time

if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))




