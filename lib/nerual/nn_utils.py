import torch
import torch.nn.functional as F
import numpy as np
from lib.function_initialization import read_plaintextANDmask
from lib.hdf5_files_import import load_ascad
from lib.data_transforms import Data
from lib.get_labels import *
from hyperparam import hyperparams 
import matplotlib.pyplot as plt
from lib.SCA_preprocessing import sca_preprocessing

# define utility function
def get_all_preds_labels(model, loader, device, mean, var):

    # # print sum of first layer weight
    # print(np.sum(model.fc1.weight.data.to('cpu').numpy()))

    all_preds = torch.tensor([]).to(device).float()
    all_labels = torch.tensor([]).to(device).byte()
    mean = torch.from_numpy(mean).to(device)
    var = torch.from_numpy(var).to(device)
    for batch in loader:
        indices, traces, labels = batch
        
        # traces = F.batch_norm(traces.float(), mean.float(), var.float())
        traces = sca_preprocessing.trcs_scaled_centrolize_agmt( traces, mean, torch.sqrt(var)  )
        traces = traces.unsqueeze(1)   
        
        preds = model(traces)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels)
            ,dim=0
        )
    return all_preds,all_labels

def get_all_preds_labels_indices(model, loader, device, mean, var):

    # # print sum of first layer weight
    # print(np.sum(model.fc1.weight.data.to('cpu').numpy()))

    all_preds = torch.tensor([]).to(device).float()
    all_labels = torch.tensor([]).to(device).long()
    all_indices = torch.tensor([]).to(device).short()

    mean = torch.from_numpy(mean).to(device)
    var = torch.from_numpy(var).to(device)
    for batch in loader:
        indices, traces, labels = batch
        
        traces = F.batch_norm(traces.float(), mean.float(), var.float())
        traces = traces.unsqueeze(1)   
        
        preds = model(traces)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels)
            ,dim=0
        )
        all_indices = torch.cat(
            (all_indices, indices.short())
            ,dim=0
        )
    return all_preds,all_labels,all_indices

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def get_num_correct_mse(preds,labels):
    return preds.eq(labels).sum().item()

def mse_get_num_correct( preds, labels ):
    preds_ = preds >= 0.5
    return preds_.eq(labels).sum().item()

def get_correct_preds_index(preds, labels, indices):
    # return np.take(indices, ( (preds.argmax(dim=1)-labels)==0 ).nonzero().numpy(), axis=0) 
    temp = ( (preds.argmax(dim=1)-labels)==0 ).nonzero()
    return indices[temp]

def get_num_of_elements( labels, label ):
    return np.count_nonzero( labels==label )


def nn_train_performance(network, train_loader, vali_loader, DV, Data1, seq, epoch):
    with torch.no_grad():
        # training accuracy
        all_train_preds, all_train_labels = get_all_preds_labels(model=network, loader=train_loader, device=DV.device, mean=Data1.mean, var=Data1.var)
        # mse accuracy
        # all_train_preds = all_train_preds.reshape(-1)
        # train_total_correct = mse_get_num_correct( all_train_preds, all_train_labels )
        # crossentropy accuracy
        train_total_correct = get_num_correct(all_train_preds, all_train_labels)
        train_accuracy = train_total_correct / Data1.train_size
        # train_acc[seq, epoch] = train_accuracy
        train_loss2 = F.cross_entropy(all_train_preds, all_train_labels)
      
        all_vali_preds, all_vali_labels = get_all_preds_labels(model=network, loader=vali_loader, device=DV.device, mean=Data1.mean, var=Data1.var)
        vali_total_correct = get_num_correct(all_vali_preds, all_vali_labels)
        vali_accuracy = vali_total_correct / Data1.vali_size
        vali_loss = F.cross_entropy(all_vali_preds, all_vali_labels)

        print(
            "seq:", seq,
            "epoch:", epoch,
            "vali_acc:", vali_accuracy,
            "vali_loss:", vali_loss.item(),
            "train_loss:", train_loss2.item(),
            "train_acc:", train_accuracy
        )
    return train_accuracy, train_loss2.item(), vali_accuracy, vali_loss.item()


def nn_test_performance( network, test_loader, test_size, DV, train_mean, train_var, seq, epoch ):
    with torch.no_grad():
        # training accuracy
        all_test_preds, all_test_labels = get_all_preds_labels(model=network, loader=test_loader, device=DV.device, mean=train_mean, var=train_var)
        # crossentropy accuracy
        test_total_correct = get_num_correct(all_test_preds, all_test_labels)
        test_accuracy = test_total_correct / test_size
        # train_acc[seq, epoch] = train_accuracy
        test_loss = F.cross_entropy(all_test_preds, all_test_labels)
    return test_accuracy, test_loss.item()




def load_whole_ascad_byte2_data_bit_label(byte_pos, bit_pos):
    hp = hyperparams()
    ASCAD_path = r'Data\ASCAD\ASCAD.h5'
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad( ASCAD_path, True )
    # print(Y_attack)
    # print(Y_profiling)
    plt_pr, masks_pr = read_plaintextANDmask(Metadata_profiling)
    plt_at, masks_at = read_plaintextANDmask(Metadata_attack)
    plt = np.concatenate( (plt_pr, plt_at), axis=0 )
    data = np.concatenate( (X_profiling, X_attack), axis=0  )
    labels = get_BFB( atk_round=hp.atk_round, byte_pos=byte_pos, plt=plt, cpt=None, bit_pos=bit_pos )
    print("labels shape:", labels.shape)
    print("data shape:", data.shape)
    Data1 = Data(data, labels.T)
    
    return Data1


def load_whole_ascad_byte2_data():
    hp = hyperparams()
    ASCAD_path = r'Data\ASCAD\ASCAD.h5'
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad( ASCAD_path, True )
    # print(Y_attack)
    # print(Y_profiling)
    plt_pr, masks_pr = read_plaintextANDmask(Metadata_profiling)
    plt_at, masks_at = read_plaintextANDmask(Metadata_attack)
    plt = np.concatenate( (plt_pr, plt_at), axis=0 )
    masks = np.concatenate( (masks_pr, masks_at), axis=0 )
    data = np.concatenate( (X_profiling, X_attack), axis=0  )
    print("data shape:", data.shape)
    
    return data, plt, masks


def load_whole_ascad_byte2_data_byte_label(byte_pos):
    hp = hyperparams()
    ASCAD_path = r'Data\ASCAD\ASCAD.h5'
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad( ASCAD_path, True )
    # print(Y_attack)
    # print(Y_profiling)
    plt_pr, masks_pr = read_plaintextANDmask(Metadata_profiling)
    plt_at, masks_at = read_plaintextANDmask(Metadata_attack)
    plt = np.concatenate( (plt_pr, plt_at), axis=0 )
    data = np.concatenate( (X_profiling, X_attack), axis=0  )
    labels = get_HammingWeight( atk_round=1, byte_pos=byte_pos, plt=plt, cpt=None )
    print("labels shape:", labels.shape)
    print("data shape:", data.shape)
    Data1 = Data(data, labels.T)
    
    return Data1


# from resources.Read_Trace_txt import raw_data

# def load_whole_FPGAgf_byte15_data(byte_pos, bit_pos):
#     hp = hyperparams()
#     raw_data_1 = raw_data( hp.path_trace, hp.path_plt, hp.path_cpt )
#     raw_data_1.read_multi_h5()
#     # raw_data_1.read_multi_plt()
#     raw_data_1.read_multi_cpt()
#     labels = get_BFB_HD_last( atk_round=hp.atk_round, byte_pos=byte_pos, plt=None, cpt=raw_data_1.cpt, bit_pos=bit_pos )
#     print("labels shape:", labels.shape)
#     print("data shape:", raw_data_1.data_train.shape)
#     Data1 = Data(raw_data_1.data_train[:,hp.start:hp.end], labels.T)

#     return Data1











                # with torch.no_grad():
                #     # validation 
                #     all_vali_preds, all_vali_labels = get_all_preds_labels(model=network, loader=vali_loader, device=DV.device, mean=Data1.mean, var=Data1.var)
        
                #     # mse accuracy
                #     # all_vali_preds = all_vali_preds.reshape(-1)
                #     # vali_total_correct = mse_get_num_correct( all_vali_preds, all_vali_labels )
                #     # crossentropy accuracy
                #     vali_loss = F.cross_entropy(all_vali_preds, all_vali_labels)
                #     vali_total_correct = get_num_correct(all_vali_preds, all_vali_labels)
                #     vali_accuracy = vali_total_correct / Data1.vali_size
                #     vali_acc[seq, epoch] = vali_accuracy
                #     # print(
                #     #     "key guess j:", key_guess,
                #     #     "epoch:", epoch,
                #     #     "vali_set_total_correct:", vali_total_correct,
                #     #     "train_total_loss:", train_total_loss
                #     # )
                #     # confusion_matrix( all_vali_labels, all_vali_preds, 2 )

                #     # training accuracy
                #     all_train_preds, all_train_labels = get_all_preds_labels(model=network, loader=train_loader, device=DV.device, mean=Data1.mean, var=Data1.var)
                #     # mse accuracy
                #     # all_train_preds = all_train_preds.reshape(-1)
                #     # train_total_correct = mse_get_num_correct( all_train_preds, all_train_labels )
                #     # crossentropy accuracy
                #     train_total_correct = get_num_correct(all_train_preds, all_train_labels)
                #     train_accuracy = train_total_correct / Data1.train_size
                #     train_acc[seq, epoch] = train_accuracy
                #     # confusion_matrix( all_train_labels, all_train_preds, 2 )
                #     print(
                #         "kg 256 seq:", seq,
                #         "epoch:", epoch,
                #         "vali_acc:", vali_accuracy,
                #         "vali_loss:", vali_loss.item(),
                #         "train_t_loss:", train_total_loss,
                #         "train_acc:", train_accuracy
                #     )


def check_masked_01_ratio():
    key = 224
    hp = hyperparams()
    points = 60
    point_gap = 1000
    ratio_arr = np.zeros( (256, points) )
    data, plts, masks = load_whole_ascad_byte2_data()
    masked_intmv = get_masked_INTMV( atk_round=1, byte_pos=2, plt=plts, cpt=None, masks=masks )
    
    
    for bit_pos in range(8):
        masked_bit = np.bitwise_and(np.right_shift(masked_intmv, bit_pos),1)
        masked_bit = masked_bit[ 224, np.newaxis ]
        data_plus = np.concatenate( (data, masked_bit.T), axis=1 )
        labels = get_BFB( atk_round=1, byte_pos=2, plt=plts, cpt=None, bit_pos=bit_pos )
        Data1 = Data(data_plus, labels.T)
        for key_guess in range(256):
            Data1.no_resample( key_guess, hp )
            Data1.data_spilt()
            train_TempTracesMSB = [[] for _ in range(2)]
            for point in range(points):
                # trace = (point+1)*point_gap
                for index in range( 1000 ):
                    train_TempTracesMSB[ Data1.train_labels[point*point_gap+index] ].append( Data1.train[point*point_gap+index] )
                one = np.array( train_TempTracesMSB[0] )[:,700].sum()
                ratio_arr[key_guess, point] =  one / len( train_TempTracesMSB[0] )
            if ( key_guess==key ):
                ck = key_guess
            elif ( key_guess==222 ):
                wk = key_guess
            else:
                plt.plot( ratio_arr[key_guess], color="grey" )
                # plt.show()
        
        plt.plot( ratio_arr[wk], color="blue" )
        plt.plot( ratio_arr[ck], color="red" )
        fig = plt.gcf()
        fig.set_size_inches(32, 18)
        plt.savefig(f"bit{bit_pos}", bbox_inches='tight')
        # plt.show()
        plt.close()
        
        


def main():
    check_masked_01_ratio()
  


import time 

if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))             

                
                    
        

    