from lib.get_labels import *
# from resources.Read_Trace_txt import raw_data
from lib.data_transforms import Data
from lib.TO_device import TO_device, DeviceDataLoader
from lib.nerual.nn_utils import *
from lib.custom_dataset import mydataset
# from nn_class_MLP import Network
# from nn_class_CNN import ascadCNNbest
from lib.nerual.nn_class_CNN import Network_l2, Network_l3, Network_l3_u
# from nn_save import nn_save
# from plot_accuracy import plot_acc
#from nn_tensorboard import *
#from noise_addition import white_noise
# from POI_selection import POI_selection
# from confusion_matrix import confusion_matrix
# from resample import binary_resample_balanced
#from threeD_plotting import threeD_plotting
from lib.hdf5_files_import import read_multi_plt, read_multi_h5, load_ascad_metadata, load_raw_ascad
from lib.function_initialization import read_plts
from lib.SCA_preprocessing import sca_preprocessing

import numpy as np
import matplotlib.pyplot as mplt

import torch.nn.functional as F
import torch.optim as optim
from itertools import product
from lib.nerual.nn_loss_functions import loss_functions
# import torch
# test


import wandb




def train(model, device, train_loader, optimizer, epoch, data_temp):
    model.train()
    model.eval()

    train_total_loss = 0

    for batch in train_loader:
        index, traces, labels = batch
        traces = sca_preprocessing.trcs_scaled_centrolize_agmt( traces, torch.from_numpy(data_temp.mean).to(device), torch.sqrt( torch.from_numpy(data_temp.var) ).to(device) )
        traces = traces.unsqueeze(1)
        preds = model(traces)
        #train_loss = loss_functions.KNLL( preds, labels.long() )

        train_loss = loss_functions.corr_loss(preds, labels)

        #loss = torch.nn.NLLLoss()
        #train_loss = loss(preds, labels.long())





        optimizer.zero_grad()

        train_loss.backward()
        optimizer.step()
        train_total_loss += train_loss.item()
    return train_total_loss


def test(models, device, test_loader, data_temp):
    models.eval()

    with torch.no_grad():
        all_vali_preds, all_vali_labels = get_all_preds_labels(model=models, loader=test_loader, device=device, mean=data_temp.mean, var=data_temp.var)
        #vali_loss = loss_functions.KNLL( all_vali_preds, all_vali_labels.long() )
        vali_loss = loss_functions.corr_loss(all_vali_preds, all_vali_labels)

        #loss = torch.nn.NLLLoss()
        #vali_loss = loss(all_vali_preds, all_vali_labels.long())



        vali_total_correct = get_num_correct(all_vali_preds, all_vali_labels)

    return vali_loss, vali_total_correct

def nn_train(hp, plt, cpt, data, bit_poss, byte_pos):



    network = Network_l3_u( traceLen=hp.sample_num, num_classes=hp.output )
    stat_params = network.state_dict()
    #labels = get_LSB( atk_round=hp.atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )
    labels = get_HammingWeight( atk_round=hp.atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )
    DV = TO_device()
    DV.get_default_device()

    Data1 = Data(data, labels.T)
    key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]

    for i in range(1):
        folder_comment = f'CNN_Comp_ASCADde50_first_HW-byte={byte_pos}'
        for key_guess in range(10):
            #key_guess = 0   #key[byte_pos]
            if(key_guess ==0):
                key_guess = key[byte_pos]
            Data1.no_resample(key_guess, hp)
            Data1.data_spilt()
            Data1.features_normal_db()
            Data1.to_torch() 
            md_train = mydataset( Data1.train, Data1.train_labels.byte() )
            train_loader = torch.utils.data.DataLoader(md_train, batch_size=hp.train_batch_size, shuffle=True, drop_last=True)
            train_loader = DeviceDataLoader(train_loader, DV.device)

            md_vali = mydataset( Data1.vali, Data1.vali_labels.byte() )
            vali_loader = torch.utils.data.DataLoader(md_vali, batch_size=hp.vali_batch_size)
            print(Data1.vali_labels.max())
            vali_loader = DeviceDataLoader(vali_loader, DV.device)
            network = Network_l3_u( traceLen=hp.sample_num, num_classes=hp.output )
            network.load_state_dict( stat_params )
            network.train()
            # move network to deivce
            TO_device.to_device(network, DV.device)




           # optimizer = optim.Adam(network.parameters(), lr=0.0001)
            optimizer = optim.SGD(network.parameters(), lr=0.0002, momentum=0.9, nesterov=True)
            
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.learning_rate, max_lr=0.001, step_size_up=60, mode='triangular', cycle_momentum=False, last_epoch=-1)
                
            train_total_loss = 0

            
            for epoch in range(50):

                vali_loss, vali_total_correct = test(network,DV.device,vali_loader,Data1)
                print(
                        "key guess j:", key_guess,
                        "epoch:", epoch,
                        "vali_set_total_correct:", vali_total_correct,
                        "vali_loss:", vali_loss.item(), 
                        "train_total_loss:", train_total_loss
                    )



                train_total_loss = train(network, DV.device, train_loader, optimizer,epoch, Data1)
            torch.save(network.state_dict(), 'model.h5')




