from lib.get_labels import *
# from resources.Read_Trace_txt import raw_data
from lib.data_transforms import Data
from lib.TO_device import TO_device, DeviceDataLoader
from lib.nerual.nn_utils import *
from lib.custom_dataset import mydataset
# from nn_class_MLP import Network
# from nn_class_CNN import ascadCNNbest
from lib.nerual.nn_class_CNN import Network_l2, Network_l3, Network_l3_u, mlp, mlp_jc, Network_jc, mlp_3
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
import math
# import torch
# test

import h5py
import wandb




def train(model, device, train_loader, optimizer, epoch, data_temp, scheduler):
    model.train()
    model.eval()

    train_total_loss = 0
    total_grad = []
    for batch in train_loader:
        index, traces, labels = batch
        traces = sca_preprocessing.trcs_scaled_centrolize_agmt( traces, torch.from_numpy(data_temp.mean).to(device), torch.sqrt( torch.from_numpy(data_temp.var) ).to(device) )
        traces = traces.unsqueeze(1)
        traces.requires_grad=True
        preds = model(traces)
       # train_loss = loss_functions.KNLL( preds, labels.long() )
        if(wandb.config.loss_function=='mse'):
            loss = torch.nn.MSELoss()
            train_loss = loss(preds, labels.long())
        elif(wandb.config.loss_function =='nll'):

            loss = torch.nn.NLLLoss()
            train_loss = loss(preds, labels.long())
        elif(wandb.config.loss_function == 'cross'):

            loss = torch.nn.CrossEntropyLoss()
            train_loss = loss(preds, labels.long())
        elif(wandb.config.loss_function == 'mine_cross'):
            if(wandb.config.layer==8 or wandb.config.layer==9):
                preds = torch.sum(preds,dim=2)
            train_loss = loss_functions.corr_loss(preds, labels)





        optimizer.zero_grad()

        train_loss.backward()
        optimizer.step()
        if epoch <= 189: scheduler.step()
        else: 
            for g in optimizer.param_groups: g['lr'] = g['lr'] * 0.95

        train_total_loss += train_loss.item()
        temp_grad = traces.grad
        #temp_grad = torch.abs(temp_grad)
        temp_grad_cpu = torch.sum(temp_grad,dim=(0,1)).cpu().detach().numpy()
        total_grad.append(temp_grad_cpu)

    return train_total_loss, total_grad


def test(models, device, test_loader, data_temp):
    models.eval()

    with torch.no_grad():
        all_vali_preds, all_vali_labels = get_all_preds_labels(model=models, loader=test_loader, device=device, mean=data_temp.mean, var=data_temp.var)
        #vali_loss = loss_functions.KNLL( all_vali_preds, all_vali_labels.long() )
        if(wandb.config.loss_function=='mse'):
            loss = torch.nn.MSELoss()
            vali_loss = loss(all_vali_preds, all_vali_labels.long())
        elif(wandb.config.loss_function =='nll'):

            loss = torch.nn.NLLLoss()
            vali_loss = loss(all_vali_preds, all_vali_labels.long())
        elif(wandb.config.loss_function == 'cross'):

            loss = torch.nn.CrossEntropyLoss()
            vali_loss = loss(all_vali_preds, all_vali_labels.long())
        elif(wandb.config.loss_function == 'mine_cross'):
            if(wandb.config.layer==8 or wandb.config.layer==9):
                all_vali_preds = torch.sum(all_vali_preds,dim=2)
            vali_loss = loss_functions.corr_loss(all_vali_preds, all_vali_labels)




        vali_total_correct = get_num_correct(all_vali_preds, all_vali_labels)

    return vali_loss, vali_total_correct

def nn_train( plt, cpt, data, bit_poss, byte_pos, sample_num):

    #Network part
    if(wandb.config.layer==2):
        network = Network_l2( traceLen=sample_num, num_classes=1 )
    elif(wandb.config.layer==3):
        network = Network_l3( traceLen=sample_num, num_classes=1 )
    elif(wandb.config.layer==4):
        network = Network_l3_u(traceLen=sample_num, num_classes=1)
    elif(wandb.config.layer==6):
        network = Network_jc(traceLen=sample_num, num_classes=1)
    elif(wandb.config.layer==7):
        network = mlp(traceLen=sample_num, num_classes=1)
    elif(wandb.config.layer==8):
        network = mlp_jc(traceLen=sample_num, num_classes=1)
    elif(wandb.config.layer==9):
        network = mlp_3(traceLen=sample_num, num_classes=1)






    #labels = get_LSB( atk_round=hp.atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )
    labels = get_HammingWeight( atk_round=1, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )
    DV = TO_device()
    DV.get_default_device()

    Data1 = Data(data, labels.T)
    key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]

    for i in range(1):

        

        #Wrong key or right key
        key_list = np.arange(256)
        key_list = key_list[key_list!= key[byte_pos]]
    
        for key_guess in range(1):
            if(wandb.config.wrong_key==0):
                key_guess = key[byte_pos]
            else:
                key_guess = key_list[wandb.config.wrong_key-1]


            Data1.no_resample(key_guess, wandb.config.traces_num*0.8, wandb.config.traces_num*0.2, 0)
            Data1.data_spilt()
            Data1.features_normal_db()
            Data1.to_torch() 
            md_train = mydataset( Data1.train, Data1.train_labels.byte() )
            train_loader = torch.utils.data.DataLoader(md_train, batch_size=wandb.config.train_batch_size, shuffle=True, drop_last=True)
            train_loader = DeviceDataLoader(train_loader, DV.device)

            md_vali = mydataset( Data1.vali, Data1.vali_labels.byte() )
            vali_loader = torch.utils.data.DataLoader(md_vali, batch_size=wandb.config.vali_batch_size)
            print(Data1.vali_labels.max())
            vali_loader = DeviceDataLoader(vali_loader, DV.device)


            network.train()
            # move network to deivce
            TO_device.to_device(network, DV.device)

            #Optimizer part
            if wandb.config.optimizer=='sgd':
                optimizer = optim.SGD(network.parameters(), lr=wandb.config.lr, momentum=0.9, nesterov=True)
                
            elif wandb.config.optimizer=='rmsprop':
                optimizer = optim.RMSprop(network.parameters(), lr=wandb.config.lr, weight_decay=1e-5)
            elif wandb.config.optimizer=='adam':
                optimizer = optim.Adam(network.parameters(), lr=wandb.config.lr)  
            elif wandb.config.optimizer=='nadam':
                optimizer = optim.NAdam(network.parameters(), lr=wandb.config.lr, betas=(0.9,0.999))

            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=wandb.config.lr, max_lr=0.001, step_size_up=60, mode='triangular', cycle_momentum=False, last_epoch=-1)


         
            train_total_loss = 0
            total_grad_list = []

            wandb.watch(network,log='all')
            for epoch in range(wandb.config.epochs):

                vali_loss, vali_total_correct = test(network,DV.device,vali_loader,Data1)
                print(
                        "key guess j:", key_guess,
                        "epoch:", epoch,
                        "vali_set_total_correct:", vali_total_correct,
                        "vali_loss:", vali_loss.item(), 
                        "train_total_loss:", train_total_loss
                    )
                wandb.log({"epoch":epoch, 
                            "vali_set_total_correct": vali_total_correct,
                            "vali_loss": vali_loss.item(),
                            "loss": train_total_loss,
                            "key" : key_guess,

                })
                if(math.isnan(vali_loss.item())):
                    continue

                train_total_loss,total_grad = train(network, DV.device, train_loader, optimizer,epoch, Data1, scheduler)

                total_grad_list.append(total_grad)
            total_grad_np = np.array(total_grad_list).astype(np.float32)
                
            with h5py.File(f'grad5/{wandb.config.project_name}_{wandb.config.wrong_key}.h5', 'w') as f:
                  f.create_dataset('grad', data=total_grad_np)

            torch.save(network.state_dict(), 'model/model.h5')
            wandb.save('model.h5')



