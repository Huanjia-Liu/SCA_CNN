from lib.get_labels import *
from lib.data_transforms import Data
from lib.TO_device import TO_device, DeviceDataLoader
from lib.nerual.nn_utils import *
from lib.custom_dataset import mydataset
from lib.nerual.nn_class_CNN import Network_l2, Network_l3, Network_l3_u, mlp, mlp_jc, Network_jc, mlp_3, assign_variable_nn, cnn_co
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

from hyperparam import hyperparam as hp
from sweep_para import sweep_para as sp

import h5py
import wandb


save_path = "/home/admin1/Documents/git/SCA_CNN_result/"

#train part
def train(model, device, train_loader, optimizer, epoch, data_temp, scheduler):
    global loss_function, sweep_enable
    model.train()
    model.eval()

    train_total_loss = 0
    total_grad = []
    for batch in train_loader:

        index, traces, labels = batch

        #WST and DL are processed in GRAM in GRAM for efficiency
        if(pre_process == "scattering" and sweep_enable == True):        
            traces = sca_preprocessing.scattering(traces, J = wandb.config.J, M = traces.shape[1], Q = wandb.config.Q)

        traces = sca_preprocessing.trcs_scaled_centrolize_agmt( traces, torch.from_numpy(data_temp.mean).to(device), torch.sqrt(torch.from_numpy(data_temp.var)).to(device) )
        traces = traces.unsqueeze(1)
        traces.requires_grad=True
        preds = model(traces)

        #loss_function selection 
        if(loss_function=='mse'):
            loss = torch.nn.MSELoss()
            train_loss = loss(preds, labels.long())
        elif(loss_function =='nll'):

            loss = torch.nn.NLLLoss()
            train_loss = loss(preds, labels.long())
        elif(loss_function == 'cross'):

            loss = torch.nn.CrossEntropyLoss()
            train_loss = loss(preds, labels.long())
        elif(loss_function == 'mine_cross'):
            #if(layer == 7 or layer==8 or layer==9):
            #    preds = torch.sum(preds,dim=1)
            train_loss = loss_functions.corr_loss(preds, labels)





        optimizer.zero_grad()

        train_loss.backward()
        optimizer.step()
        if epoch <= 189: scheduler.step()
        else: 
            for g in optimizer.param_groups: g['lr'] = g['lr'] * 0.95

        train_total_loss += train_loss.item()
        if(False):
            temp_grad = traces.grad
            temp_grad = torch.abs(temp_grad)
            temp_grad_cpu = torch.sum(temp_grad,dim=(0,1)).cpu().detach().numpy()
            total_grad.append(temp_grad_cpu)
        else:
            total_grad = 0

    return train_total_loss, total_grad


#test_part
def test(models, device, test_loader, data_temp):
    global loss_function, sweep_enable
    models.eval()

    with torch.no_grad():
        if(pre_process == 'scattering' and sweep_enable == True):
            all_vali_preds, all_vali_labels = get_all_preds_labels_gpu(model=models, loader=test_loader, device=device, data_temp= data_temp)
        else:
            all_vali_preds, all_vali_labels = get_all_preds_labels(model=models, loader=test_loader, device=device, mean=data_temp.mean, var=data_temp.var)
        #Different loss function
        if(loss_function=='mse'):
            loss = torch.nn.MSELoss()
            vali_loss = loss(all_vali_preds, all_vali_labels.long())
        elif(loss_function =='nll'):

            loss = torch.nn.NLLLoss()
            vali_loss = loss(all_vali_preds, all_vali_labels.long())
        elif(loss_function == 'cross'):

            loss = torch.nn.CrossEntropyLoss()
            vali_loss = loss(all_vali_preds, all_vali_labels.long())
        elif(loss_function == 'mine_cross'):
            #if(layer ==7 or layer==8 or layer==9):
            #    all_vali_preds = torch.sum(all_vali_preds,dim=1)
            vali_loss = loss_functions.corr_loss(all_vali_preds, all_vali_labels)    #shape (n,1) (n)

    return vali_loss









def assign_variable(sweep_mode):
    global loss_function, layer, optimizer, wrong_key, lr, epochs
    if(sweep_mode == 'tensorboard'):
        loss_function = sp.loss_function
        layer = sp.layer
        optimizer = sp.optimizer
        wrong_key = sp.wrong_key
        lr = sp.lr
        epochs = sp.epochs
    elif(sweep_mode == 'wandb'):
        loss_function = wandb.config.loss_function
        layer = wandb.config.layer
        optimizer = wandb.config.optimizer
        wrong_key = wandb.config.wrong_key
        lr = wandb.config.lr
        epochs = wandb.config.epochs





def nn_train( plt, cpt, data, bit_poss, byte_pos, sample_num, sweep_mode, pre_process_main, sweep_enable_main):
    global layer, wrong_key, optimizer, epoch, lr, pre_process, sweep_enable
    assign_variable(sweep_mode)
    assign_variable_nn(sweep_mode)
    
    pre_process = pre_process_main
    sweep_enable = sweep_enable_main
    
    
    #Network structure 
    if(layer==2):
        network = Network_l2( traceLen=sample_num, num_classes=1 )
    elif(layer==3):
        network = Network_l3( traceLen=sample_num, num_classes=1 )
    elif(layer==4):
        network = Network_l3_u(traceLen=sample_num, num_classes=1)
    elif(layer==6):
        network = Network_jc(traceLen=sample_num, num_classes=1)
    elif(layer==7):
        network = mlp(traceLen=sample_num, num_classes=1)
    elif(layer==8):
        network = mlp_jc(traceLen=sample_num, num_classes=1)
    elif(layer==9):
        network = mlp_3(traceLen=sample_num, num_classes=1)
    elif(layer==10):
        network = cnn_co(traceLen=sample_num, num_classes=1)





    #Calculate label using power model, the model is determined by CPA
    if(hp.power_model == 'hd'):
        labels = get_HD_last(atk_round=10, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )
    elif(hp.power_model == 'hw'):
        labels = get_HammingWeight( atk_round=1, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )
    elif(hp.power_model == 'lsb'):
        labels = get_LSB( atk_round=hp.atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt ).astype( 'uint8' )

    #Combine data and labels togehter
    DV = TO_device()
    DV.get_default_device()
    Data1 = Data(data, labels.T)


    key = hp.key 

    for i in range(1):

        #Wrong key or right key
        key_list = np.arange(256)
        key_list = key_list[key_list!= key[byte_pos]]
    
        for key_guess in range(1):
            if(wrong_key==0):
                key_guess = key[byte_pos]
            else:
                key_guess = key_list[wrong_key-1]

            #Split data to train and test part following the ratio 8:2
            Data1.no_resample(key_guess, (hp.trace_end-hp.trace_start)*0.8, (hp.trace_end-hp.trace_start)*0.2, 0)
            Data1.data_spilt()
            
            #Prepare data
            if(pre_process == 'scattering' and sweep_enable == True):
                Data1.features_normal_db_gpu()
            else:    
                Data1.features_normal_db()                     
            Data1.to_torch() 
            md_train = mydataset( Data1.train, Data1.train_labels.byte() )
            train_loader = torch.utils.data.DataLoader(md_train, batch_size=hp.train_batch, shuffle=True, drop_last=True)
            train_loader = DeviceDataLoader(train_loader, DV.device)
            md_vali = mydataset( Data1.vali, Data1.vali_labels.byte() )
            vali_loader = torch.utils.data.DataLoader(md_vali, batch_size=hp.vali_batch)
            print(Data1.vali_labels.max())
            vali_loader = DeviceDataLoader(vali_loader, DV.device)
            print(f"finish---{torch.cuda.memory_reserved(0)/1024/1024/1024}")
            network.train()

            # move network to deivce
            TO_device.to_device(network, DV.device)

            #Optimizer part
            if optimizer=='sgd':
                optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, nesterov=True)
                
            elif optimizer=='rmsprop':
                optimizer = optim.RMSprop(network.parameters(), lr=lr, weight_decay=1e-5)
            elif optimizer=='adam':
                optimizer = optim.Adam(network.parameters(), lr=lr)  
            elif optimizer=='nadam':
                optimizer = optim.NAdam(network.parameters(), lr=lr, betas=(0.9,0.999))

            #Scheduler
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.001, step_size_up=60, mode='triangular', cycle_momentum=False, last_epoch=-1)


            total_vali_list = []
            train_total_loss = 0
            total_grad_list = []
            if(sweep_mode == 'wandb'):
                wandb.watch(network,log='all')
            for epoch in range(epochs):

                vali_loss = test(network,DV.device,vali_loader, Data1)
                print(
                        "key guess j:", key_guess,
                        "epoch:", epoch,
                        "vali_loss:", vali_loss.item(), 
                        "train_total_loss:", train_total_loss
                    )
                if(sweep_mode == 'wandb'):
                    wandb.log({"epoch":epoch, 
                                "vali_loss": vali_loss.item(),
                                "loss": train_total_loss,
                                "key" : key_guess,

                    })
                print(f"g-ram ---{torch.cuda.memory_reserved(0)/1024/1024/1024}GB")

                total_vali_list.append(vali_loss.item())



                if(math.isnan(vali_loss.item())):
                    continue
                train_total_loss,total_grad = train(network, DV.device, train_loader, optimizer,epoch, Data1, scheduler)

                #total_grad_list.append(total_grad)
                #total_grad_np = np.array(total_grad_list).astype(np.float32)
                
#            with h5py.File(f'{save_path}grad/{wandb.config.project_name}_{wandb.config.wrong_key}.h5', 'w') as f:
#                  f.create_dataset('grad', data=total_grad_np)

            torch.save(network.state_dict(), f'{save_path}model.h5')
            if(sweep_mode == 'wandb'):
                wandb.save(f'{save_path}wandb/model.h5')
    return total_vali_list





