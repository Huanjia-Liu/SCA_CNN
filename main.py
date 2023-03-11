import time
from matplotlib.pyplot import scatter  
import numpy as np

from hyperparam import hyperparam as hp       
from sweep_para import sweep_para as sp
from wandb_para import wandb_para as wp

from lib.mem_visual import mem_thread
from lib.hdf5_files_import import read_multi_plt, read_multi_h5, load_ascad_metadata, load_raw_ascad, load_sx_file, read_plts_sx, read_cpts_sx
from lib.function_initialization import read_plts
from lib.SCA_preprocessing import sca_preprocessing as scap
from mine_train import nn_train
import wandb
import torch
import scipy
import sys
import h5py



wrong_key = [x for x in range(256)] 




def config_extract(project_name, index):
    api = wandb.Api()
    project_name = "scattering_5k_100_final"
    runs = api.runs(f'aceleo/{project_name}')
    wrong_key = [x for x in range(256)]   
    run = runs[0]
    for key in run:
        run[key] = {'values': [run[key]]}
    run['wrong_key']['values'] = wrong_key

    sweep_configuration = {
        'method': 'grid',              
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': run
    }
    return sweep_configuration


#Keep seed extractly same
def seed_init():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache() 





def byte_machine( byte, plts, cpts, trcs, sample_num, sweep_mode):
    global sweep_enable

    for bit in range(1):
        bit = 0
        total_vali_list = nn_train( plts, cpts, trcs.astype(np.float32), bit, byte, sample_num, sweep_mode, pre_process, sweep_enable)
    return total_vali_list


##################################################
#
# Tempatly be dropped
#
##################################################


def scattering_batch(traces, batch_size, J, Q):
    trcs_num = traces.shape[0] 
    batch_num = trcs_num // batch_size
    temp_sum = 0
    #scatter_array = np.zeros((trcs_num,))

    if(batch_num == 0):
        scatter_trcs = scap.scattering( trcs=traces[:,:].astype(np.float32), J=J, M=traces.shape[1],Q=Q )
        scatter_array = np.copy(scatter_trcs)
    else:
        for i in range(batch_num):
            print(i)
            if(i == 0):
                scatter_trcs = scap.scattering( trcs=traces[i*batch_size: (i+1)*batch_size,:].astype(np.float32), J=J, M=traces.shape[1],Q=Q )
                scatter_array = np.copy(scatter_trcs)
            else:
                scatter_trcs = scap.scattering( trcs=traces[i*batch_size: (i+1)*batch_size,:].astype(np.float32), J=J, M=traces.shape[1],Q=Q )
                scatter_array = np.vstack((scatter_array,scatter_trcs))

            temp_sum = i
        if(trcs_num%batch_size != 0):
            scatter_trcs = scap.scattering( trcs=traces[temp_sum*batch_size: ,:].astype(np.float32), J=J, M=traces.shape[1],Q=Q )
            if(batch_num != 0):
                scatter_array = np.vstack((scatter_array,scatter_trcs))
            else:
                scatter_array = scatter_trcs.copy()

    return scatter_array


###############################################################
#   Sweep component:
#           preprocess data (scattering or stft) each runs
###############################################################

def sweep_main():

    global sweep_mode, pre_process 
    
    seed_init()
    print(f"g-ram after empty ---{torch.cuda.memory_reserved(0)/1024/1024/1024}GB")


    #load data depend on the type of dataset 
    if(file_type == 'sx'):
        trcs, metadata = load_sx_file(sx_file=hp.path, idx_srt=hp.trace_start, idx_end=hp.trace_end, start=hp.signal_start, end=hp.signal_end, load_metadata=True)
    elif(file_type == 'ascad'):
        trcs, metadata = load_raw_ascad(ascad_database_file =hp.path, idx_srt=hp.trace_start, idx_end=hp.trace_end, start=hp.signal_start, end=hp.signal_end, load_metadata=True)


    print('read data finish')    

    if(sweep_mode == 'wandb'):
        
        wandb.init(project=f"total_test")
        if(pre_process == 'scattering'):
            J = wandb.config.J
            Q = wandb.config.Q
            
            scattered_sample = scap.scattering(trcs[:2], J, trcs[:2].shape[1], Q)  # calculate the shape of trcaces after scattering
            scattered_trcs = trcs

        
        elif(pre_process == 'stft'):
            windows = wandb.config.windows
            f,t,scattered_trcs = scipy.signal.stft(trcs, nperseg=windows)
        elif(pre_process == 'co'):
            scattered_trcs = trcs


    elif(sweep_mode == 'tensorboard'):
        if(pre_process == 'scattering'):
            J = sp.J
            Q = sp.Q
            scattered_trcs = scattering_batch(trcs, 80000, J, Q)
        elif(pre_process == 'stft'):
            windows = sp.windows
            f,t,scattered_trcs = scipy.signal.stft(trcs, nperseg=windows)

               
    #calculate the shape of trace, used in network sturcture    
    print('pre_process is finished')
    if(pre_process == 'stft'):
        sample_num = ( scattered_trcs.shape[1], scattered_trcs.shape[2] )
        print(f'shape is {sample_num}')
    elif(pre_process == 'scattering'):
        sample_num = ( scattered_sample.shape[1], scattered_sample.shape[2] )
        print(f'shape is {sample_num}')
    else:
        sample_num = scattered_trcs.shape[1]
    
    #load plain text and cipher text from dataset
    if(file_type == 'sx'):
        plts = read_plts_sx(metadata = metadata, trace_num = hp.trace_end-hp.trace_start)
        cpts = read_cpts_sx(metadata = metadata, trace_num = hp.trace_end-hp.trace_start)
    elif(file_type == 'ascad'):
        plts = read_plts(metadata = metadata)
        cpts = 0



    for byte in range(1):
        byte=0
        seed_init()
        total_vali_list = byte_machine( hp.byte, plts, cpts, scattered_trcs ,sample_num, sweep_mode)
        print("byte:", byte, "Done!")
    return total_vali_list


###############################################################
#   Keyguess component:
#           preprocess data (scattering or stft) at first (one time)
###############################################################

def keyguess_main():
    global sweep_mode
    
    if(sweep_mode == 'tensorboard'):

        for i in range(256):

            seed_init()
            start_time = time.time()
         
            for byte in range(1):
                
               
                byte_machine( hp.byte, plts, cpts,  scattered_trcs ,sample_num, sweep_mode)
                print("byte:", byte, "Done!")
            stop_time = time.time()
            print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))    
        
    elif(sweep_mode == 'wandb'):
        seed_init()
        wandb.init(project='wandbtest')
        for byte in range(1):
            byte = 2

            byte_machine( hp.byte, plts, cpts, scattered_trcs, sample_num, sweep_mode)
            print("byte:", byte, "Done!")

        






import time
import os
if "__main__" == __name__:
    #hyperparam
    global pre_process, sweep_mode, sweep_enable, network_type, file_type
    pre_process = 'scattering'          # scattering or stft
    sweep_mode = 'wandb'                # wandb or tensorboard 
    network_type = 'mlp'                # mlp or cnn
    file_type = 'ascad'                 # sx or ascad

    sweep_enable = False
    project_name = 'scattering_100_2.5k_time_real_new_0302'
    sweep_num = 300
    

    pid = os.getpid()                   # show memory usage 
   # memory = mem_thread(pid,2)
   # memory.start()


    #key guess part   
    if(not(sweep_enable)):
        if(file_type == 'sx'):
            trcs, metadata = load_sx_file(sx_file=hp.path, idx_srt=hp.trace_start, idx_end=hp.trace_end, start=hp.signal_start, end=hp.signal_end, load_metadata=True)
        elif(file_type == 'ascad'):
            trcs, metadata = load_raw_ascad(ascad_database_file =hp.path, idx_srt=hp.trace_start, idx_end=hp.trace_end, start=hp.signal_start, end=hp.signal_end, load_metadata=True)

        seed_init()
        #to avoid repeated preprocessing, it should be done before entering the loop
        if(pre_process == 'scattering'): 
            if(sweep_mode == 'tensorboard'):
                J = sp.J
                Q = sp.Q
            elif(sweep_mode == 'wandb'):
                J = wp.scattering_keyguess['parameters']['J']['values'][0]          #
                Q = wp.scattering_keyguess['parameters']['Q']['values'][0] 
                sweep_config = wp.scattering_keyguess
            scattered_sample = scap.scattering(trcs[:], J, trcs[:].shape[1], Q)     #Process WST in GPU, need to pick out data from GRAM 
            scattered_trcs = scattered_sample.numpy()
        elif(pre_process == 'stft' ):
            if(sweep_mode == 'tensorboard'):
                windows = sp.windows
            elif(sweep_mode == 'wandb'):
                windows = wp.stft_keyguess['parameters']['windows']['values'][0]
                sweep_config = wp.stft_keyguess
            f,t,scattered_trcs = scipy.signal.stft(trcs, nperseg=windows)
        elif(pre_process == 'co'):
            scattered_trcs = trcs
            sweep_config = wp.co_keyguess
        
        if(pre_process == 'stft' or pre_process == 'scattering'):
            sample_num = ( scattered_trcs.shape[1], scattered_trcs.shape[2] )
            print(f'shape is {sample_num}')
        else:
            sample_num = scattered_trcs.shape[1]


        if(file_type == 'sx'):
            plts = read_plts_sx(metadata = metadata, trace_num = hp.trace_end-hp.trace_start)
            cpts = read_cpts_sx(metadata = metadata, trace_num = hp.trace_end-hp.trace_start)
        elif(file_type == 'ascad'):
            plts = read_plts(metadata = metadata)
            cpts = 0
        
        if(sweep_mode == 'tensorboard'):
            keyguess_main()
        elif(sweep_mode == 'wandb'):
            sweep_id = wandb.sweep(sweep = sweep_config, project = project_name) 
            wandb.agent(sweep_id, function = keyguess_main, count = 256)
            



    #parameter sweep part 
    elif(sweep_enable):
        if(pre_process == 'scattering'):
            sweep_config = wp.scattering_sweep
        elif(pre_process == 'stft'):
            sweep_config = wp.stft_sweep
        elif(pre_process == 'co'):
            sweep_config = wp.co_sweep

        if(sweep_mode == 'wandb'):
            sweep_id = wandb.sweep(sweep = sweep_config, project = project_name)
            wandb.agent(sweep_id, function = sweep_main, count = sweep_num)

        elif(sweep_mode == 'tensorboard'):
            keys, para = sp.read_wandb(f"{pre_process}_sweep")
            for i in range(len(para)):
                try:
                    sp.apply_para(keys,para[i])
                    print(f"\n{keys}\n{para[i]}")
                    total_vali_list = sweep_main()
                    with open(f"/home/admin1/Documents/git/SCA_CNN_result/11_21/{i}.txt", 'w') as fp:
                        fp.write(','.join(map(str, para[i])))
                        fp.write('\n')
                        fp.write(','.join(map(str, total_vali_list)))
                    
                except:
                    print('ignore')
    
