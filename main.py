import time
from matplotlib.pyplot import scatter  
import numpy as np

from hyperparam import hyperparam as hp       
from sweep_para import sweep_para as sp
from wandb_para import wandb_para as wp

from lib.hdf5_files_import import read_multi_plt, read_multi_h5, load_ascad_metadata, load_raw_ascad, load_sx_file, read_plts_sx
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
    # Project is specified by <entity/project-name>
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


def seed_init():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache() 





def byte_machine( byte, plts,  trcs, sample_num, sweep_mode):

    for bit in range(1):
        bit = 0
        nn_train( plts, None, trcs.astype(np.float32), bit, byte, sample_num, sweep_mode)






###############################################################
#   Sweep component:
#           preprocess data (scattering or stft) each runs
###############################################################

def sweep_main():

    global sweep_mode, pre_process 

    seed_init()
    
    if(sweep_mode == 'wandb'):
        
        wandb.init(project=f"total_test")
        trcs, metadata = load_sx_file(sx_file=hp.path, idx_srt=hp.trace_start, idx_end=hp.trace_end, start=hp.signal_start, end=hp.signal_end, load_metadata=True)
        if(pre_process == 'scattering'):
            J = wandb.config.J
            Q = wandb.config.Q
            scattered_trcs = scap.scattering( trcs=trcs.astype(np.float32), J=J, M=trcs.shape[1], Q=Q )
        elif(pre_process == 'stft'):
            windows = wandb.config.windows
            f,t,scattered_trcs = scipy.signal.stft(trcs, nperseg=windows)

        sample_num = ( scattered_trcs.shape[1], scattered_trcs.shape[2] )
        plts = read_plts_sx(metadata = metadata, trace_num = hp.trace_end-hp.trace_start)

    for byte in range(1):
        byte=2
       
        byte_machine( byte, plts, scattered_trcs ,sample_num, sweep_mode)
        print("byte:", byte, "Done!")

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
                byte=2
               
                byte_machine( byte, plts, scattered_trcs ,sample_num, sweep_mode)
                print("byte:", byte, "Done!")
            stop_time = time.time()
            print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))    
        
    elif(sweep_mode == 'wandb'):
        seed_init()
        wandb.init(project='wandbtest')
        for byte in range(1):
            byte = 2

            byte_machine( byte, plts, scattered_trcs, sample_num, sweep_mode)
            print("byte:", byte, "Done!")

        






# test code
import time

if "__main__" == __name__:
    global pre_process, sweep_mode
    pre_process = 'scattering'
    sweep_mode = 'wandb'
    sweep_enable = True

    sweep_num = 100




#key guess part   
    if(not(sweep_enable)):
        trcs, metadata = load_sx_file(sx_file=hp.path, idx_srt=hp.trace_start, idx_end=hp.trace_end, start=hp.signal_start, end=hp.signal_end, load_metadata=True)
        if(pre_process == 'scattering'): 
            if(sweep_mode == 'tensorboard'):
                J = sp.J
                Q = sp.Q
            elif(sweep_mode == 'wandb'):
                J = wp.scattering_keyguess['parameters']['J']['values'][0] 
                Q = wp.scattering_keyguess['parameters']['Q']['values'][0] 
                sweep_config = wp.scattering_keyguess
            scattered_trcs = scap.scattering( trcs = trcs.astype(np.float32), J=J, M=trcs.shape[1], Q=Q )
        elif(pre_process == 'stft' ):
            if(sweep_mode == 'tensorboard'):
                windows = sp.windows
            elif(sweep_mode == 'wandb'):
                windows = wp.scattering_['parameters']['windows']['values'][0]
                sweep_config = wp.stft_keyguess
            f,t,scattered_trcs = scipy.signal.stft(trcs, nperseg=windows)
    
        sample_num = (scattered_trcs.shape[1], scattered_trcs.shape[2]) 
    
        plts = read_plts_sx(metadata = metadata, trace_num = hp.trace_end-hp.trace_start)
        
        if(sweep_mode == 'tensorboard'):
            keyguess_main()
        elif(sweep_mode == 'wandb'):
            sweep_id = wandb.sweep(sweep = sweep_config, project = 'nov16') 
            wandb.agent(sweep_id, function = keyguess_main, count = 256)
            
    elif(sweep_enable):
        if(pre_process == 'scattering'):
            sweep_config = wp.scattering_sweep
        elif(pre_process == 'stft'):
            sweep_config = wp.stft_sweep

        if(sweep_mode == 'wandb'):
            sweep_id = wandb.sweep(sweep = sweep_config, project = 'nov16')
            wandb.agent(sweep_id, function = sweep_main, count = sweep_num)
