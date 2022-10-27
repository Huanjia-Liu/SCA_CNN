import time
from matplotlib.pyplot import scatter  
import numpy as np

from hyperparam import hyperparams        
#from nn_train_FPGA_first import nn_train
# from resources.Read_Trace_txt import raw_data
from lib.hdf5_files_import import read_multi_plt, read_multi_h5, load_ascad_metadata, load_raw_ascad
from lib.function_initialization import read_plts
from lib.SCA_preprocessing import sca_preprocessing as scap
from mine_train import nn_train
import wandb
import torch
wrong_key = [x for x in range(256)] 
#############################################Scattering#######################################
'''
    sweep_configuration00 = {
    'method': 'grid',              #'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'vali_loss'},
    'parameters': 
    {
        'epochs': {'values': [200]},
        'lr':  {'values': [0.00001]},              #{'max':0.001, 'min':0.0001 },
        'Q' : {'values' : [32]},                            #[16,20,24,28,32]
        'J' : {'values' : [2]},
        #'windows' : {'values': [84]},

        'optimizer' : {'values': ['adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
        'loss_function' : {'values': ['mine_cross']},
        'wrong_key': {'values':wrong_key},        #add number to increase wrong key number
        'layer' : {'values': [6]},                  #[2,3,4]
        'kernel' : {'values': [2]},
        'kernel_width' : {'values':[3]},                       #[16,24,32,36]
        'dense' : {'values': [1]},
        'path' : {'values': ["/data/SCA_data/ASCAD_data/ASCAD_databases/ASCAD.h5"]},               #\ASCAD_desync50.h5"
        'traces_num': {'values': [2500]},
        'project_name': {'values': ['total_50_2.5k_scattering']},
        'channel_1' : {'values': [4]},
        'channel_2' : {'values': [16]},
        'channel_3' : {'values': [32]},
        "train_batch_size" : {'values': [1024]},
        "test_batch_size" : {'values': [1000]}
     
        }
    }
     

'''



sweep_configuration00 = {
    'method': 'bayes',              #'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'vali_loss'},
    'parameters': 
    {
        'epochs': {'values': [200]},
        'lr':  {'max':0.0001, 'min':0.00001 },             #{'max':0.001, 'min':0.0001 },
        'Q' : {'values' : [8,12,16,20,24,36,48,52,64]},                            #[8,12,16,20,24,36,48,52,64]
        'J' : {'values' : [1,2,3,4,5]},
        #'windows' : {'values': [84]},

        'optimizer' : {'values': ['adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
        'loss_function' : {'values': ['mine_cross']},
        'wrong_key': {'values':[0]},        #add number to increase wrong key number
        'layer' : {'values': [7]},                  #[2,3,4]
        'kernel' : {'values': [2]},
        'kernel_width' : {'values':[3]},                       #[16,24,32,36]
        'dense' : {'values': [1]},
        'path' : {'values': ["/data/SCA_data/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5"]},               #\ASCAD_desync50.h5"
        'traces_num': {'values': [2500]},
        'project_name': {'values': ['total_50_2.5k_scattering']},
        'channel_1' : {'values': [10,20,30,40,52,64,72,84,96,128,256,512]},
        'channel_2' : {'values': [64]},
        'channel_3' : {'values': [32]},
        "train_batch_size" : {'values': [500]},
        "vali_batch_size" : {'values': [1000]}
     
        }
    }




#############################################STFT#######################################
'''
    sweep_stft_10 = {
    'method': 'bayes',              #'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'vali_loss'},
    'parameters': 
    {
        'epochs': {'values': [200]},
        'lr':  {'values': [0.00005]},              #{'max':0.001, 'min':0.0001 },

        'windows' : {'values': [36,40,48,54,64,72,84,96,128]},

        'optimizer' : {'values': ['sgd', 'rmsprop', 'adam', 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
        'loss_function' : {'values': ['mine_cross']},
        'wrong_key': {'values':[0]},        #add number to increase wrong key number
        'layer' : {'values': [2]},                  #[2,3,4]
        'kernel' : {'values': [2,3,4,5]},
        'kernel_width' : {'values':[2,3,4,5]},                       #[16,24,32,36]
        'dense' : {'values': [1,2]},
        'path' : {'values': ["/data/SCA_data/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5"]},               #\ASCAD_desync50.h5"
        'traces_num': {'values': [7000]},
        'project_name': {'values': ['stft_6k_50']},
        'channel_1' : {'values': [2,4,8]},
        'channel_2' : {'values': [8,16,32]},
        'channel_3' : {'values': [32]},
        "train_batch_size" : {'values': [1024]},
        "vali_batch_size" : {'values': [1000]}
     
        }
     
    }
'''

##############################################Test_generate##############################



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




def byte_machine( byte, plts,  trcs, sample_num):
    # srt = [30800, 24500, 45000, 32500, 47500, 41000, 37000, 34500, 26500, 39000, 28500, 43000, 20000, 22000, 49000, 18000]
    # end = [33800, 27500, 48000, 35500, 50500, 44000, 40000, 37500, 29500, 42000, 31500, 46000, 23000, 25000, 52000, 21000]
    # srt = 0
    # end = 3000
    for bit in range(1):
        bit = 0
        # traces = load_raw_ascad( ascad_database_file=hp.path_reproduce, num_trc=num_trc, start=srt, end=end )
        nn_train( plts, None, trcs.astype(np.float32), bit, byte, sample_num)


def main():



    wandb.init(project=f"total_test")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache() 



    trcs, metadata = load_raw_ascad(ascad_database_file=wandb.config.path, idx_srt=0, idx_end=wandb.config.traces_num, start=0, end=700, load_metadata=True)
    
    J = wandb.config.J
    Q = wandb.config.Q
    scattered_trcs = scap.scattering( trcs=trcs.astype(np.float32), J=J, M=trcs.shape[1], Q=Q )

    #STFT
    #f,t,scattered_trcs = scipy.signal.stft(trcs, nperseg=wandb.config.windows)

#####################################################
    #scattered_trcs = scattered_trcs.reshape(scattered_trcs.shape[0], -1)
    #scattered_trcs = np.log10( abs(scattered_trcs))



######################################################


    sample_num = ( scattered_trcs.shape[1], scattered_trcs.shape[2] )
#    sample_num = scattered_trcs.shape[1]


    #normalize here
    # if(wandb.config.normalize):
    #    v_min = scattered_trcs.min(axis=2, keepdims=True)
    #    v_max = scattered_trcs.max(axis=2, keepdims=True)

    #    scattered_trcs = (scattered_trcs-v_min)/(v_max-v_min)
    

    plts = read_plts( metadata=metadata )
    
    for byte in range(1):
        byte=2
        byte_machine( byte, plts, scattered_trcs ,sample_num)
        print("byte:", byte, "Done!")


# test code
import time

if "__main__" == __name__:
    sweep_config_list  = [sweep_configuration00]




    for i in range(1):

        start_time = time.time()
        sweep_id = wandb.sweep(sweep=sweep_config_list[i], project=f"2.5k_50_final1")
        wandb.agent(sweep_id, function = main, count=1)

        stop_time = time.time()

        print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))
