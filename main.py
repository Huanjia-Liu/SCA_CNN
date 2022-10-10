import time  
import numpy as np

from hyperparam import hyperparams        
#from nn_train_FPGA_first import nn_train
# from resources.Read_Trace_txt import raw_data
from lib.hdf5_files_import import read_multi_plt, read_multi_h5, load_ascad_metadata, load_raw_ascad
from lib.function_initialization import read_plts
from lib.SCA_preprocessing import sca_preprocessing as scap
from mine_train import nn_train
from wandb_config import wandb_config
import torch
def byte_machine( byte, plts, hp, trcs):
    # srt = [30800, 24500, 45000, 32500, 47500, 41000, 37000, 34500, 26500, 39000, 28500, 43000, 20000, 22000, 49000, 18000]
    # end = [33800, 27500, 48000, 35500, 50500, 44000, 40000, 37500, 29500, 42000, 31500, 46000, 23000, 25000, 52000, 21000]
    # srt = 0
    # end = 3000
    for bit in range(1):
        bit = 0
        # traces = load_raw_ascad( ascad_database_file=hp.path_reproduce, num_trc=num_trc, start=srt, end=end )
        nn_train( hp, plts, None, trcs.astype(np.float32), bit, byte)


def main():




    hp = hyperparams()
    
    num_trc = hp.train_size + hp.vali_size
    trcs, metadata = load_raw_ascad(ascad_database_file=hp.path_trace, idx_srt=0, idx_end=num_trc, start=hp.start, end=hp.end, load_metadata=True)
    
    J = wandb_config.J
    Q = wandb_config.Q
    scattered_trcs = scap.scattering( trcs=trcs.astype(np.float32), J=J, M=trcs.shape[1], Q=Q )
    hp.sample_num = ( scattered_trcs.shape[1], scattered_trcs.shape[2] )

    #normalize here
    # if(wandb_config.normalize):
    #    v_min = scattered_trcs.min(axis=2, keepdims=True)
    #    v_max = scattered_trcs.max(axis=2, keepdims=True)

    #    scattered_trcs = (scattered_trcs-v_min)/(v_max-v_min)
    

    plts = read_plts( metadata=metadata )
    
    for byte in range(1):
        byte=2
        byte_machine( byte, plts, hp, scattered_trcs )
        print("byte:", byte, "Done!")


# test code
import time

if "__main__" == __name__:

    # sweep_configuration = {
    # 'method': 'random',
    # 'name': 'sweep',
    # 'metric': {'goal': 'minimize', 'name': 'train_total_loss'},
    # 'parameters': 
    # {
    #     'epochs': {'values': [20000]},
    #     'lr': {'values': [0.001]},
    #     'Q' : {'values' : [8]},
    #     'J' : {'values' : [3]},
    #     'optimizer' : {'values': ['rmsprop']}

    #  }


    # }

    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))
