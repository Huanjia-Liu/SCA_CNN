import torch

class hyperparams():
    def __init__(self):
        # self.path_trace  = r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ML_non_profiiling_attack\Data\Standard_simulation_10k_AES\traces\*.h5'
        # self.path_plt    = r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ML_non_profiiling_attack\Data\Standard_simulation_10k_AES\plaintext\*.txt'
        # self.path_cpt    = r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ML_non_profiiling_attack\Data\Standard_simulation_10k_AES\ciphertext\*.txt'
        # self.path_trace = r"D:\Data\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD_desync50.h5"
        self.path_trace = "/data/SCA_data/ASCAD_data/ASCAD_databases/ASCAD.h5"

        # self.path_trace = r"C:\JC\Data\ASCAD_raw\ATMega8515_raw_traces.h5"
        # self.path_reproduce = r"C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ASCAD_non_profiling_attack-CAE_BA-norm_Allbytes_customized - Copy\reproduced_data\ASCAD_reproduce_INnorm_selu_20e_12-2-layers_byte0_10000.h5"
        # self.path_plt   = r'Data\Sbox_sim\plaintext\*.txt'
        # self.path_cpt   = r'Data\Sbox_sim\ciphertext\*.txt' 
        # self.path_trace = r'Data\std_nano\NanoV1\traces\*.h5'
        # self.path_plt = r'Data\std_nano\NanoV1\plaintext\*.txt'
        # self.path_cpt = r'Data\std_nano\NanoV1\ciphertext\*.txt'
        # self.path_trace = r'D:\Data\sasebo_org_GF_std_aes\traces\*.h5'
        # self.path_plt   = r'D:\Data\sasebo_org_GF_std_aes\plaintext\*.txt'
        # self.path_cpt   = r'D:\Data\sasebo_org_GF_std_aes\ciphertext\*.txt' 
        # self.path_trace = r'D:\Data\fpgaGF_stdaes_1M\traces\*.h5'
        # self.path_cpt =  r'D:\Data\fpgaGF_stdaes_1M\cpts\*.txt'
        # self.path_plt =  r'D:\Data\fpgaGF_stdaes_1M\plts\*.txt'

        # self.path_trace = r'D:\Data\simulation_MaskedAES\For JC\traces20k\*.h5'
        # self.path_plt =   r'D:\Data\simulation_MaskedAES\For JC\plaintext20k\*.txt'   
        # self.path_cpt = r'D:\Data\simulation_MaskedAES\For JC\ciphertext20k\*.txt'        

        # used by get_labels
        self.atk_round = 1
        # self.byte_pos = 4
        #
        self.train_batch_size =  1000
        self.vali_batch_size = 1000

        self.test_batch_size = 0

        self.numPOIs = 2      # How many POIs do we want?
        self.POIspacing = 5     # How far apart do the POIs have to be?

        self.learning_rate = 0.001
        self.epoch_num = 100

        self.start = 0
        self.end = 700
        self.sample_num = self.end - self.start

        self.hidden1 = 10           
        self.hidden2 = 10
        self.hidden3 = 10  
        self.hidden4 = 200 
        self.hidden5 = 200 
        self.output = 9
    
        self.key_guess_num = 256

        self.train_size = 4000
        self.vali_size = 1000
        self.test_size = 0

        self.ratio_arr = torch.tensor( [1,0.5] ).double()

        self.net_path = 'model/MLP/MSB_byte_1/'

        self.neurongap = 10
        # number of neuron per layer
        self.nnpl1_srt = 128
        self.nnpl2_srt = 64
        self.nnpl3_srt = 64
        self.nnpl1 = 3
        self.nnpl2 = 3
        self.nnpl3 = 3



# test code
import time
def test():
    hp = hyperparams()
    print( hp.epoch_num )

def main():
    test()


if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))
