from lib.function_initialization import init_cpa, init_cpa_hd, read_plaintextANDmask
# from resources import Read_Trace_txt as Rtt
from lib.power_models import power_models
from lib.hdf5_files_import import load_ascad
import numpy as np

import pandas as pd


######################################
# unmasked model
######################################
def get_HammingWeight( atk_round, byte_pos, plt, cpt ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)    ####extract plain text in postion byte
    pm = power_models()
    HW_f_kg = []                            ###power model              ####can be combined together !!!!!!!
    for i in range(256):
        pm.assign_intmv( pm.intermediate(txt,i) )
        HW_f_kg.append( pm.get_hw() )
    print( "get hamming weight done!" )
    return ( np.array( HW_f_kg ) )






# def get_HammingWeight( atk_round, byte_pos, plt, cpt ):
#     txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
#     pm = power_models()
#     HW_f_kg = []
#     for i in range(256):
#         pm.assign_intmv( pm.intermediate(txt,i) )
#         HW_f_kg.append( pm.get_hw() )
#     print( "get hamming weight done!" )
#     return ( np.array( HW_f_kg ) )


def get_MSB( atk_round, byte_pos, plt, cpt ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    MSB_f_kg = []
    for i in range(256):
        pm.assign_intmv( pm.intermediate(txt,i) )
        MSB_f_kg.append( pm.get_msb() )
    print( "get MSB done!" )
    return ( np.array( MSB_f_kg ) )

def get_LSB( atk_round, byte_pos, plt, cpt ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    LSB_f_kg = []
    for i in range(256):
        pm.assign_intmv( pm.intermediate(txt,i) )
        LSB_f_kg.append( pm.get_lsb() )
    print( "get LSB done!" )
    return ( np.array( LSB_f_kg ) )

def get_INTMV( atk_round, byte_pos, plt, cpt ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    INTMV_f_kg = []
    for i in range(256):
        pm.assign_intmv( pm.intermediate(txt,i) )
        INTMV_f_kg.append( pm.get_intmv() )
    print( "get intermediate value done!" )
    return ( np.array( INTMV_f_kg ) )

def get_ZM( atk_round, byte_pos, plt, cpt ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    ZM_f_kg = []
    for i in range(256):
        pm.assign_intmv( pm.sbox_in(txt,i) )
        ZM_f_kg.append( pm.get_zm() )
    print( "get zero value model done!" )
    return ( np.array( ZM_f_kg ) )

# get bit from byte first round HW
def get_BFB( atk_round, byte_pos, plt, cpt, bit_pos ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    BFB_f_kg = []
    for i in range(256):
        pm.assign_intmv( pm.intermediate(txt,i) )
        BFB_f_kg.append( pm.get_bitfrombyte(bit_pos) )
    print( "get BFB done:", bit_pos )
    return ( np.array( BFB_f_kg ) )

# get bit from byte last round HW
def get_BFB_last( atk_round, byte_pos, plt, cpt, bit_pos ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    BFB_f_kg = []
    for i in range(256):
        pm.assign_i_intmv( pm.i_intermediate(txt,i) )
        BFB_f_kg.append( pm.get_bitfrombyte_last(bit_pos) )
    print( "get BFB last done:", bit_pos )
    return ( np.array( BFB_f_kg ) )

# hamming distance of last round input and output
def get_HD_last( atk_round, byte_pos, plt, cpt ):
    byte1, byte2 = init_cpa_hd(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    HD_f_kg = []
    for i in range(256):
        pm.assign_i_intmv( pm.i_intermediate(byte1,i) )
        HD_f_kg.append( pm.get_hd_last( byte2 ) )
    print( "get HD done: byte ", byte_pos  )
    return( np.array( HD_f_kg ) )

def get_BFB_HD_last( atk_round, byte_pos, plt, cpt, bit_pos ):
    byte1, byte2 = init_cpa_hd(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    HDBFB_f_kg = []
    for i in range(256):
        pm.assign_i_intmv( pm.i_intermediate(byte1,i) )
        HDBFB_f_kg.append( pm.get_bitfrombyte_hd( byte2, bit_pos ) )
    print( "get last round HD done: byte ", byte_pos  )
    return( np.array( HDBFB_f_kg ) )

######################################
# masked model
######################################
def get_masked_INTMV( atk_round, byte_pos, plt, cpt, masks ):
    txt = init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt)
    pm = power_models()
    mINTMV_f_kg = []
    for i in range(256):
        pm.assign_intmv( pm.intermediate(txt,i) )
        mINTMV_f_kg.append( pm.get_masked_intmv(masks) )
    print( "get masked intermedia value done!" )
    return ( np.array( mINTMV_f_kg ) )

# test code
import time
# path_trace  = r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ML_non_profiiling_attack\Data\Standard_simulation_10k_AES\traces\*.h5'
# path_plt    = r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ML_non_profiiling_attack\Data\Standard_simulation_10k_AES\plaintext\*.txt'
# path_cpt    = r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\ML_non_profiiling_attack\Data\Standard_simulation_10k_AES\ciphertext\*.txt'
# path_trace  =   r'Data\std_aes_simulation_10k_GF\traces\*.h5' 
# path_plt    =   r'Data\std_aes_simulation_10k_GF\plaintext\*.txt'
# path_cpt    =   r'Data\std_aes_simulation_10k_GF\ciphertext\*.txt'
# FPGA Std AES GF
path_trace = r'Data\ASCAD\ASCAD.h5'
# path_plt   = r'Data\Sbox_sim\plaintext\*.txt'
# path_cpt   = r'Data\Sbox_sim\ciphertext\*.txt' 

atk_round = 1
byte_pos = 2

# raw_data_1 = Rtt.raw_data( path_trace, path_plt, path_cpt )
# raw_data_1.read_multi_h5()
# raw_data_1.read_multi_plt()

KNOWN_KEYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def main():
    # print( "shape of byte %x hamming weight for 256 key guess:" % byte_pos, get_MSB( atk_round, byte_pos, raw_data_1.plt, raw_data_1.cpt ).shape                       )
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad( path_trace, True )
    plt, masks = read_plaintextANDmask(Metadata_profiling)
    labels = get_LSB( atk_round, byte_pos, plt, None )
    labels2 = get_BFB( atk_round, byte_pos, plt, None, bit_pos=0  )
    print(np.array_equal(labels,  labels2))

    # print( "msb", msb.shape, "hw2:", hw2.shape )
    # print( "MSB", MSB.shape, "hw:", hw.shape )
    # check when plt 1st byte=0, whether all the key guess correct or not
    # print( np.array_equal(s, np.expand_dims( MSB[0], axis=0 )  ) )
    # check hw and msb correct or not
    # print( np.array_equal(MSB.T , msb), np.array_equal(hw.T , hw2 ) )


if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))