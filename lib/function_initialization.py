# from resources import Read_Trace_txt as Rtt
from lib.hdf5_files_import import load_ascad
import numpy as np

from lib.aes_cipher.constants import s_row

s_row = np.array(s_row)


# KNOWN_KEYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# print info about key, traces and plaintext
# print( 
#         'key:', 
#         type(KNOWN_KEYS), 
#         np.asarray(KNOWN_KEYS).dtype,
#         'shape:', len(KNOWN_KEYS),
#         'value:', KNOWN_KEYS
        
#     )

# print( 
#         'traces:', 
#         type(raw_data_1.data_train), 
#         raw_data_1.data_train.dtype,
#         'shape:', raw_data_1.data_train.shape
#         # 'example-', raw_data_1.data_train[0]

#     )

# print( 
#         'plaintext:', 
#         type(raw_data_1.plt), 
#         raw_data_1.plt.dtype,
#         'shape:', raw_data_1.plt.shape,
#         'example-', raw_data_1.plt[0]
#     )

# def init_aes():
#     ## convert plaintext strings to integers
#     plt = np.array([])
#     for i in range( len(raw_data_1.plt) ):
#         plt0 = ''.join([''.join(row) for row in raw_data_1.plt[i]])
#         plt0 = int(plt0,16) 
#         plt = np.append(plt, plt0)

    # print( hex(plt0) )

    ## print plaintext as hex values
    # vhex = np.vectorize(hex)
    # print( vhex(plt) )
    # print( np.array2string(vhex(plt), formatter={'int':lambda x: hex(vhex(plt))}) )

    # print( "Initialization for cpa is Done" )
def read_plaintextANDmask(metadata, byte_pos):
    plt = []
    masks = []
    for item in metadata:
        plt.append( item[0] )
        masks.append( item[3][byte_pos] )
    return np.array(plt), np.array(masks)


def read_plts( metadata ):
    plts=[]
    for item in metadata:
        plts.append( item[0] )
    return np.array( plts )

def read_masks( metadata, byte ):
    masks = []
    for item in metadata:
        masks.append( item[3][byte] )
    return np.array( masks )




def init_cpa(atk_round, byte_pos, plt, cpt):


    # KNOWN_KEYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    if (atk_round == 1):
        txt = plt
    elif (atk_round == 10):
        txt = cpt
    else:
        txt = None
        print( "attack cannot proceed unless atk_round = 1 or 10" )

    txt = txt[:, byte_pos]
    
    # disabled when processing ascad data
    if (isinstance(txt[0], str)):
        txt = np.array([ int(element,16) for element in txt ])

    return txt


def init_cpa_hd(atk_round, byte_pos, plt=None, cpt=None):

    # raw_data_1 = Rtt.raw_data( path_trace, path_plt, path_cpt )
    # raw_data_1.read_multi_h5()
    # raw_data_1.read_multi_plt()

    # KNOWN_KEYS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    if (atk_round == 1):
        txt = plt
    elif (atk_round == 10):
        txt = cpt
    else:
        txt = None
        print( "attack cannot proceed unless atk_round = 1 or 10" )
    # print("the whole set:", txt)

    # original byte
    byte1 = txt[:, byte_pos]
    # byte after shift row
    byte2 = txt[:, s_row[byte_pos]]
    # print( "attacked byte", byte_pos, "bytes shape:", txt.shape)
    # txt = ''.join(row) for row in raw_data_1.plt[i]
    print( 'plaintext or ciphertext byte %x:' % byte_pos, byte1 )
    
    # disabled when processing ascad data
    #byte1 = np.array([ int(element,16) for element in byte1 ])
    #byte2 = np.array([ int(element,16) for element in byte2 ])

    return byte1, byte2




# test code
import time

path_trace = r'Data\ASCAD\ASCAD.h5'
# path_plt   = r'Data\Sbox_sim\plaintext\*.txt'
# path_cpt   = r'Data\Sbox_sim\ciphertext\*.txt' 

# raw_data_1 = Rtt.raw_data( path_trace, path_plt, None )
# raw_data_1.read_multi_h5()
# raw_data_1.read_multi_plt()
atk_round = 1
byte_pos = 5



def main():
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad( path_trace, True )
    read_plaintextANDmask(Metadata_attack)
    # print( init_cpa(atk_round=atk_round, byte_pos=byte_pos, plt=plt, cpt=cpt) )


if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))

