from lib.aes_cipher.constants import s_box, is_box
import numpy as np


# constants
# HW = np.array( [bin(n).count("1") for n in range(0, 256)] )
s_box = np.array( s_box )
is_box = np.array( is_box )



class power_models():
    
    def __init__(self):
        self.intmv = None
        self.i_intmv = None
        self.HW = np.array( [bin(n).count("1") for n in range(0, 256)] )
    
    # get the output of sbox( plt, keyguess )
    @staticmethod
    def intermediate(txt_byte, keyguess):
        return s_box[np.bitwise_xor(txt_byte, keyguess)]

    # get the output of inverse_sbox( cpt, keyguess ) 
    @staticmethod
    def i_intermediate( txt_byte, keyguess ):
        return is_box[np.bitwise_xor(txt_byte, keyguess)]
    
    @staticmethod
    def sbox_in(txt_byte, keyguess):
        return np.bitwise_xor(txt_byte, keyguess)

    @staticmethod
    def isbox_in( txt_byte, keyguess ):
        return np.bitwise_xor( txt_byte, keyguess )
    
    def get_masked_intmv(self, masks):
        return np.bitwise_xor( self.intmv, masks )

    def assign_intmv(self, intmv):
        self.intmv = intmv
    
    def assign_i_intmv(self, i_intmv):
        self.i_intmv = i_intmv 

    def get_intmv(self):
        return self.intmv
    
    def get_hw(self):
        return self.HW[self.intmv]
    
    def get_msb(self):
        return ( np.right_shift(self.intmv,7) )
    
    def get_lsb(self):
        return ( np.bitwise_and(self.intmv,1) )

    # wrong! should make pt ^ key = 0
    def get_zm(self):
        self.intmv[self.intmv>0]=1
        return ( self.intmv )
    
    def get_bitfrombyte( self, bit_pos ):
        return (np.bitwise_and(np.right_shift(self.intmv, bit_pos),1))

    def get_bitfrombyte_last( self, bit_pos ):
        return (np.bitwise_and(np.right_shift(self.i_intmv, bit_pos),1))
        

    def get_hd_last( self, byte2 ):
        return ( self.HW[np.bitwise_xor( self.i_intmv, byte2 )] )

    def get_bitfrombyte_hd( self, byte2, bit_pos ):
        return (np.bitwise_and( np.right_shift(np.bitwise_xor( self.i_intmv, byte2 ), bit_pos),1 ))
    
        
    
    # def ...
    # add more models

# test code sbox intput result:
# 10111001 10101011 10101011 00000000 in bin
import time

def tester():      
    A = [[0x12, 0x00, 0x00, 0xAB],[0x12, 0x00, 0x00, 0xAB]]
    A = np.array(A) 

    pm = power_models()
    pm.assign_intmv( pm.sbox_in(A, 0xAB) )
    # print( pm.get_bitfrombyte(1).sum() )
    print( pm.get_lsb() )



if "__main__" == __name__:
    start_time = time.time()

    tester()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))