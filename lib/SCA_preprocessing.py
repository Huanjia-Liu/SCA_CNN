# import sys
# sys.path.insert(1, r'C:\Users\jceeto\OneDrive - Nanyang Technological University\PhD_July30_2019\Projects\Python_VScode\Traditional_SCA\Preprocess')
from lib.hdf5_files_import import read_multi_h5, read_multi_plt
from lib.function_initialization import init_cpa

# from tftb.generators import fmlin
# from tftb.processing.linear import ShortTimeFourierTransform
import matplotlib.pyplot as plt
from scipy.signal import hamming
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kymatio.numpy import Scattering1D


class sca_preprocessing():

    def __init__(self):
        self.processed_trcs = []
####################################################### POIs Selection ####################################################33
    # Variance of input mean POis selection
    @staticmethod
    def VOM_POI( traces, txts, numGrps, referencing=False, normalize=False ):
        if ( normalize==True ):
            trc_mean = np.average( traces, axis=0 )
            trc_std = np.std( traces, axis=0 )
            traces = (traces - trc_mean)/trc_std 

        grps = [[] for _ in range(numGrps)]
        grps_mean = [[] for _ in range(numGrps)]
        if ( referencing == True ):
            grps_ref = np.zeros( traces.shape[0] ).astype( np.int64 )
            for index in range( numGrps ):
                indices = np.where( txts==index )[0]
                grps_ref[indices] = index
                grps[index] = np.take( traces, indices, axis=0 )
                grps[index] = np.array( grps[index] )

                # plt.plot( grps[index].T )
                # plt.show()

                grps_mean[index] = np.average( grps[index], axis=0 )

                # plt.plot( grps_mean[index], color='black', linewidth=3 )
                # plt.show()

                plt.plot( grps_mean[index] )
            grps_mean = np.array( grps_mean )

            return grps_mean, grps_ref #, np.var( grps_mean, axis=0 )
        
        else:

            for index in range( numGrps ):
                indices = np.where( txts==index )[0]
                grps[index] = np.take( traces, indices, axis=0 )
                grps[index] = np.array( grps[index] )

                # plt.plot( grps[index].T )

                grps_mean[index] = np.average( grps[index], axis=0 )

                # plt.plot( grps_mean[index], color='black', linewidth=3 )
                # plt.show()

            grps_mean = np.array( grps_mean )

            return grps_mean #, np.var( grps_mean, axis=0 )
    
    @staticmethod
    def VOM_POI_bit( traces, txts, numGrps, referencing=False, normalize=False ):
        if ( normalize==True ):
            trc_mean = np.average( traces, axis=0 )
            trc_std = np.std( traces, axis=0 )
            traces = (traces - trc_mean)/trc_std 

        grps = [[] for _ in range(numGrps)]
        grps_mean = [[] for _ in range(numGrps)]
        if ( referencing == True ):
            grps_ref = np.zeros( traces.shape[0] ).astype( np.int64 )
            for index in range( numGrps ):
                indices = np.where( txts==index )[0]
                grps_ref[indices] = index
                grps[index] = np.take( traces, indices, axis=0 )
                grps[index] = np.array( grps[index] )

                # plt.plot( grps[index].T )
                # plt.show()

                grps_mean[index] = np.average( grps[index], axis=0 )

                # plt.plot( grps_mean[index], color='black', linewidth=3 )
                # plt.show()

                plt.plot( grps_mean[index] )
            grps_mean = np.array( grps_mean )

            return grps_mean, grps_ref #, np.var( grps_mean, axis=0 )
        
        else:

            for index in range( numGrps ):
                indices = np.where( txts==index )[0]
                grps[index] = np.take( traces, indices, axis=0 )
                grps[index] = np.array( grps[index] )

                # plt.plot( grps[index].T )

                grps_mean[index] = np.average( grps[index], axis=0 )

                # plt.plot( grps_mean[index], color='black', linewidth=3 )
                # plt.show()

            grps_mean = np.array( grps_mean )

            return grps_mean #, np.var( grps_mean, axis=0 )
    

####################################################### POIs Selection ####################################################
####################################################### Data Normalization ################################################
    @staticmethod
    def trcs_centrolize( traces ):
        trc_mean = np.average( traces, axis=0 )
        return ( traces - trc_mean )
    
    @staticmethod
    def trcs_scaled_centrolize( traces, parameter=False ):
        trc_mean = np.average( traces, axis=0 )
        trc_std = np.std( traces, axis=0 )
        if( parameter==False ):
            return ( (traces - trc_mean)/trc_std )
        else:
            return ( (traces - trc_mean)/trc_std ), trc_mean, trc_std

    @staticmethod
    def trcs_scaled_centrolize_agmt( traces, mean, std ):
        return ( (traces - mean)/std )

    @staticmethod
    def instance_norm( traces ):
        trc_mean = np.average( traces, axis=1 ).reshape( traces.shape[0], 1 )
        trc_std = np.std( traces, axis=1 ).reshape( traces.shape[0], 1 )
        return ( (traces - trc_mean)/trc_std )

    @staticmethod
    def scattering(trcs, J, M, Q):
        S = Scattering1D(J, M, Q)
        Sx = S.scattering(trcs)

        return Sx
####################################################### Data Normalization ################################################
####################################################### short time fourier transform ################################################
    # @staticmethod
    # def stft( traces, n_fbins ):
    #     n_fbins_half = int(n_fbins/2)
    #     window = hamming(33)
    #     tfr, _, _ = ShortTimeFourierTransform(traces, timestamps=None, n_fbins=n_fbins, fwindow=window).run()
    #     tfr = tfr[int(n_fbins_half):, :]
    #     tfr = np.abs(tfr) ** 2
    #     # plot
    #     t = np.arange(tfr.shape[1])
    #     f = np.linspace(0, 0.5, tfr.shape[0])
    #     T, F = np.meshgrid(t, f)

    #     fig, axScatter = plt.subplots(figsize=(10, 8))
    #     axScatter.contour(T, F, tfr, 10)
    #     axScatter.grid(True)
    #     axScatter.set_title('Squared modulus of STFT')
    #     axScatter.set_ylabel('Frequency')
    #     axScatter.yaxis.set_label_position("right")
    #     axScatter.set_xlabel('Time')
    #     divider = make_axes_locatable(axScatter)
    #     axTime = divider.append_axes("top", 1.2, pad=0.5)
    #     axFreq = divider.append_axes("left", 1.2, pad=0.5)
    #     axTime.plot(np.real(traces))
    #     axTime.set_xticklabels([])
    #     axTime.set_xlim(0, traces.shape[0])
    #     axTime.set_ylabel('Real part')
    #     axTime.set_title('Signal in time')
    #     axTime.grid(True)
    #     axFreq.plot((abs(np.fft.fftshift(np.fft.fft(traces))) ** 2)[::-1][:n_fbins_half], f[:n_fbins_half])
    #     axFreq.set_yticklabels([])
    #     axFreq.set_xticklabels([])
    #     axFreq.invert_xaxis()
    #     axFreq.set_ylabel('Spectrum')
    #     axFreq.grid(True)
    #     plt.show()
    #     return ( tfr )
  
####################################################### short time fourier transform ################################################
####################################################### Higher order Preprocessing ########################################
    # # second order absolute difference
    # @staticmethod
    # def SecondO_abd( traces ):
        
####################################################### Higher order Preprocessing ########################################


import matplotlib.pyplot as plt


def main():
    num_trc = 1000
    srt = 8000
    end = 9000
    numGrps = 256

    # cpt_file = r"D:\Data\Arty-FPGA_SaseboGF-StdAES_EM-1M\cpts\*.txt"
    # traces_file = r"D:\Data\Arty-FPGA_SaseboGF-StdAES_EM-1M\traces\*.h5"
    # cpt_file = r"D:\Data\Masked_Sbox_lastR_200k_v4\cpts\*.txt"
    # traces_file = r"D:\Data\Masked_Sbox_lastR_200k_v4\traces\*.h5"
    cpt_file = r'D:\JC\Data\Fully_masked_AES-JS\cpts\*.txt'
    traces_file = r"D:\JC\Data\Fully_masked_AES-JS\trcs\*.h5"
    cpts = read_multi_plt( cpt_file, num_trc )
    traces = read_multi_h5( traces_file, num_trc, srt, end )
  

    for byte in range(1):
        byte=5
        txt = init_cpa(atk_round=10, byte_pos=byte, plt=None, cpt=cpts)
        grps_mean, grps_ref = sca_preprocessing.VOM_POI( traces, txt, numGrps, referencing=True )
    

        ###################################################################################
        # tfr = sca_preprocessing.stft( traces[0], 346 )
        # plt.plot( tfr, color='red' )
        # plt.title(f'Maksed Sbox byte{byte} trc:{num_trc}')
        ###################################################################################

        # figure = plt.gcf()
        # figure.set_size_inches(32, 18)
        # plt.savefig(f'Maksed Sbox byte{byte}', bbox_inches='tight')
        # plt.close()
        # plt.show()

        # processed_trc = sca_preprocessing.trcs_centrolize( traces )
        # plt.plot( traces[0], color="blue" )
        # plt.plot( processed_trc[0], color="red" )
        # plt.show()
  


import time 

if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))

        
        
        



        


