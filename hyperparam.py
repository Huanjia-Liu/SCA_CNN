import torch

class hyperparams():
    def __init__(self):



        self.train_batch_size =  1024
        self.vali_batch_size = 1000

        self.test_batch_size = 0

        self.numPOIs = 2      # How many POIs do we want?
        self.POIspacing = 5     # How far apart do the POIs have to be?



        self.start = 0
        self.end = 700
        self.sample_num = self.end - self.start

        self.output = 1
    
        self.key_guess_num = 256

        self.train_size = 4000
        self.vali_size = 1000
        self.test_size = 0

        self.ratio_arr = torch.tensor( [1,0.5] ).double()







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