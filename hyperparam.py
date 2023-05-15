class hyperparam():
    path = "/data/SCA_data/ASCAD_data/ASCAD_databases/ASCAD.sx"
    trace_start = 0
    trace_end = 2500
    signal_start = 0
    signal_end = 700
    sca_batch = 10000
    train_batch = 512
    vali_batch = 512
    atk_round = 1
    byte = 2  
    power_model = 'hw'         #lsb, hd(hamming distance), hw(hamming weight)

    model_save_path = "/home/admin1/Documents/git/SCA_CNN_result/"
    grad_output = False
    GPU_num = 1



    key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]        #ascad
    #key = [220,250,102,75,101,200,156,56,14,20,62,177,76,142,57,8]                          #for jc's data
    #key = [208, 20, 249, 168]                                                              #aes_rd
    #key = [43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60]          #aes_hd
    #key = [202, 8, 7, 6, 5, 4, 3, 2, 1]                                                    #reassure
    #key = [43, 126, 21, 22]                                                                #aes_rd




    # path = "/data/jc/proj/python/data/SOCure/20221201/Data/SOCure_EM_5M.sx"
    # trace_start = 0
    # trace_end = 5000000
    # signal_start = 0
    # signal_end = 1200
    # sca_batch = 10000
    # train_batch = 3000
    # vali_batch = 3000
    # atk_round = 10
    # byte = 2  
    # power_model = 'hw'         #lsb, hd(hamming distance), hw(hamming weight)



    # path = "/data/SCA_data/ASCAD_data/ASCAD_databases/ASCAD_desync100.h5"
    # trace_start = 0
    # trace_end = 2500
    # signal_start = 0
    # signal_end = 700
    # sca_batch = 2500
    # train_batch = 500
    # vali_batch = 500
    # byte = 2  
