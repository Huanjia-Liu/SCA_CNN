class wandb_para():
  
    co_sweep = {
        'method': 'bayes',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            
            'lr': {'max':0.001, 'min':0.00001},             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [1]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [1]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': ['sgd', 'rmsprop', 'adam', 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values':[0]},        #add number to increase wrong key number
            'layer' : {'values': [10]},                  #[2,3,4]
            'kernel_length' : {'values': [4,8,12,16]},
            'kernel_width' : {'values':[2]},                       #[16,24,32,36]
            'dense' : {'values': [1,2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [2,4,8]},
            'channel_2' : {'values': [8,16,32]},
            'channel_3' : {'values': [32,64,96,128]},
            }
        }

    scattering_sweep = {
        'method': 'bayes',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            
            'lr': {'max':0.001, 'min':0.00001},             #{'max':0.001, 'min':0.0001 },{'values' : [0.0004, 0.0002, 0.00008, 0.00005, 0.00003, 0.00001]}
            'Q' : {'values' : [8,12,16,20,24,36,48,52,64]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [2,3,4]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': ['sgd', 'rmsprop', 'adam', 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['MI']},
            'wrong_key': {'values':[0]},        #add number to increase wrong key number
            'layer' : {'values': [7,8,9]},                  #[2,3,4]
            'kernel_length' : {'values': [5]},
            'kernel_width' : {'values':[5]},                       #[16,24,32,36]
            'dense' : {'values': [1,2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [2]},
            'channel_2' : {'values': [10,16,24,32,48,56,64,84,96,128,192,256]},
            'channel_3' : {'values': [10,16,24,32,48,56,64,84,96,128,192,256]},
            'channel_4' : {'values': [2]},
            'channel_5' : {'values': [2]},

            }
        }
    







    
     
    stft_sweep = {
        'method': 'bayes',              #'bayes',
        'name': 'grid',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'max':0.001, 'min':0.00001},             #{'max':0.001, 'min':0.0001 },

    
            'windows' : {'values': [36,40,48,54,64,72,84,96,128]},
            
            'optimizer' : {'values': ['sgd', 'rmsprop', 'adam', 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values':[0]},        #add number to increase wrong key number
            'layer' : {'values': [2,3,4]},                  #[2,3,4]
            'kernel_length' : {'values': [2,3,4,5]},
            'kernel_width' : {'values':[2,3,4,5]},                       #[16,24,32,36]
            'dense' : {'values': [1,2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [2,4,8]},
            'channel_2' : {'values': [8,16,32]},
            'channel_3' : {'values': [32,64,96,128]},
            }
         
        }
    
           
    scattering_keyguess = {
        'method': 'grid',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'values': [0.0005847] },             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [24]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [4]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': [ 'adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values': [x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [2]},                  #[2,3,4]
            'kernel_length' : {'values': [4]},
            'kernel_width' : {'values':[4]},                       #[16,24,32,36]
            'dense' : {'values': [1]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [4]},
            'channel_2' : {'values': [8]},
            'channel_3' : {'values': [64]},
            }
        }
 
    stft_keyguess = {
        'method': 'grid',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'values': [0.0007848] },             #{'max':0.001, 'min':0.0001 },
            'windows' : {'values': [84]},
    
            'optimizer' : {'values': [ 'rmsprop']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values': [x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [7]},                  #[2,3,4]
            'kernel_length' : {'values': [4]},
            'kernel_width' : {'values':[3]},                       #[16,24,32,36]
            'dense' : {'values': [2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [8]},
            'channel_2' : {'values': [32]},
            'channel_3' : {'values': [256]},
            }
        }

    co_keyguess = {
        'method': 'grid',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            
            'lr': {'values':[0.0001]},             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [1]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [1]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': ['adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values':[x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [11]},                  #[2,3,4]
            'kernel_length' : {'values': [16]},
            'kernel_width' : {'values':[2]},                       #[16,24,32,36]
            'dense' : {'values': [2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [2]},
            'channel_2' : {'values': [16]},
            'channel_3' : {'values': [64]},
            'channel_4' : {'values': [2]},
            'channel_5' : {'values': [2]},
            }
        }
               
