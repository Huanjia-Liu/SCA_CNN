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
            'lr':  {'values': [0.0003142] },             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [12]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [6]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': [ 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values': [x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [4]},                  #[2,3,4]
            'kernel_length' : {'values': [4]},
            'kernel_width' : {'values':[3]},                       #[16,24,32,36]
            'dense' : {'values': [1]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [4]},
            'channel_2' : {'values': [32]},
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
            'windows' : {'values': [64]},
    
            'optimizer' : {'values': [ 'sgd']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values': [x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [3]},                  #[2,3,4]
            'kernel_length' : {'values': [4]},
            'kernel_width' : {'values':[3]},                       #[16,24,32,36]
            'dense' : {'values': [2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [8]},
            'channel_2' : {'values': [32]},
            'channel_3' : {'values': [128]},
            }
        }

    co_keyguess = {
        'method': 'grid',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            
            'lr': {'values':[0.0003369]},             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [1]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [1]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': ['adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values':[x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [10]},                  #[2,3,4]
            'kernel_length' : {'values': [16]},
            'kernel_width' : {'values':[2]},                       #[16,24,32,36]
            'dense' : {'values': [2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [2]},
            'channel_2' : {'values': [16]},
            'channel_3' : {'values': [64]},
            }
        }
               
