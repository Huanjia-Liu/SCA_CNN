class wandb_para():
  
    scattering_sweep = {
        'method': 'bayes',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'max':0.001, 'min':0.0001 },             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [8,12,16,20,24,36,48,52,64]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [2,3,4]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': ['sgd', 'rmsprop', 'adam', 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values':[0]},        #add number to increase wrong key number
            'layer' : {'values': [2,3,4]},                  #[2,3,4]
            'kernel_length' : {'values': [2,3,4,8,12,16]},
            'kernel_width' : {'values':[2,3,4,8,12,16]},                       #[16,24,32,36]
            'dense' : {'values': [1,2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [2,4,8]},
            'channel_2' : {'values': [8,16,32]},
            'channel_3' : {'values': [32,64]},
            }
        }
    
    
     
    stft_sweep = {
        'method': 'bayes',              #'bayes',
        'name': 'grid',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'max':0.0002, 'min':0.00002 } ,           #{'max':0.001, 'min':0.0001 },
    
            'windows' : {'values': [36,40,48,54,64,72,84,96,128]},
    
            'optimizer' : {'values': ['sgd', 'rmsprop', 'adam', 'nadam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values':[0]},        #add number to increase wrong key number
            'layer' : {'values': [2,3,4]},                  #[2,3,4]
            'kernel' : {'values': [2,3,4,5,6,7,8]},
            'kernel_width' : {'values':[2,3,4,5,6,7,8]},                       #[16,24,32,36]
            'dense' : {'values': [1,2]},
            'project_name': {'values': ['stft_9k_50']},
            'channel_1' : {'values': [2,4,8]},
            'channel_2' : {'values': [8,16]},
            'mlp': {'values': [10,20,30,40,54,64,72,84,96]},
            'channel_3' : {'values': [32,64]},
            "train_batch_size" : {'values': [1024]},
            "s_step" : {'values': [20,30,40,50,60,70]},
         
            }
         
        }
    
           
    scattering_keyguess = {
        'method': 'grid',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'values': [0.0001] },             #{'max':0.001, 'min':0.0001 },
            'Q' : {'values' : [12]},                            #[8,12,16,20,24,36,48,52,64]
            'J' : {'values' : [2]},
            #'windows' : {'values': [84]},
    
            'optimizer' : {'values': [ 'adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values': [x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [4]},                  #[2,3,4]
            'kernel_length' : {'values': [3]},
            'kernel_width' : {'values':[3]},                       #[16,24,32,36]
            'dense' : {'values': [2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [4]},
            'channel_2' : {'values': [16]},
            'channel_3' : {'values': [32]},
            }
        }
 
    stft_keyguess = {
        'method': 'grid',              #'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'vali_loss'},
        'parameters': 
        {
            'epochs': {'values': [200]},
            'lr':  {'values': [0.0001] },             #{'max':0.001, 'min':0.0001 },
            'windows' : {'values': [84]},
    
            'optimizer' : {'values': [ 'adam']},                         #['sgd', 'rmsprop', 'adam', 'nadam']},
            'loss_function' : {'values': ['mine_cross']},
            'wrong_key': {'values': [x for x in range(256)]},        #add number to increase wrong key number
            'layer' : {'values': [4]},                  #[2,3,4]
            'kernel_length' : {'values': [3]},
            'kernel_width' : {'values':[3]},                       #[16,24,32,36]
            'dense' : {'values': [2]},
            'project_name': {'values': ['scattering_2.5k_100']},
            'channel_1' : {'values': [4]},
            'channel_2' : {'values': [16]},
            'channel_3' : {'values': [32]},
            }
        }
               
