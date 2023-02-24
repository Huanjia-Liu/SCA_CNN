from wandb_para import wandb_para as wp
import itertools
class sweep_para():
    J = 4
    Q = 16
    windows = 64
    loss_function = 'mine_cross'
    layer = 3
    optimizer = 'adam'
    lr = 0.0002
    wrong_key = 0
    epochs = 200
    channel_1 = 4
    channel_2 = 16
    channel_3 = 32
    kernel_length = 3
    kernel_width = 3
    dense = 2
    def read_wandb(para_name):
        if(para_name == 'scattering_sweep'):
            para_config = wp.scattering_sweep['parameters']

        elif(para_name == 'stft_sweep'):
            para_config = wp.scattering_sweep['parameters']

        wandb_keys = []
        wandb_values = []

        for keys, values in para_config.items():
            wandb_keys.append(keys)
            wandb_values.append(values['values'])
        para_combination = list(itertools.product(*wandb_values))

        return wandb_keys, para_combination
    
    def apply_para(wandb_keys, para):
        for i in range(len(wandb_keys)):
            try:
                exec(f"sweep_para.{wandb_keys[i]} = {para[i]}")
            except:
                continue



