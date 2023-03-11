import wandb
api = wandb.Api(timeout=19)
project_name = "stft_55k_50_final_time"
runs = api.runs(f'aceleo/{project_name}')
total_time = 0
account = 0
for run in runs:
    total_time += run.history()['_runtime'][199]
    account += 1
print(total_time)