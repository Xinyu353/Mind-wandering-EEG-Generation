import os
import pandas as pd
import torch
import time
from train import train_wgan_gp
from dataset import MyDataset


param_grid = []
for lr_g in [0.05]: #0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001
    # 遍历lr_d
    for lr_d in [0.001]: #0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001

        for batch_size in [32]:
            epochs = 1000
            if lr_g in [0.0005, 0.0001] and lr_d in [0.0005, 0.0001]:
                epochs = 400

            param_grid.append({
                'gan_type': "wgan-gp",
                'lr_g': lr_g,
                'lr_d': lr_d,
                'batch_size': batch_size,
                'epochs': epochs,
                'noise_dim': 300,
                'lambda_gp': 10,
                'n_critic': 5,
                'lambda_wd': 1,
                'lambda_psd': 0,
                'lambda_lambda_erp': 2
            })


dataset = MyDataset(csv_file='ot_vs_c_Pz_4753_train.csv')

times = time.strftime("%Y%m%d-%H%M%S")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

save_dir = f"./trained_models/WGAN-GP/ot_pre"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

run_time = []
for params in param_grid:
    # print(f"Training with params: {params}")
    _, _, time = train_wgan_gp(dataset, params, device, save_dir)
    run_time.extend(time)

runs_df = pd.DataFrame(run_time, columns=['lr_g', 'lr_d', 'batch_size', 'epochs', 'time/s'])
csv_file_path = "rt_wgangp_mw.csv"
if os.path.exists(csv_file_path):
    header = False
else:
    header = True
runs_df.to_csv(csv_file_path, mode='a', index=False, header=header)