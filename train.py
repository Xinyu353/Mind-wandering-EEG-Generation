import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from input_noise import generate_pink_noise_torch
from loss import WGP_loss
from plot_signals import plot_losses, plot_losses_smooth, plot_losses_G_smooth
from utils import set_random_seed, weights_init_he
from wgan_cnn import Generator, Discriminator



def normalize_data(data):
    # Normalize to [0, 1]
    data_min, data_max = data.min(), data.max()
    data = (data - data_min) / (data_max - data_min)
    # Normalize to [-1, 1]
    data = data * 2 - 1
    return data, data_min, data_max

#
def denormalize_data(data, real_min, real_max):
    # Scale back to [0, 1]
    data = (data + 1) / 2
    data = data * (real_max - real_min) + real_min
    return data

def train_wgan_gp(dataset, params, device, save_dir):
    # Set random seed for reproducibility
    # set_random_seed(42)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unpack parameters
    lr_g = params['lr_g']
    lr_d = params['lr_d']
    batch_size = params['batch_size']
    lambda_gp = params.get('lambda_gp', 10.0)
    n_critic = params.get('n_critic', 5)
    epochs = params.get('epochs', 100)
    lambda_wd = params.get('lambda_wd', 1)
    lambda_psd = params.get('lambda_psd', 0.5)
    lambda_lambda_erp = params.get('lambda_lambda_erp', 1.0)
    noise_dim = params.get('noise_dim', 100)  # Assuming a default noise dimension


    # Setup TensorBoard writer
    writer = SummaryWriter(f'runs/WGAN_GP/bs{batch_size}_lrG{lr_g}_lrD{lr_d}')

    data_list = [dataset[i] for i in range(len(dataset))]
    data_tensor = torch.stack(data_list)
    dataset_normalized, real_min, real_max = normalize_data(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset_normalized, batch_size=batch_size, shuffle=True, drop_last=True)


    # Data loader
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  drop_last=True)

    # Initialize models
    # generator = G_WGAN(input_dim=noise_dim, hidden_size1=256, hidden_size2=128, hidden_size3=128, output_dim=409, dropout=0).to(device)
    # discriminator = D_WGAN(input_dim=409, hidden_size1=256, hidden_size2=256, output_dim=1, dropout=0).to(device)
    # #
    # generator = Generator(noise_dim, 128, 409).to(device)
    # discriminator = Discriminator(409).to(device)

    generator = Generator(noise_dim, eeg_length=409).to(device)
    # generator = GeneratorWithTransformer(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    # Apply He initialization
    generator.apply(weights_init_he)
    discriminator.apply(weights_init_he)


    loss_fn = WGP_loss(lambda_gp=lambda_gp, lambda_wd = lambda_wd, lambda_psd=lambda_psd, lambda_lambda_erp= lambda_lambda_erp)

    # # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.99), weight_decay=1e-6)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99), weight_decay=1e-6)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.8)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.8)

    # 用于存储损失数据
    d_loss_iterations = []
    g_loss_iterations = []
    d_loss_list, g_loss_list, wd_list = [], [], []
    d_loss_epochs, g_loss_epochs, wd_epochs = [], [], []

    g_loss_wgan_epochs, g_loss_psd_epoch, g_loss_time_epoch = [], [], []

    iter_count = 0

    # Training loop
    timestampplot = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()
    run_time = []
    for epoch in range(epochs):
        total_d_loss, total_g_loss, total_wd = 0, 0,0
        total_g_loss_wgan, total_g_loss_psd, total_g_loss_time = 0, 0, 0
        num_d_batches, num_g_updates = 0, 0
        real_data_epoch = []
        fake_data_epoch = []
        g_loss = None
        wd = None

        g_loss_wgan = None
        g_loss_psd = None
        g_loss_time = None
        for iter, real_samples in enumerate(dataloader):
            # print(f"iter={iter}")
            real_samples = real_samples.to(device)
            real_data_epoch.append(real_samples)
            # noise = torch.randn(batch_size, noise_dim, device=device)
            noise = generate_pink_noise_torch(batch_size, noise_dim, alpha=1.0, lowcut=0.5, highcut=40.0, fs=256, device=device)

            # # Calculate FFT for each sample in batch
            # noise_fft = torch.fft.fft(noise, dim=-1)
            # # Get power spectrum (magnitude squared)
            # power_spectrum = torch.abs(noise_fft) ** 2
            # # Average power spectrum across batch
            # avg_power_spectrum = power_spectrum.mean(dim=0).cpu().numpy()
            #
            # # Frequency vector
            # freqs = np.fft.fftfreq(noise_dim)
            #
            # # Plot average power spectrum
            # plt.loglog(freqs[:noise_dim // 2], avg_power_spectrum[:noise_dim // 2])
            # plt.xlabel('Frequency')
            # plt.ylabel('Power')
            # plt.title('Average Power Spectrum of Batch Noise')
            # plt.show()


            # Update discriminator
            optimizer_d.zero_grad()
            fake_samples = generator(noise)
            # 调试信息
            # print(f"real_samples shape: {real_samples.shape}")
            # print(f"fake_samples shape: {fake_samples.shape}")
            # print()


            d_loss, wd = loss_fn.d_loss(discriminator, real_samples, fake_samples)
            d_loss.backward()
            optimizer_d.step()
            total_d_loss += d_loss.item()
            total_wd += wd.item()
            num_d_batches += 1


            if ((iter + 1) % n_critic) == 0:
                # Update generator
                optimizer_g.zero_grad()
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_samples = generator(noise)
                fake_data_epoch.append(fake_samples)
                # print(f"Generator fake_samples shape: {fake_samples.shape}")

                # if iter_count % 20 == 0 and iter_count != 0:
                #     loss_fn.lambda_psd *= 1.01  # 逐步增加lambda_psd
                #     loss_fn.lambda_time *= 1.01

                # g_loss = loss_fn.g_loss(discriminator, fake_samples)
                g_loss, g_loss_wgan, g_loss_psd, g_loss_time = loss_fn.g_loss(discriminator, fake_samples, real_samples, 256)
                g_loss.backward()
                optimizer_g.step()

                total_g_loss += g_loss.item()
                total_g_loss_wgan += g_loss_wgan.item()
                total_g_loss_psd += g_loss_psd.item()
                total_g_loss_time += g_loss_time.item()

                num_g_updates += 1

                g_loss_list.append(g_loss.item())
                g_loss_iterations.append(iter_count)
                wd_list.append(wd.item())
                d_loss_list.append(d_loss.item())
                d_loss_iterations.append(iter_count)

                iter_count += 1

            # Logging
            if g_loss is not None:
                writer.add_scalars('Loss', {'discriminator': d_loss.item(),
                                            'generator': g_loss.item()}, epoch * len(dataloader) + iter)
        avg_d_loss = total_d_loss / num_d_batches
        avg_g_loss = total_g_loss / num_g_updates if num_g_updates > 0 else 0
        avg_wd = total_wd / num_d_batches

        avg_g_loss_wgan = total_g_loss_wgan / num_g_updates if num_g_updates > 0 else 0
        avg_g_loss_psd = total_g_loss_psd / num_g_updates if num_g_updates > 0 else 0
        avg_g_loss_ar = total_g_loss_time / num_g_updates if num_g_updates > 0 else 0

        # 保存每个epoch的平均损失
        d_loss_epochs.append(avg_d_loss)
        g_loss_epochs.append(avg_g_loss)
        wd_epochs.append(avg_wd)

        g_loss_wgan_epochs.append(avg_g_loss_wgan)
        g_loss_psd_epoch.append(avg_g_loss_psd)
        g_loss_time_epoch.append(avg_g_loss_ar)

        # Logging average losses
        writer.add_scalars('Average Loss', {'discriminator': avg_d_loss, 'generator': avg_g_loss}, epoch)

        # Epoch completion log
        print(f"Epoch {epoch + 1}/{epochs}, Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}, WD(GP) Loss: {avg_g_loss_wgan:.4f}, PSD Loss: {avg_g_loss_psd:.4f}, AR Loss: {avg_g_loss_ar:.4f}")

        # 这里添加学习率调度器的调用
        scheduler_g.step()
        scheduler_d.step()

        fake_data_epoch_tensor = torch.cat(fake_data_epoch, dim=0)

        # Save model checkpoints
        if (epoch != 0 and (epoch+1) % 50 == 0) or epoch == epochs - 1:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_{timestamp}.pth")
            
            # Save model state and training information
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
                'loss': {
                    'd_loss': avg_d_loss,
                    'g_loss': avg_g_loss,
                    'wd_loss': avg_wd,
                    'g_loss_wgan': avg_g_loss_wgan,
                    'g_loss_psd': avg_g_loss_psd,
                    'g_loss_time': avg_g_loss_ar
                },
                'params': params,  # Save training parameters
                'real_min': real_min,  # Save normalization parameters
                'real_max': real_max
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

            fake_data_df = pd.DataFrame(fake_data_epoch_tensor.detach().cpu().numpy())

            save_path = f"./samples_fake/WGAN-GP/{timestampplot}_bs{batch_size}_lrG{lr_g}_lrD{lr_d}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plot_fold_samples = f"Epoch{epoch + 1}.csv"
            full_save_path = os.path.join(save_path, plot_fold_samples)
            if not os.path.exists(os.path.dirname(full_save_path)):
                os.makedirs(os.path.dirname(full_save_path))

            fake_data_denormalized = denormalize_data(fake_data_epoch_tensor, real_min, real_max)
            fake_data_df = pd.DataFrame(fake_data_denormalized.detach().cpu().numpy())

            fake_data_df.to_csv(full_save_path, index=False, header=False)

            end_time = time.time()
            run_time_seconds = end_time - start_time
            time_epoch = np.array([lr_g, lr_d, batch_size, epoch+1, run_time_seconds])
            run_time.append(time_epoch)


            plot_name = f"Epoch{epoch + 1}.png"
            plot_fold = f"{timestampplot}_bs{batch_size}_lrG{lr_g}_lrD{lr_d}"
            plot_path = f"./training_plot/WGAN-GP/"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

    plot_losses(d_loss_epochs, g_loss_epochs, plot_fold, plot_path)
    plot_losses_smooth(d_loss_epochs, g_loss_epochs, plot_fold, plot_path)
    plot_losses_G_smooth(g_loss_epochs, g_loss_wgan_epochs, g_loss_psd_epoch, g_loss_time_epoch, plot_fold, plot_path)

    compressed_file_name = f"{plot_fold}.zip"
    compressed_file_path = os.path.join(plot_path, compressed_file_name)
    shutil.make_archive(base_name=compressed_file_path.replace('.zip', ''), format='zip', root_dir=plot_path,
                        base_dir=plot_fold)
    if os.path.exists(compressed_file_path):
        shutil.rmtree(os.path.join(plot_path, plot_fold))
    else:
        print(f"Compression failed, original folder {plot_fold} not removed.")


    writer.close()
    # print(f"Training completed. Models saved in {save_dir}.")

    return d_loss_epochs, g_loss_epochs,  run_time