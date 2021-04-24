from datasets.audiodata import AudioDataset as AudioDataset
from mel2wav.generator import Generator
from mel2wav.tfms_discriminator import Discriminator
from mel2wav.stft import Audio2Mel
from mel2wav.utils import save_sample

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mel2wav.stft_loss import MultiResolutionSTFTLoss

import yaml
import numpy as np
import time
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()      
    parser.add_argument("--cfg", default="configs/cmu.yml", type=Path)
    args = parser.parse_args()
    return args

def trainG(args):
    root = Path(args['logging']['save_path'])
    load_root = Path(args['logging']['load_path']) if args['logging']['load_path'] else None
    root.mkdir(parents=True, exist_ok=True)
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(args['fft']['n_mel_channels'], 
                     args['Generator']['ngf'], 
                     args['Generator']['n_residual_layers'], 
                     ratios=args['Generator']['ratios']).cuda()

    if 'G_path' in args['Generator'] and args['Generator']['G_path'] is not None:
        netG.load_state_dict(torch.load(args['Generator']['G_path'] / "netG.pt"))    
    fft = Audio2Mel(n_mel_channels=args['fft']['n_mel_channels'], 
                    n_fft=args['fft']['n_fft'], 
                    hop_length=args['fft']['hop_length'], 
                    win_length=args['fft']['win_length'], 
                    sampling_rate=args['data']['sampling_rate'],
                    mel_fmin=args['fft']['mel_fmin'],
                    mel_fmax=args['fft']['mel_fmax']).cuda()

    print(netG)
    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=args['optimizer']['lrG'], betas=args['optimizer']['betasG'])    

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))        
        print('checkpoints loaded')

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args['data']['data_path']) / "train_files_inv.txt", 
        segment_length=args['data']['seq_len'], 
        sampling_rate=args['data']['sampling_rate'],
        augment=['amp', 'flip', 'neg']
    )

    test_set = AudioDataset(
        Path(args['data']['data_path']) / "test_files_inv.txt",
        segment_length=args['data']['sampling_rate']*4,
        sampling_rate=args['data']['sampling_rate'],
        augment=None        
    )

    train_loader = DataLoader(train_set, batch_size=args['dataloader']['batch_size'], num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t.cuda()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), args['data']['sampling_rate'], audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=args['data']['sampling_rate'])

        if i == args['logging']['n_test_samples'] - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    mr_stft_loss = MultiResolutionSTFTLoss().cuda()
    for epoch in range(1, args['train']['epochs'] + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t.cuda()
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.cuda())

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()
            
            ###################
            # Train Generator #
            ###################
            loss_G = 0
            sc, sm = mr_stft_loss(x_pred_t, x_t)
            loss_G = args['losses']['lambda_sc']*sc + args['losses']['lambda_sm']*sm
            netG.zero_grad()
            loss_G.backward()
            optG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_G.item(), sc.item(), sm.item(), s_error])

            writer.add_scalar("loss/generator", costs[-1][0], steps)
            writer.add_scalar("loss/convergence", costs[-1][1], steps)
            writer.add_scalar("loss/logmag", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % args['logging']['save_interval'] == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), args['data']['sampling_rate'], pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=args['data']['sampling_rate'],
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")                

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args['logging']['log_interval'] == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args['logging']['log_interval'],
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()

def trainGD(args):
    root = Path(args['logging']['save_path'])
    load_root = Path(args['logging']['load_path']) if args['logging']['load_path'] else None
    root.mkdir(parents=True, exist_ok=True)
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(args['fft']['n_mel_channels'], 
                     args['Generator']['ngf'], 
                     args['Generator']['n_residual_layers'], 
                     ratios=args['Generator']['ratios']).cuda()

    if 'Gpath' in args['train'] and args['train']['Gpath'] is not None:
        netG.load_state_dict(torch.load(Path(args['train']['Gpath']) / "best_netG.pt"))        
    netD = Discriminator(
        args['Discriminator']['num_D'], args['Discriminator']['ndf'], args['Discriminator']['n_layers_D'], args['Discriminator']['downsamp_factor']
    ).cuda()

    fft = Audio2Mel(n_mel_channels=args['fft']['n_mel_channels'], 
                    n_fft=args['fft']['n_fft'], 
                    hop_length=args['fft']['hop_length'], 
                    win_length=args['fft']['win_length'], 
                    sampling_rate=args['data']['sampling_rate'],
                    mel_fmin=args['fft']['mel_fmin'],
                    mel_fmax=args['fft']['mel_fmax']).cuda()

    print(netG)
    print(netD)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=args['optimizer']['lrG'], betas=args['optimizer']['betasG'])
    optD = torch.optim.Adam(netD.parameters(), lr=args['optimizer']['lrD'], betas=args['optimizer']['betasD'])

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))
        print('checkpoints loaded')

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args['data']['data_path']) / "train_files_inv.txt", 
        segment_length=args['data']['seq_len'], 
        sampling_rate=args['data']['sampling_rate'],
        augment=['amp', 'flip', 'neg']
    )

    test_set = AudioDataset(
        Path(args['data']['data_path']) / "test_files_inv.txt",
        segment_length=args['data']['sampling_rate']*4,
        sampling_rate=args['data']['sampling_rate'],
        augment=None,
    )

    train_loader = DataLoader(train_set, batch_size=args['dataloader']['batch_size'], num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1)

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t.cuda()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), args['data']['sampling_rate'], audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=args['data']['sampling_rate'])

        if i == args['logging']['n_test_samples'] - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    mr_stft_loss = MultiResolutionSTFTLoss().cuda()
    for epoch in range(1, args['train']['epochs'] + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t.cuda()
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.cuda())

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.cuda().detach())
            D_real = netD(x_t.cuda())

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.cuda())

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()
                                    
            sc, sm = mr_stft_loss(x_pred_t, x_t)

            netG.zero_grad()
            (loss_G + args['losses']['lambda_sc']*sc + args['losses']['lambda_sm']*sm).backward()            
            optG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_D.item(), loss_G.item(), sc.item(), sm.item(), s_error])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/convergence", costs[-1][2], steps)
            writer.add_scalar("loss/logmag", costs[-1][3], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][4], steps)
            steps += 1

            if steps % args['logging']['save_interval'] == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), args['data']['sampling_rate'], pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=args['data']['sampling_rate'],
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args['logging']['log_interval'] == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args['logging']['log_interval'],
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()

def main():
    args = parse_args()
    with args.cfg.open() as f:
        args = yaml.load(f)    
    if args['train']['mode'] == 'G':
        trainG(args)
    if args['train']['mode'] == 'GD':
        trainGD(args)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()
