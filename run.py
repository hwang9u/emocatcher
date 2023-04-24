#########################################################################
## Train EmoCatcher for Speech Emotion Recognition on RAVDESS dataset  ##
#########################################################################

import torch
from torch.utils.data import DataLoader
from emocatcher.trainer import train, evaluate
from adabelief_pytorch import AdaBelief
from emocatcher.utils.env import create_config, save_model, create_folder
from emocatcher.utils.criterion import LabelSmoothingLoss
from emocatcher.model import EmoCatcher
from emocatcher.dataset import RAVDESSMelS
from emocatcher.dataloader import get_meta, split_meta, mels2batch
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

config = create_config()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') ## I tested on the CPU

# Extract meta information from .wav file name
meta = get_meta(config.dataset_dir)

# Mel spectrogram parameters
mel_kwargs = {"n_mels":64, "fmax": 6000, "win_length": 2048, "hop_length" :2048//3*2, "n_fft": 2048*2}

if config.kfcv:    
    # Stratified 5-Fold CV
    splitted_meta= split_meta(meta, kfcv=True, n_splits=5, target='emotion', random_state=config.random_state)
else:
    # Hold-out 8:2
    splitted_meta= split_meta(meta=meta, kfcv = False, test_ratio=0.2, stratify=True, target='emotion', random_state=config.random_state)
    
print('device: {}'.format(device))
print('OUTPUT_PATH: {}'.format(config.out_dir))

for k, (tr, ts) in enumerate(splitted_meta):
    if config.kfcv:
        # K-Fold CV
        print(f'----------- Fold {k+1} -----------')    
        train_meta = meta.iloc[tr, :]
        test_meta = meta.iloc[ts, :]
    else:
        # Hold-out
        train_meta, test_meta = splitted_meta
    train_dset = RAVDESSMelS(dataset_dir= config.dataset_dir , mel_specs_kwargs=mel_kwargs, speech_list = train_meta.speech_list.tolist())
    test_dset = RAVDESSMelS(dataset_dir= config.dataset_dir , mel_specs_kwargs=mel_kwargs,speech_list=test_meta.speech_list.tolist())

    # Train/test dataloader
    train_dloader = DataLoader(train_dset, batch_size=config.batch_size, collate_fn=mels2batch, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=64, collate_fn=mels2batch, shuffle=False)

    # Define model
    model = EmoCatcher(input_dim = mel_kwargs['n_mels'], hidden_dim = 128,kernel_size= 3, num_classes=8).to(device)
    crit = LabelSmoothingLoss(n_classes=8, smoothing=.1, dim=-1)
    optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decay=1e-7, weight_decouple = False, rectify = False, print_change_log=False) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.7, patience=2, cooldown = 3,  verbose=True, min_lr=1e-6, mode = 'min', threshold=0.001)
    
    MODEL_SAVE_DIR = os.path.join(config.out_dir, f'model/fold{k+1}/' if config.kfcv else 'model/') 
    LOG_OUT_DIR = os.path.join(config.out_dir, 'logs/')
    LOG_FILENAME = os.path.join(LOG_OUT_DIR, f'fold{k+1}_{config.log}' if config.kfcv else config.log)
    N_EPOCHS = config.n_epochs
    create_folder(config.out_dir)
    create_folder(MODEL_SAVE_DIR)
    create_folder(LOG_OUT_DIR)
    print('LOG_OUT_PATH: {}'.format(LOG_FILENAME))
    print('MODEL_OUT_PATH: {}'.format(os.path.join(MODEL_SAVE_DIR, config.model_name)))


    # Training
    print('Training {} Epochs'.format(N_EPOCHS))
    train_loss_list = [] ; test_loss_list = []
    train_acc_list = [] ; test_acc_list = []

    best_test_loss = torch.inf
    best_test_acc = 0


    for e in range(N_EPOCHS):
        train_loss, train_acc = train(train_dloader, model, optimizer, crit,e+1, device = device)
        test_loss, test_acc = evaluate(test_dloader, model, crit, e+1, device = device)
        
        f = open(f"{LOG_FILENAME}", "a")
        epoch_result = "[Epoch {e}] train Loss: {trainL:.5f} Accuracy: {trainAcc:.4f} | test Loss: {testL:.5f} Accuracy: {testAcc:.4f} ".format(e = e+1, trainL = train_loss, trainAcc = train_acc, testL = test_loss, testAcc = test_acc)
        print(epoch_result, file = f)
        print(epoch_result)
        
        scheduler.step(train_loss)
        
        train_loss_list.append(train_loss); test_loss_list.append(test_loss)
        train_acc_list.append(train_acc); test_acc_list.append(test_acc)
        f.close()
        
        if (test_acc > best_test_acc):
            save_model(MODEL_SAVE_DIR, test_loss = test_loss, test_acc = test_acc, model = model, model_name=config.model_name)
            best_test_acc = test_acc
            best_test_loss = test_loss

        elif ((best_test_acc - test_acc) <1e-3) & (test_loss < best_test_loss):
            save_model(MODEL_SAVE_DIR, test_loss = test_loss, test_acc = test_acc, model = model, model_name=config.model_name)
            best_test_acc = test_acc
            best_test_loss = test_loss
        else:
            pass


    # Save training log
    metric_log_dict = {'train': {
        'acc': train_acc_list,
        'loss': train_loss_list
    },
    'test': {
        'acc': test_acc_list,
        'loss': test_loss_list
    }
    }

    with open(os.path.join(LOG_OUT_DIR, (f'fold{k+1}_' if config.kfcv else '') + 'metric_log_dict.pkl'), 'wb') as f:
        pickle.dump(metric_log_dict, f)
        
        
    # Last epoch
    save_model(MODEL_SAVE_DIR, test_loss = test_loss, test_acc = test_acc, model = model, model_name="last_epoch", init=False)
    if config.kfcv == False:
        break