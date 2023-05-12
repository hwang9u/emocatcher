import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from utils.audio_tools import gvad


def get_meta(dataset_dir):
    speech_list = glob.glob(os.path.join(dataset_dir, '*/*.wav'))
    features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
    meta_info = pd.DataFrame(columns=features)
    for fpath in speech_list:
        fn = fpath.split("/")[-1].split(".")[0]
        
        speech_info = {f:[c] for c, f in zip(fn.split("-"), features)}
        meta_info = pd.concat( (meta_info, pd.DataFrame(speech_info)), axis=0)
    meta_info.insert(0, "speech_list", speech_list)
    return meta_info


def split_meta(meta, kfcv=False, n_splits = 5,test_ratio = 0.2, stratify = True, target = None, verbose = True, random_state = None): 
    if kfcv == True:
        skf = StratifiedKFold(n_splits= n_splits, random_state=random_state, shuffle=True)
        if verbose:
            N = len(meta)
            print('size: {}, {}'.format( int(N*(1 - 1/n_splits)), int(N*1/n_splits) ))
            print('ratio: {}, {}'.format(1 - 1/n_splits, 1/n_splits))
        return skf.split(meta[ ['speech_list', target]] , meta[target] )
    else:    
        train_meta, test_meta = train_test_split(meta[ ['speech_list', target] ],
                                                    stratify = meta[target] if stratify else None,
                                                    shuffle = True,
                                                    test_size = test_ratio,
                                                    random_state = random_state)
        if verbose:
            N = len(meta)
            print('size: {}, {}'.format(len(train_meta), len(test_meta)) )
            print('ratio: {}, {}'.format(len(train_meta)/N, len(test_meta)/N))
        return train_meta, test_meta




def mels2batch(batch):
    mel_specs = []
    info_list = []
    L_list = []
    for mel_spec, sr, info, fn in batch:
        info_list.append(info)       
        L_list.append(mel_spec.shape[1])
        mel_specs.append( mel_spec.transpose(0,1).contiguous())
    padded_mel_specs = nn.utils.rnn.pad_sequence( mel_specs, batch_first = True, padding_value= 0).unsqueeze(1).transpose(2,3).contiguous()
    return padded_mel_specs, torch.Tensor(pd.DataFrame(info_list).emotion.values.astype('int16')-1).long(), torch.tensor(L_list)


def sig2batch(batch, mel_specs_kwargs = {}):
    mel_specs = []
    info_list = []
    L_list = []
    for sig, sr, info, fn in batch:
        info_list.append(info)
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(sig, sr= sr, **mel_specs_kwargs))
        va_point = gvad(librosa.db_to_amplitude(mel_spec)**2)
        mel_spec = mel_spec[:, va_point[0]:va_point[1]]
        L_list.append(mel_spec.shape[1]) 
        mel_specs.append( torch.Tensor(mel_spec).transpose(0,1))
    padded_mel_specs = nn.utils.rnn.pad_sequence( mel_specs, batch_first = True).unsqueeze(1).transpose(2,3)
    return padded_mel_specs, torch.Tensor(pd.DataFrame(info_list).emotion.values.astype('int16')-1).long(), L_list



if __name__ == "__main__":
    meta = get_meta('../../dataset/ravdess/')