
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import glob
import librosa
from utils.audio_tools import gvad
import numpy as np

class RAVDESSMelS(Dataset):
    def __init__(self, dataset_dir, mel_specs_kwargs = {}, speech_list= None, out_dir = None):
        self.dataset_dir = dataset_dir
        self.mel_specs_kwargs = mel_specs_kwargs
        self.out_dir = out_dir
        
        if speech_list == None:
            self.speech_list = glob.glob(f"{dataset_dir}*/*.wav")
        else:
            self.speech_list = speech_list
    
    def __getitem__(self, index):
        fpath = self.speech_list[index]
        fn = fpath.split("/")[-1].split(".")[0]
        
        sig, sr = librosa.load(self.speech_list[index])
            
        features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
        info = {f:c for c, f in zip(fn.split("-"), features)}
        mel_spec =  truncate_melspecs(sig, return_va_point=False, sr = sr, mel_specs_kwargs=self.mel_specs_kwargs)   
        return torch.Tensor(mel_spec), sr, info, fn
    
    def __len__(self):
        return len(self.speech_list)


def truncate_melspecs(sig, return_va_point = False, sr = 22050, mel_specs_kwargs = {}):
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr= sr, **mel_specs_kwargs))
    va_point = gvad(librosa.db_to_amplitude(mel_spec)**2)
    mel_spec = mel_spec[:, va_point[0]:va_point[1]]
    if return_va_point:
        return mel_spec, va_point
    else:
        return mel_spec


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
 }

