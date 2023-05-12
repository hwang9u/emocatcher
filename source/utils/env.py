import torch
import argparse
import os

def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', default = '../../../dataset/ravdess')
    parser.add_argument('-o', '--out_dir', default = "./output/")
    parser.add_argument('-e', '--n_epochs', default = 100, type = int)
    parser.add_argument('-b', '--batch_size', default=32, type = int)
    parser.add_argument('-c', '--kfcv', default= False, type = lambda x:  True if x.lower() == 'true' else False)
    parser.add_argument('-r', '--random_state', default=999, type = int)
    parser.add_argument('-l', '--log', default = 'log.txt')
    parser.add_argument('-m', '--model_name', default = 'best_model')
    config = parser.parse_args()
    return config


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 

def delete_all_files(fpath):
    if os.path.exists(fpath):
        for file in os.scandir(fpath):
            os.remove(file.path)
        return "remove all"
    else:
        return "directory not found"

def save_model(dir, test_loss, test_acc, model, model_name = 'best_model', init = True):
    if init:
        delete_all_files(dir)
    torch.save(model.state_dict(), f"{dir}{model_name}_{test_acc:.4f}_{test_loss:.4f}.pth")
    print("model saved")

