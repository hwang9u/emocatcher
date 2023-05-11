import torch
from tqdm import tqdm
from time import sleep
from utils.env import create_folder

def train(train_dloader, model,  optimizer, crit, epoch, device, max_norm = None):
    model.train()
    train_loss = 0
    train_acc = 0
    with tqdm(train_dloader, unit = "batch") as tepoch:
        for x, y, L in tepoch:
            tepoch.set_description(f"[Epoch {epoch}]")
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, L)
            loss_i = crit( y_pred, y)
            acc_i = y_pred.argmax(dim = 1).eq(y).sum().item()
            optimizer.zero_grad()
            loss_i.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            train_loss += loss_i.item()
            train_acc += acc_i
            tepoch.set_postfix(train_loss = loss_i.item(), train_acc = acc_i/y_pred.shape[0]*100)
            sleep(0.01)
    return train_loss / len(train_dloader), train_acc/ len(train_dloader.dataset)



def evaluate(test_dloader, model, crit, epoch, device):
    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.no_grad():
        with tqdm(test_dloader, unit = "batch") as tepoch:
            for x, y, L in tepoch:
                tepoch.set_description(f"[Epoch {epoch}]")
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x, L) 
                acc_i = y_pred.argmax(dim = 1).eq(y).sum().item()
                test_acc += acc_i
                loss_i = crit(y_pred, y)
                test_loss += loss_i.item()
                tepoch.set_postfix(test_loss = loss_i.item(), test_acc = acc_i/y_pred.shape[0]*100)
                sleep(0.01)
    return test_loss/ len(test_dloader), test_acc / len(test_dloader.dataset)
