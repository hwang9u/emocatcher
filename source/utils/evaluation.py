import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchmetrics import ConfusionMatrix
import numpy as np

# emotion class dictionary
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

def create_confusion_matrix(model, dataloader, n_class):
    '''
    confusion matrix
    - row: actual
    - col: predicted
    '''
    cm = torch.zeros( (n_class, n_class) )
    model.eval()
    for x_test, y_test, L in dataloader:
        y_pred_test = model(x_test,L).argmax(dim = 1)
        cm +=ConfusionMatrix( num_classes=n_class, task="multiclass" )(y_pred_test, y_test)
    print("Accuracy: {}".format(cm.trace()/ len(dataloader.dataset)) )
    return cm


def average_recall(cm, weighted = True):
    '''
    if weighted == True, WAR(Weighted Average Recall)
    '''
    n_samples_k = torch.sum(cm, dim = 1)
    tpr_k = cm.diag() / n_samples_k
    weight = n_samples_k/cm.sum() if weighted else 1/cm.shape[0] 
    ar = torch.sum(weight * tpr_k)
    return ar

def plot_metric_curve(train_acc_loss, test_acc_loss, plot_kwargs = {}):
    train_acc_list, train_loss_list = train_acc_loss
    test_acc_list, test_loss_list = test_acc_loss
    fig, ax = plt.subplots(1,2, **plot_kwargs)
    ax[0].grid(); ax[1].grid()
    ax[0].plot(train_acc_list, c='blue')
    ax[0].plot(test_acc_list, c='red')
    ax[0].legend(['Train Accuracy', 'Test Accuracy'])
    ax[1].plot(train_loss_list, c='blue')
    ax[1].plot(test_loss_list, c='red')
    ax[1].legend(['Train Loss', 'Test Loss'])
    ax[0].set_ylabel('Accuracy') ; ax[1].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch') ; ax[1].set_xlabel('Epoch')
    return fig



def plot_predicted_example(batch, model, n_rows=8, n_cols=4, figsize = (20,20), title_font_size = 20):
    x, y, L = batch
    model.eval()
    y_prob = torch.nn.functional.softmax(model(x, L), dim = -1).detach()
    y_pred = np.argmax(y_prob, axis=1)
    fig, ax = plt.subplots( n_rows, n_cols, figsize = figsize, dpi = 100)
    for img_ind in range(n_rows* n_cols):
        ax[img_ind//n_cols][img_ind%n_cols].imshow( torch.flip(( x[img_ind, :, :, :L[img_ind]].squeeze().transpose(0,1)), dims = [0] ))
        ax[img_ind//n_cols][img_ind%n_cols].set_title( emotion_dict[f"0{y[img_ind].item()+1}"] +"/"+  emotion_dict[f"0{y_pred[img_ind]+1}"]+ "({p:.2f} %)".format(p = y_prob[img_ind].numpy().max()*100), size = title_font_size )
    plt.tight_layout()
    
    
import seaborn as sn

def plot_cm(cm, normalize = False, ax = None):
    """
    plot Confusion Matrix
    
    """
    if not isinstance(cm,np.ndarray):
        cm = cm.detach().numpy()
    
    cm_df = pd.DataFrame(cm, columns = [ emotion_dict[f"0{i}"] for i in range(1,9) ], index=[ emotion_dict[f"0{i}"] for i in range(1,9) ] )
    factor = np.sum(cm_df, axis=1) if normalize else 1
    p = sn.heatmap(cm_df/factor, annot=True, ax=ax)
    p.set_ylabel('actual')
    p.set_xlabel('predicted')

