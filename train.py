# -* coding:utf-8 -*

import copy
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from utils import get_dataloader_DCCN, parse_arguments_bpsk
from model_nsa import TWO_DCCN
from visdom import Visdom

torch.cuda.set_device(0)


args = parse_arguments_bpsk()
batch_size = 64
num_epochs = 100
learning_rate = 1E-4
weight_decay =  0.001

class EarlyStopping:
    def __init__(self, patience=5, warmup=10,delta=0.001):

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.warmup = warmup

    def __call__(self, test_loss):

        if epoch < self.warmup:
            print('No Early Stopping')
            return False
        if self.best_loss is None:
            self.best_loss = test_loss
        elif test_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = test_loss
            self.counter = 0
        return self.early_stop
##################################   Dataset   #####################################
train_data_loader = get_dataloader_DCCN(
    './dataset/mobile_dataset/train_dataset_20_mobile_20.mat',
    'train_data', 'train_label',
    batch_size, shuffle=True)
test_data_loader = get_dataloader_DCCN(
    './dataset/mobile_dataset/test_dataset_20_mobile_20.mat',
    'test_data', 'test_label',
    batch_size, shuffle=False)
#################################################################################

net = TWO_DCCN().cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # AdamW

early_stopping = EarlyStopping(patience=5, warmup=10, delta=0.001)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(5, num_epochs, 5)), gamma=0.8)

torch.autograd.set_detect_anomaly(True)

#######################################################################
# visdom
viz = Visdom(env='loss and ber')
loss_win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title=f'Loss  dataset:d_snr_mix_v0_p1<br> b={batch_size}_lr={learning_rate}_L2={weight_decay}',
            xlabel='Epochs', ylabel='Loss', legend=['Train Loss', 'Val Loss', 'Test Loss']))
berl_win = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title=f'Acc  dataset:d_snr_mix_v0_p1<br> b={batch_size}_lr={learning_rate}_L2={weight_decay}',
            xlabel='Epochs', ylabel='Acc', legend=['Train Acc', 'Val Acc', 'Test Acc']))
#######################################################################


for epoch in range(num_epochs):
    print('Epoch:', epoch, 'Weight Decay:', weight_decay, 'lr:', optimizer.param_groups[0]['lr'])
    net.train()
    train_loss_num = []
    train_berl_num = []
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for (sample, label) in tqdm(train_data_loader):

        sample = Variable(sample.cuda())                        # [b,2,1164,2]
        label = Variable(label.cuda())                          # [b,1,1,1664] 0101010 bit stream
        labels = torch.reshape(label, (-1,))

        predict_labels = net(sample)
        predict_labels = torch.reshape(predict_labels, (-1, 2))

        optimizer.zero_grad()
        loss = criterion(predict_labels, labels.long())

        total_loss += loss.item() * labels.numel()
        total_correct += (predict_labels.argmax(1) == labels).sum().item()
        total_samples += labels.numel()

        loss.backward()

        clip_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
        optimizer.step()

    scheduler.step()
    train_acc = total_correct / total_samples
    train_loss = total_loss / total_samples
    train_berl_num.append(train_acc)
    train_loss_num.append(train_loss)
    tqdm.write('Train berl: {:.6f}'.format(train_acc))
    tqdm.write('Train loss :{}'.format(train_loss))

###################################################################

    net.eval()
    val_loss_num = []
    val_berl_num = []
    val_loss, val_correct, val_samples = 0.0, 0, 0
    with torch.no_grad():
        for (sample, label) in tqdm(test_data_loader):
            sample = Variable(sample.cuda())
            label = Variable(label.cuda())
            labels = torch.reshape(label, [-1])

            predict_labels = net(sample)
            predict_labels = torch.reshape(predict_labels, (-1, 2))

            loss = criterion(predict_labels, labels.long())
            val_loss += loss.item() * labels.numel()
            val_correct += (predict_labels.argmax(1) == labels).sum().item()
            val_samples += labels.numel()

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        val_berl_num.append(val_acc)
        val_loss_num.append(val_loss)
        print('Certify berl: {:.6f}'.format(val_acc))
        print('Certify loss :{}'.format(val_loss))
        print("********************************************")

    if early_stopping(val_loss):
        print(f"Early stopping triggered at epoch {epoch}. Best val loss: {early_stopping.best_loss:.4f}")
        dir = 'result/'
        torch.save(net, dir + f'DARNet_{epoch}.pth')
        break

    # Visdom
    viz.line(X=np.array([epoch]), Y=np.array([train_loss]),       win=loss_win,   update='append', name='Train Loss')
    viz.line(X=np.array([epoch]), Y=np.array([val_loss]),  win=loss_win,   update='append', name='Val Loss')


    viz.line(X=np.array([epoch]), Y=np.array([train_acc]),        win=berl_win,   update='append', name='Train Acc')
    viz.line(X=np.array([epoch]), Y=np.array([val_acc]),   win=berl_win,   update='append', name='Val Acc')

###################################################################
# Result Save Model

    if val_acc > 0.99:
        dir = 'result/'
        torch.save(net, dir + 'DARNet_{}.pth'.format(epoch))
    elif epoch  == num_epochs-1:
        dir = 'result/'
        torch.save(net, dir + 'DARNet_{}.pth'.format(epoch))


