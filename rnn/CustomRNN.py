import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)


class erp_rnn(nn.Module):
    def __init__(self, ncha=8, fs=128, dropout_rate=0.25, n_classes=2, scales_time=(500, 250, 125)):
        super(erp_rnn, self).__init__()

        self.ncha = ncha
        self.fs = fs
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.activation = nn.ELU(inplace=True)
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 2 - 1, scales_sample // 2, )),
                nn.Conv2d(1, 8, (scales_sample, 1)),
                nn.BatchNorm2d(8),
                self.activation,
                nn.Dropout(dropout_rate),
                nn.Conv2d(8, 16,
                          (1, 8), bias=False, groups=8),  # DepthwiseConv2D
                nn.BatchNorm2d(16),
                self.activation,
                nn.Dropout(dropout_rate)，
            ) for scales_sample in scales_samples
        ])

        self.lstm = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(input_size=128, hidden_size=16, num_layers=3,
                                  batch_first=True, bidirectional=True, dropout=0.5)
            ) for _ in range(3)
        ])

        self.avg_pool = nn.AvgPool2d((1, 4))

        self.dense = nn.Sequential(
            nn.Linear(4 * 1 * 6, n_classes),
            nn.Softmax(1)
        )

        self.out = nn.Sequential(
            nn.Conv2d(1, 8, (1, 1)),
            nn.BatchNorm2d(8),
            self.activation,
            self.avg_pool,
            nn.Dropout(0.25)
        )

        self.dense = nn.Sequential(
            nn.Linear(8*48*8, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        concat_in = []
        for i in range(3):
            lstm_in = self.inception1[i](x).squeeze()
            lstm_out, _ = self.lstm[i](lstm_in)
            concat_in.append(lstm_out)
        concat_out = torch.cat(concat_in, 1)  # 32*48*16
        feat = concat_out.unsqueeze(1)
        feat = self.out(feat)
        dense_in = torch.flatten(feat, 1)

        return self.dense(dense_in)

class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score+self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print('EarlyStopping')
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss,model,path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
