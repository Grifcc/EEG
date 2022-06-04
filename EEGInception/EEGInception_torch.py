import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)


class EEGInception(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True), n_classes=2):
        super().__init__()

        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]

        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 2 - 1, scales_sample // 2, )),
                nn.Conv2d(1, filters_per_branch, (scales_sample, 1)),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
                nn.Conv2d(filters_per_branch, filters_per_branch*2,
                          (1, ncha), bias=False, groups=filters_per_branch),  # DepthwiseConv2D
                nn.BatchNorm2d(filters_per_branch * 2),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool1 = nn.AvgPool2d((4, 1))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 8 -
                           1, scales_sample // 8, )),
                nn.Conv2d(
                    len(scales_samples) * 2 * filters_per_branch,
                    filters_per_branch, (scales_sample // 4, 1),
                    bias=False
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(

            CustomPad((0, 0, 4, 3)),
            nn.Conv2d(
                24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                bias=False

            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),

            CustomPad((0, 0, 2, 1)),
            nn.Conv2d(
                12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                bias=False

            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            # nn.Dropout(dropout_rate),
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),  #收敛更快
        )

        self.dense = nn.Sequential(
            nn.Linear(4 * 1 * 6, n_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = torch.cat([net(x) for net in self.inception1], 1)  # concat
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], 1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x


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
