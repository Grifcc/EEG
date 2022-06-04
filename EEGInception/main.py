from EEGInception_torch import EEGInception, EarlyStopping
from torch import nn, optim
import numpy as np
from dataset import ERPDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import csv


dataset_path = '../dataset/GIB-UVA_ERP-BCI.hdf5'
model_path = "./train"
epochs = 500

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
input_time = 1000
fs = 128
n_cha = 8
filters_per_branch = 8
scales_time = (500, 250, 125)
dropout_rate = 0.25
n_classes = 2
learning_rate = 0.001

train_dataset = ERPDataset(dataset_path, "train", 0.2, device)
val_dataset = ERPDataset(dataset_path, "val", 0.2, device)
val_loader = DataLoader(val_dataset, 32)
model = EEGInception(input_time, fs, n_cha, filters_per_branch,
                     scales_time, dropout_rate, n_classes=n_classes)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                       betas=(0.9, 0.999), amsgrad=False)
loss_fn = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(10, verbose=False, delta=0.0001)
nums_val = len(val_loader)


def train():
    print("-------start training----------")
    train_info = []
    for epoch in range(0, epochs):
        print("epoch:{}/{}".format(epoch+1, epochs))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1024,  # 批量大小
            shuffle=True  # 是否打乱数据顺序
        )

        model.train()
        t = tqdm(train_loader)
        correct_t = torch.zeros(1).squeeze().cuda()
        total_t = torch.zeros(1).squeeze().cuda()
        for _, (feat, label) in enumerate(t):
            optimizer.zero_grad()
            out = model(feat)
            loss_train = loss_fn(out, label)
            loss_train.backward()
            optimizer.step()
            prediction_t = torch.argmax(out, 1)
            correct_t += (prediction_t == label).sum().float()
            total_t += len(label)
            t.set_postfix(loss=loss_train.item(),
                          accuracy=(correct_t/total_t).item())

        with torch.no_grad():
            print("--------------val--------------------")
            model.eval()
            loss_totle = torch.zeros(1).squeeze().cuda()
            correct_v = torch.zeros(1).squeeze().cuda()
            total_v = torch.zeros(1).squeeze().cuda()
            t = tqdm(val_loader)
            for _, (val_feat, val_label) in enumerate(t):
                val_output = model(val_feat)
                loss_val = loss_fn(val_output, val_label)
                loss_totle += loss_val
                prediction_v = torch.argmax(val_output, 1)
                correct_v += (prediction_v == val_label).sum().float()
                total_v += len(val_label)
                t.set_postfix(loss=loss_val.item(),
                              accuracy=(correct_v/total_v).item())
            loss = (loss_totle.item()/nums_val)
            accuracy_v=(correct_v/total_v).item()
            print("val_loss:{:.4f} accuracy:{:.4f} ".format(loss,accuracy_v))
            train_info.append(
                ["epoch {}".format(epoch+1), "{:.4f}".format(loss), "{:.4f}".format(accuracy_v)])
            early_stopping(loss, model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    model.load_state_dict(torch.load(model_path+'/checkpoint.pth'))
    with open(model_path+"/train_info.csv", "w", encoding="utf-8", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["epochs/500","loss","accuracy"])
        f_csv.writerows(train_info)


train()
