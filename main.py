from EEGInception_torch import EEGInception, EarlyStopping
from torch import nn, optim
import numpy as np
from dataset import ERPDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import csv


dataset_path = './dataset/GIB-UVA_ERP-BCI.hdf5'
model_path="./train"
epochs=500

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
nums_val=len(val_loader)

def train():
    print("-------start training----------")
    val_epoch_loss = []
    for epoch in range(0, epochs):
        print("epoch:{}/{}".format(epoch+1, epochs))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1024,  # 批量大小
            shuffle=True  # 是否打乱数据顺序
        )

        model.train()
        t=tqdm(train_loader)
        for _,(feat, label) in enumerate(t):
            optimizer.zero_grad()
            out = model(feat)
            loss_train = loss_fn(out, label)
            loss_train.backward()
            optimizer.step()
            t.set_postfix(loss=loss_train.item())
        with torch.no_grad():
            print("--------------val--------------------")
            model.eval()
            loss_val = torch.zeros(1, device=device)
            for _,(val_feat, val_label) in  enumerate(tqdm(val_loader)):
                val_output = model(val_feat)
                loss_val += loss_fn(val_output, val_label)
            
            loss=(loss_val.cpu().numpy()/nums_val)[0]
            print("val_loss of epoch{}/{}   = {:.4f}".format(epoch+1, epochs,loss))
            val_epoch_loss.append(loss)
            early_stopping(loss, model,model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    model.load_state_dict(torch.load(model_path+'/checkpoint.pth'))
    with open(model_path+"/train_info.csv","w",encoding="utf-8",newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(val_epoch_loss)

train()
