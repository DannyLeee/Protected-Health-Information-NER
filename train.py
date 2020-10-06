#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AdamW
from data_loader import TalkDataset
from model_budling import PHI_NER
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BATCH_SIZE = 4
data_path = "./dataset/train_1_train_512_bert_data.pt"

list_of_dict = torch.load(data_path)
# list_of_dict = list_of_dict[:1] #############
# train_list = list_of_dict[:80]

""" model setting (training)"""
trainSet = TalkDataset("train", list_of_dict)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = PHI_NER()
optimizer = AdamW(model.parameters(), lr=1e-5) # AdamW = BertAdam

BIO_weight = torch.FloatTensor([98.33333333, 53.5694687,   1.        ]).cuda()
type_weight = torch.FloatTensor([1.00000000e+00, 4.79372641e+02, 5.39595000e+02, 4.83475676e+01,
4.38265956e+03, 1.13018437e+04, 4.99416203e+03, 9.71971425e+07,
3.90457045e+03, 2.68375880e+04, 7.15339819e+04, 3.97687436e+03,
9.71971425e+07, 1.07261502e+05, 3.57801575e+04, 1.07378815e+03,
9.71971425e+07, 4.60856189e+02, 9.71971425e+07,]).cuda()

BIO_loss_fct = nn.CrossEntropyLoss(weight=BIO_weight)
type_loss_fct = nn.CrossEntropyLoss(weight=type_weight)

# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
#             print(_)
    else:
        print("{:15} {}".format(name, module))

""" training """
from datetime import datetime,timezone,timedelta

model = model.to(device)
model.train()

EPOCHS = 10
dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
print(dt2)
for epoch in range(EPOCHS):
    running_loss = 0.0
    type_running_loss = 0.0
    BIO_running_loss = 0.0
    for data in trainLoader:        
        tokens_tensors, segments_tensors, masks_tensors,    \
        type_label, BIO_label = [t.to(device) for t in data]

        # 將參數梯度歸零
        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

        type_pred = outputs[0]
        type_pred = torch.transpose(type_pred, 1, 2)
        type_loss = type_loss_fct(type_pred, type_label)

        BIO_pred = outputs[1]
        BIO_pred = torch.transpose(BIO_pred, 1, 2)
        BIO_loss = BIO_loss_fct(BIO_pred, BIO_label)

        loss = BIO_loss + type_loss

        # backward
        loss.backward()
        optimizer.step()

        # 紀錄當前 batch loss
        running_loss += loss.item()
        type_running_loss += type_loss.item()
        BIO_running_loss += BIO_loss.item()

    if ((epoch + 1) % 10 == 0 or True): #####
        CHECKPOINT_NAME = './model/train_1_E' + str(epoch + 1) + '.pt' ########################
        torch.save(model.state_dict(), CHECKPOINT_NAME)

        dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
        dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
        print('%s\t[epoch %d] loss: %.3f, type_loss: %.3f, BIO_loss: %.3f' %
              (dt2, epoch + 1, running_loss, type_running_loss, BIO_running_loss))