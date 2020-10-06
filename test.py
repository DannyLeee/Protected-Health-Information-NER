#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
import numpy as np
from data_loader import TalkDataset
from model_budling import PHI_NER
from transformers import BertTokenizer
import json
import pandas as pd

type_dict = {0:"NONE", 1:"name", 2:"location", 3:"time", 4:"contact",
             5:"ID", 6:"profession", 7:"biomarker", 8:"family",
             9:"clinical_event", 10:"special_skills", 11:"unique_treatment",
             12:"account", 13:"organization", 14:"education", 15:"money",
             16:"belonging_mark", 17:"med_exam", 18:"others"}

FILE_PATH = "./dataset/development_1_test_512_bert_data.pt" ##########
test_file = torch.load(FILE_PATH)
# test_file = test_file[:1] ############

PRETRAINED_LM = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)
tokenizer.add_tokens(['…', '痾', '誒', '擤', '嵗', '曡', '厰', '聼', '柺'])

json_file = open('./dataset/development_1.json') #########
data_file = json.load(json_file)

"""
type_vote
vote type of the prediction span
input: type list (list of int)
output: type(int)
"""
def type_vote(type_list):
    type_ = [0] * 19
    for i in type_list:
        type_[i] += 1
    return np.argmax(type_)

def get_position(id, span, text):
    start = 0
    article = data_file[id]['article'].lower()
    start = article.find(span) + span.find(text)
    return start

def bio_2_string(tokens_tensors, type_pred, BIO_tagging, id):
    result = []
    for j in range(1, 512):
        if (BIO_tagging[j] == 0):
            start = j
            end = j + 1
            while (end < 512 and BIO_tagging[end] == 1):
                end += 1

            tgt = tokenizer.decode(token_ids = tokens_tensors[start : end]).replace(' ', '')
            token_span = tokens_tensors[start : end]
            for i in range(4):
                if (tokens_tensors[end + i] == tokenizer.vocab['[SEP]']):
                    token_span = torch.cat((token_span, tokens_tensors[end : end+i]), 0)
                    break
            for i in range(0, -4, -1):
                if (tokens_tensors[start + i] == tokenizer.vocab['[SEP]']):
                    temp = torch.tensor([tokenizer.vocab['：']]).to(device)
                    token_span = torch.cat((temp, tokens_tensors[start+i+1 : start], token_span), 0)
                    break
            span = tokenizer.decode(token_ids = token_span).replace(' ', '')
            type_ = type_vote(type_pred[start : end])
            if (type_ != 0):
                s_pos = get_position(id, span, tgt)
                result.append([id, s_pos, s_pos+len(tgt), tgt, type_dict[type_]])
            start = end

    return result

def get_predictions(model, testLoader, BATCH_SIZE):
    result = []
    total_count = 0 # 第n筆data
    with torch.no_grad():
        for data in testLoader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            ids = data[-1]
            outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

            for i in range(outputs[0].shape[0]):  # run batchsize times
                type_pred = outputs[0][i].argmax(1) # 19*512 into class label
                BIO_pred = outputs[1][i].argmax(1) # 3*512 into class label
                text_token = tokens_tensors[i]
                r = bio_2_string(text_token, type_pred, BIO_pred, ids[i].item())
                result.append(r)
                total_count += 1
    return result

"""testing"""
MODEL_PATH = "./model/train_1_E10.pt" ##############
# MODEL_PATH = "./model/test_E500.pt"

model = PHI_NER()
model.load_state_dict(torch.load(MODEL_PATH))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BATCH_SIZE = 6
testSet = TalkDataset("test", test_file)
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE)


predictions = get_predictions(model, testLoader, BATCH_SIZE)

h = ["article_id", "start_position", "end_position", "entity_text", "entity_type"]
df = pd.DataFrame(columns=h)
for p in predictions:
    temp = pd.DataFrame(p, columns=h)
    df = df.append(temp, ignore_index=True)
df = df.drop_duplicates()
df.to_csv('./result/development_1.tsv', index=False, sep="\t")  ##########