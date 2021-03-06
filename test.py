#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
import numpy as np
from data_loader import TalkDataset
from model_budling import PHI_NER
from transformers import BertTokenizer, AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm
import argparse

type_dict = {0:"NONE", 1:"name", 2:"location", 3:"time", 4:"contact",
             5:"ID", 6:"profession", 7:"biomarker", 8:"family",
             9:"clinical_event", 10:"special_skills", 11:"unique_treatment",
             12:"account", 13:"organization", 14:"education", 15:"money",
             16:"belonging_mark", 17:"med_exam", 18:"others"}

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", type=str, required=True)
parser.add_argument("-json_path", type=str, required=True)
parser.add_argument("-lm", default="hfl/chinese-bert-wwm", type=str)
parser.add_argument("-batch_size", default=16, type=int)
parser.add_argument("-model_path", type=str, required=True)
args = parser.parse_args()

test_file = torch.load(args.data_path)

tokenizer = AutoTokenizer.from_pretrained(args.lm)
# tokenizer.add_tokens(['…', '痾', '誒', '擤', '嵗', '曡', '厰', '聼', '柺', '齁'])

json_file = open(args.json_path)
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

def get_position(id, span, text, beg):
    start = 0
    article = data_file[id]['article'].lower()
    start = article.find(span, beg) + span.find(text)
    return start

def bio_2_string(tokens_tensors, type_pred, BIO_tagging, id):
    result = []
    b = 0
    for j in range(1, 512):
        if (BIO_tagging[j] == 0):
            start = j
            end = j + 1
            while (end < 512 and BIO_tagging[end] == 1):
                end += 1

            tgt = tokenizer.decode(tokens_tensors[start : end])
            token_span = tokens_tensors[start : end]
            for i in range(4):
                if (end + i >= 512):
                    break
                if (tokens_tensors[end + i] == tokenizer.sep_token):
                    token_span = torch.cat((token_span, tokens_tensors[end : end+i]), 0)
                    break
            for i in range(0, -4, -1):
                if (end + i <= 0):
                    break
                if (tokens_tensors[start + i] == tokenizer.sep_token):
                    temp = torch.tensor([tokenizer.convert_tokens_to_ids('：')]).to(device)
                    token_span = torch.cat((temp, tokens_tensors[start+i+1 : start], token_span), 0)
                    break
            span = tokenizer.decode(token_ids = token_span).replace(' ', '')
            type_ = type_vote(type_pred[start : end])
            if (type_ != 0):
                s_pos = get_position(id, span, tgt, b)
                b = s_pos
                result.append([id, s_pos, s_pos+len(tgt), tgt, type_dict[type_]])
            start = end

    return result

def get_predictions(model, testLoader, BATCH_SIZE):
    result = []
    total_count = 0 # 第n筆data
    with torch.no_grad():
        for data in tqdm(testLoader):
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
model = PHI_NER(args.lm)
model.load_state_dict(torch.load(args.model_path))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model = model.to(device)
model.eval()
testSet = TalkDataset("test", test_file)
testLoader = DataLoader(testSet, batch_size=args.batch_size)


predictions = get_predictions(model, testLoader, args.batch_size)

h = ["article_id", "start_position", "end_position", "entity_text", "entity_type"]
df = pd.DataFrame(columns=h)
for p in predictions:
    temp = pd.DataFrame(p, columns=h)
    df = df.append(temp, ignore_index=True)
df = df.drop_duplicates()
model_name = args.model_path[args.model_path.find("model/")+6 : -3]
if args.json_path.find("dev") != -1:
    df.to_csv(args.json_path[:-5].replace("dataset/", "result/") + model_name + '.tsv', index=False, sep="\t")
else:
    df.to_csv("result/" + model_name + '.tsv', index=False, sep="\t")