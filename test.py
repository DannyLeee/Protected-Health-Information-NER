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

FILE_PATH = "./dataset/sample_512_bert_data.pt"
test_file = torch.load(FILE_PATH)
# test_file = test_file[:1] ############

PRETRAINED_LM = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)

len(test_file)

json_file = open('./dataset/sample.json')
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
    tpos = span.find(text)
    sep = span.find("[SEP]", 0, tpos)
    rsep = span.rfind("[SEP]", tpos)
    if (sep!=-1 and rsep!=-1):
        span = span[sep+5 : rsep]
    elif (sep != -1):
        span = span[sep+5:]
    elif (rsep != -1):
        span = span[:rsep]
    article = data_file[id]['article'].lower()
    start = article.find(span) + span.find(text)
    return start

def bio_2_string(tokens_tensors, type_pred, BIO_tagging, id):
#     result_type = []
#     result_text = []
#     start_pos = []
#     end_pos = []
    result = []
    for j in range(1, 512):
        if (BIO_tagging[j] == 0):
            start = j
            end = j + 1
            while (end < 512 and BIO_tagging[end] == 1):
                end += 1

            tgt = tokenizer.decode(token_ids = tokens_tensors[start : end]).replace(' ', '')
            span = tokenizer.decode(token_ids = tokens_tensors[start-3 : end+3]).replace(' ', '')
            type_ = type_vote(type_pred[start : end])
            if (type_ != 0):
#                 result_type.append(type_dict[type_])
#                 result_text.append(tgt)
                s_pos = get_position(id, span, tgt)
#                 print(span)
#                 print(tgt)
#                 print("---")
#                 start_pos.append(s_pos)
#                 end_pos.append(s_pos + len(tgt))
                result.append([id, s_pos, s_pos+len(tgt), tgt, type_dict[type_]])
#             print(s)
            start = end

    # print('---')
    return result

def get_predictions(model, testLoader, BATCH_SIZE):
    result = []
    total_count = 0 # 第n筆data
    with torch.no_grad():
        for data in testLoader:
#             for i, t in enumerate(data):
#                 data[i] = torch.reshape(t, (t.size(0) * t.size(1),512))
#                 BATCH_SIZE = t.size(0) * t.size(1)
#             print(data)
#             print(data[0].size())
#             break
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

#             count = min(outputs[0].shape[0], BATCH_SIZE)
            for i in range(outputs[0].shape[0]):  # run batchsize times
                type_pred = outputs[0][i].argmax(1) # 19*512 into class label
                BIO_pred = outputs[1][i].argmax(1) # 3*512 into class label
#                 print(type_pred)
#                 print(BIO_pred)
                text_token = tokens_tensors[i]
                r = bio_2_string(text_token, type_pred, BIO_pred, ids[i].item())
                result.append(r)
                total_count += 1
                # break
                # print(result)
#             break
    return result

"""testing"""
MODEL_PATH = "./model/smaple_E10.pt" ##############
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
df.to_csv('./result/smaple.tsv', index=False, sep="\t")  ##########

for i, p in enumerate(predictions):
#     type_label = []
#     BIO_text = []
    ans = []
    for j, b in enumerate(test_file[i]['BIO_label']):
        if (b == 0):
            start = j
            end = j + 1
            while (end < 512 and test_file[i]['BIO_label'][end] == 1):
                end += 1
#             type_label.append(type_dict[type_vote(test_file[i]['type_label'][start : end])])
#             BIO_text.append(tokenizer.decode(token_ids = test_file[i]['input_ids'][start : end]).replace(" ", ""))
            type_ = type_dict[type_vote(test_file[i]['type_label'][start : end])]
            tgt = tokenizer.decode(token_ids = test_file[i]['input_ids'][start : end]).replace(" ", "")
            ans.append([type_, tgt])
    
    print(ans)
    print(p)
    print("#####\n")