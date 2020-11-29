import json
import pandas as pd
import argparse
import numpy as np
from sklearn_crfsuite import metrics

parser = argparse.ArgumentParser()
parser.add_argument("-ans_path", type=str, required=True)
parser.add_argument("-pred_path", type=str, required=True)
args = parser.parse_args()

label_list = ["O", "B-name", "B-location", "B-time", "B-contact",
             "B-ID", "B-profession", "B-biomarker", "B-family",
             "B-clinical_event", "B-special_skills", "B-unique_treatment",
             "B-account", "B-organization", "B-education", "B-money",
             "B-belonging_mark", "B-med_exam", "B-others",
             "I-name", "I-location", "I-time", "I-contact",
             "I-ID", "I-profession", "I-biomarker", "I-family",
             "I-clinical_event", "I-special_skills", "I-unique_treatment",
             "I-account", "I-organization", "I-education", "I-money",
             "I-belonging_mark", "I-med_exam", "I-others"]

ans_BIO_list = []
pred_BIO_list = []

df = pd.read_csv(args.pred_path, sep="\t")
json_file = open(args.ans_path, 'r')
ans_dict = json.load(json_file)

def BIO_tagging(tag_list, tag_info):
    for info in tag_info:
        tag_list[info[1]] = "B-" + info[4]
        for i in range(info[1]+1, info[2]):
            tag_list[i] = "I-" + info[4]
    return

id_set = set()
id_list = []
for index, rows in df.iterrows():
    id = rows.article_id
    pred_item = [rows.tolist()]
    if(id not in id_set):
        id_set.add(id)
        id_list.append(id)
        length = len(ans_dict[id]['article'])
        ans_BIO_list.append(['O'] * length)
        BIO_tagging(ans_BIO_list[-1], ans_dict[id]['item'])
        pred_BIO_list.append(['O'] * length)
        BIO_tagging(pred_BIO_list[-1], pred_item)
    else:
        BIO_tagging(pred_BIO_list[-1], pred_item)

print("BIO tagging f1 score: %6f" % metrics.flat_f1_score(ans_BIO_list, pred_BIO_list, average='weighted', labels=label_list, zero_division=0))
json_file.close()

def eval(pred, ans):
    if pred == ans:
        return 1, 1, 1
    else:
        TP = 0
        for a in ans:
            for p in pred:
                if (a == p):
                    TP += 1
                    break
        FP = len(pred) - TP
        FN = len(ans) - TP
        if TP == 0:
            return 0, 0, 0

        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        return 2 * (precision*recall / (precision+recall)), precision, recall

def eval_all(pred_list, ans_list):
    f1 = 0
    precision = 0
    recall = 0
    for p, a in zip(pred_list, ans_list):
       F, P, R = eval(p, a)
       f1 += F
       precision += P
       recall += R
    f1 /= len(pred_list)
    precision /= len(pred_list)
    recall /= len(pred_list)
    return f1, precision, recall

a = []
p = []

pred_df = pd.read_csv(args.pred_path, sep='\t')
id_list = []
temp = []
with open(args.ans_path) as json_file:
    j = json.load(json_file)
    for index, rows in pred_df.iterrows():
        id = rows.article_id
        temp.append(rows.tolist())
        if (id not in id_list):
            a.append(j[id]['item'])
            id_list.append(id)
            if (index != 0):
                p.append(temp)
                temp = []
    p.append(temp) # last id's prediction

print("term f1: %6f\tprecision: %6f\trecall: %6f" % eval_all(p, a))