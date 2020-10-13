import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ans_path", type=str, required=True)
parser.add_argument("-pred_path", type=str, required=True)
args = parser.parse_args()

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

print("f1: %6f\tprecision: %6f\trecall: %6f" % eval_all(p, a))