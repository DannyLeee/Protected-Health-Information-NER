import pandas as pd
import numpy as np
import json
from sklearn_crfsuite import metrics
import sys

if (len(sys.argv) != 3):
    print("usage: python3 evaluate.py {ans_file_path(json)} {prediction_file_path(tsv)}")
    exit(-1)

ans_file = sys.argv[1]
pred_file = sys.argv[2]

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

df = pd.read_csv(pred_file, sep="\t")
json_file = open(ans_file, 'r')
ans_dict = json.load(json_file)

def BIO_tagging(tag_list, tag_info):
    result = np.array(tag_list)
    for info in tag_info:
        result[info[1]] = "B-" + info[4]
        result[info[1]+1: info[2]] = "I-" + info[4]
    result = result.tolist()
    return result

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
        ans_BIO_list[-1] = BIO_tagging(ans_BIO_list[-1], ans_dict[id]['item'])
        pred_BIO_list.append(['O'] * length)
        pred_BIO_list[-1] = BIO_tagging(pred_BIO_list[-1], pred_item)
    else:
        BIO_tagging(pred_BIO_list[-1], pred_item)

print("f1 score: %6f" % metrics.flat_f1_score(ans_BIO_list, pred_BIO_list, average='weighted', labels=label_list, zero_division=0))
json_file.close()