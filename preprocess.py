import json
import numpy as np
import torch
import sys
from transformers import BertTokenizer
from random import sample
from math import ceil
from tqdm import tqdm

type_dict = {"none":0, "name":1, "location":2, "time":3, "contact":4,
             "ID":5, "profession":6, "biomarker":7, "family":8,
             "clinical_event":9, "special_skills":10, "unique_treatment":11,
             "account":12, "organization":13, "education":14, "money":15,
             "belonging_mark":16, "med_exam":17, "others":18}

if (len(sys.argv) < 3):
    print("usage: python3 preprocess.py {mode} {prerpocessed_file_name(without extension)} [testing_id_list]")
    sys.exit(-1)
mode = sys.argv[1]
assert mode in ["train", "dev"]

print("\rloading pretrained language model ...", end='')
PRETRAINED_LM = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)
tokenizer.add_tokens(['…', '痾', '誒', '擤', '嵗', '曡', '厰', '聼', '柺'])

bert_data = []
with open ('./dataset/' + sys.argv[2] + '.json', 'r') as json_file:
    data_file = json.load(json_file)
    print("\rstart preprocessing ...              ")
    c = 0
    for data in tqdm(data_file):
        article = data['article']
        if (mode == "train"):
            type_list = []
            for i, item in enumerate(data['item']):
                article = article[:item[1] + i*2] + "_" + item[3] + "_" + article[item[2] + i*2:]
                type_list.append(type_dict[item[4]])
        article = article.replace("醫師：", "[SEP]") \
        .replace("民眾：", "[SEP]").replace("家屬：", "[SEP]") \
        .replace("個管師：", "[SEP]").replace("護理師：", "[SEP]")
        tokens = tokenizer.tokenize(article)
        if (mode == "train"):
            BIO_label = np.full(len(tokens), 2)
            type_label = np.full(len(tokens), 0)
            count_back = 0
            begin = 0
            j = 0
            remove_list = []
            for i in range(len(tokens)):
                if (tokens[i] == '_'):
                    remove_list.append(i)
                    if (count_back == 0):
                        begin = i+1
                        count_back += 1
                    else:
                        BIO_label[begin] = 0
                        BIO_label[begin+1 : i] = 1
                        type_label[begin : i] = type_list[j]
                        j += 1
                        count_back = 0

            BIO_label = BIO_label.tolist()
            type_label = type_label.tolist()

            for i in sorted(remove_list, reverse=True):
                del tokens[i], BIO_label[i], type_label[i]
        
        # tokens[0] = "[CLS]"
        if (tokens[0] == "[SEP]"):
            del tokens[0]
            if (mode == "train"):
                del BIO_label[0], type_label[0]
        tokens.append("[SEP]")
        if (mode == "train"):
            BIO_label.append(2)
            type_label.append(0)

        ids = tokenizer.convert_tokens_to_ids(tokens)

        if (mode == "train"):
            pt_dict = {'input_ids':ids, "BIO_label":BIO_label, "type_label":type_label,
                        'article_id':data['id']}
        else:
            pt_dict = {'input_ids':ids, 'article_id':data['id']}
        bert_data.append(pt_dict)
        c += 1
        print("\rprocessed %d data" %c, end="")
    
torch.save(bert_data, "./dataset/" + sys.argv[2] + "_bert_data.pt")
print("")

"""to length 512"""
bert_data_train_512 = []
bert_data_test_512 = []
c = 0
c1 = 0
c2 = 0
length = len(bert_data)
try:
    test_list = set(eval(sys.argv[3]))
except:
    test_list = sample(range(length), ceil(length * 0.33)) # split 1/3 testing data


for data in tqdm(bert_data):
    c += 1
    ids = data['input_ids']
    if (mode == "train"):
        BIO = data['BIO_label']
        type_ = data['type_label']
    pos = 0
    flag = 0
    sep_pos = 0
    new_pos = 0
    while (pos < len(ids)):
        ids_512 = ids[pos : pos+512]
        count_back = 0
        for i in range(min(510, len(ids_512)-1), 0, -1):
            if (ids_512[i] == 102): # 102 = [SEP]
                count_back += 1
                if (count_back == 1):
                    sep_pos = i
                    new_pos = pos + i + 1
                elif (count_back <= 3): # overlap n-1 sentences 
                    new_pos = pos + i + 1
        if(count_back == 0):
            sep_pos = 510
            new_pos = pos + 510 - 5 # overlap n tokens

        ids_512 = [101] + ids[pos : pos+sep_pos+1] + [0] * (512 - sep_pos - 2)
        seg_512 = [0] * 512

        att_512 = [1] * (sep_pos + 2) + [0] * (512 - sep_pos - 2)
        if (mode == "train"):
            BIO_512 = [2] + BIO[pos : pos+sep_pos+1] + [2] * (512 - sep_pos - 2)
            type_512 = [0] + type_[pos : pos+sep_pos+1] + [0] * (512 - sep_pos - 2)
        flag = 1 if (pos+sep_pos+1 >= len(ids)) else 0
        pos = new_pos

        if (mode == "train"):
            pt_dict = {"input_ids":ids_512, "seg":seg_512, "att":att_512,
                        "BIO_label":BIO_512, "type_label":type_512,
                        "article_id":data['article_id']}
        else:
            pt_dict = {"input_ids":ids_512, "seg":seg_512, "att":att_512,
                        "article_id":data['article_id']}

        if (data['article_id'] not in test_list and mode=="train"):
            bert_data_train_512.append(pt_dict)
            c1 += 1
        else:
            bert_data_test_512.append(pt_dict)
            c2 += 1
        
        print("\rprocessed %d data to length 512" %(c1+c2), end="")

        if (flag): # read single talk
            break

torch.save(bert_data_train_512, "./dataset/" + sys.argv[2] + "_train_512_bert_data.pt")
torch.save(bert_data_test_512, "./dataset/" + sys.argv[2] + "_test_512_bert_data.pt")

print("")
print("processed %d origin datas to %d train datas and %d test datas in length 512"
    % (c, c1, c2))
print("Preprocess Done !!")
print("Testing set id list: ", test_list)