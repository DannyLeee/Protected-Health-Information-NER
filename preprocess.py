import json
import numpy as np
import torch
import sys
from transformers import BertTokenizer
from random import sample
from math import ceil
from tqdm import tqdm
import argparse

type_dict = {"none":0, "name":1, "location":2, "time":3, "contact":4,
             "ID":5, "profession":6, "biomarker":7, "family":8,
             "clinical_event":9, "special_skills":10, "unique_treatment":11,
             "account":12, "organization":13, "education":14, "money":15,
             "belonging_mark":16, "med_exam":17, "others":18}

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, choices=['train','dev'], required=True)
parser.add_argument("-sep_mode", type=str, default="replace", choices=['replace', 'add'])
parser.add_argument("-seg_mode", type=int, default=0, choices=[0,1])
parser.add_argument("-file_name", type=str, required=True, help="prerpocessed file name (without extension)")
parser.add_argument("-pretrained_lm", type=str, default="hfl/chinese-bert-wwm")
parser.add_argument("-test_list", type=str, default="[]", help="will overwrite test_percent")
parser.add_argument("-test_percent", type=float, default=0.33)
args = parser.parse_args()

print("\rloading pretrained language model ...", end='')
tokenizer = BertTokenizer.from_pretrained(args.pretrained_lm)
tokenizer.add_tokens(['…', '痾', '誒', '擤', '嵗', '曡', '厰', '聼', '柺'])

bert_data = []
with open ('./dataset/' + args.file_name + '.json', 'r') as json_file:
    data_file = json.load(json_file)
    print("\rstart preprocessing ...              ")
    c = 0
    for data in tqdm(data_file):
        article = data['article']
        if (args.mode == "train"):
            type_list = []
            for i, item in enumerate(data['item']):
                article = article[:item[1] + i*2] + "_" + item[3] + "_" + article[item[2] + i*2:]
                type_list.append(type_dict[item[4]])
        if (args.sep_mode == "replace"):
            article = article.replace("醫師：", "[SEP]") \
            .replace("民眾：", "[SEP]").replace("家屬：", "[SEP]") \
            .replace("家屬1：", "[SEP]").replace("家屬2：", "[SEP]") \
            .replace("個管師：", "[SEP]").replace("護理師：", "[SEP]")
        elif (args.sep_mode == "add"):
            article = article = article.replace("醫師：", "[SEP]醫師：") \
            .replace("民眾：", "[SEP]民眾：").replace("家屬：", "[SEP]家屬：") \
            .replace("家屬1：", "[SEP]家屬1：").replace("家屬2：", "[SEP]家屬2：") \
            .replace("個管師：", "[SEP]個管師：").replace("護理師：", "[SEP]護理師：")

        tokens = tokenizer.tokenize(article)
        if (args.mode == "train"):
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
            if (args.mode == "train"):
                del BIO_label[0], type_label[0]
        tokens.append("[SEP]")
        if (args.mode == "train"):
            BIO_label.append(2)
            type_label.append(0)

        ids = tokenizer.convert_tokens_to_ids(tokens)

        if (args.mode == "train"):
            pt_dict = {'input_ids':ids, "BIO_label":BIO_label, "type_label":type_label,
                        'article_id':data['id']}
        else:
            pt_dict = {'input_ids':ids, 'article_id':data['id']}
        bert_data.append(pt_dict)
        c += 1
        print("\rprocessed %d data" %c, end="")
    
torch.save(bert_data, "./dataset/" + args.file_name + "_bert_data.pt")
print("")

"""to length 512"""
bert_data_train_512 = []
bert_data_test_512 = []
c = 0
c1 = 0
c2 = 0
length = len(bert_data)
test_list = set(eval(args.test_list)) if args.test_list != "[]" else sample(range(length), ceil(length * args.test_percent))

for data in tqdm(bert_data):
    c += 1
    ids = data['input_ids']
    if (args.mode == "train"):
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
        if (args.seg_mode == 1):
            i = 0
            flag = 0
            while (i < 512):
                if (ids_512[i] == tokenizer.vocab['醫'] or ids_512[i] == tokenizer.vocab['個'] \
                    or ids_512[i] == tokenizer.vocab['護']):
                    flag = 0
                elif (ids_512[i] == tokenizer.vocab['民'] or ids_512[i] == tokenizer.vocab['家']):
                    flag = 1
                seg_512[i] = flag
                i += 1
            

        att_512 = [1] * (sep_pos + 2) + [0] * (512 - sep_pos - 2)
        if (args.mode == "train"):
            BIO_512 = [2] + BIO[pos : pos+sep_pos+1] + [2] * (512 - sep_pos - 2)
            type_512 = [0] + type_[pos : pos+sep_pos+1] + [0] * (512 - sep_pos - 2)
        flag = 1 if (pos+sep_pos+1 >= len(ids)) else 0
        pos = new_pos

        if (args.mode == "train"):
            pt_dict = {"input_ids":ids_512, "seg":seg_512, "att":att_512,
                        "BIO_label":BIO_512, "type_label":type_512,
                        "article_id":data['article_id']}
        else:
            pt_dict = {"input_ids":ids_512, "seg":seg_512, "att":att_512,
                        "article_id":data['article_id']}

        if (data['article_id'] not in test_list and args.mode=="train"):
            bert_data_train_512.append(pt_dict)
            c1 += 1
        else:
            bert_data_test_512.append(pt_dict)
            c2 += 1
        
        print("\rprocessed %d data to length 512" %(c1+c2), end="")

        if (flag): # read single talk
            break

torch.save(bert_data_train_512, "./dataset/" + args.file_name + "_train_512_bert_data.pt")
torch.save(bert_data_test_512, "./dataset/" + args.file_name + "_test_512_bert_data.pt")

print("")
print("processed %d origin datas to %d train datas and %d test datas in length 512"
    % (c, c1, c2))
print("Preprocess Done !!")
print("Testing set id list: ", test_list)