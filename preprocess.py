import json
import numpy as np
import torch
import sys
from transformers import BertTokenizer

type_dict = {"none":0, "name":1, "location":2, "time":3, "contact":4,
             "ID":5, "profession" : 6, "biomarker":7, "family": 8,
             "clinical_event": 9, "special_skills":10, "unique_treatment":11,
             "account":12, "organization":13, "education":14, "money":15,
             "belonging_mark":16, "med_exam":17, "others":18}

if (len(sys.argv) != 2):
    print("usage: python3 preprocess.py {prerpocessed_file_name(without extension)}")
    sys.exit(-1)
PRETRAINED_LM = "hfl/chinese-bert-wwm"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)

bert_data = []
with open ('./dataset/' + sys.argv[1] + '.json', 'r') as json_file:
    data_file = json.load(json_file)
    print("start preprocessing...")
    c = 0
    for data in data_file:
        article = data['article']
        type_list = []
        for i, item in enumerate(data['item']):
            article = article[:item[1] + i*2] + "_" + item[3] + "_" + article[item[2] + i*2:]
            type_list.append(type_dict[item[4]])
        article = article.replace("醫師：", "[SEP]") \
        .replace("民眾：", "[SEP]").replace("家屬：", "[SEP]") \
        .replace("個管師：", "[SEP]").replace("護理師：", "[SEP]")
        tokens = tokenizer.tokenize(article)
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
            del tokens[0], BIO_label[0], type_label[0]
        tokens.append("[SEP]")
        BIO_label.append(2)
        type_label.append(0)

        ids = tokenizer.convert_tokens_to_ids(tokens)

        pt_dict = {'input_ids':ids, "BIO_label":BIO_label, "type_label":type_label}
        bert_data.append(pt_dict)
        c += 1
        print("\rprocessed %d data" %c, end="")
    
torch.save(bert_data, "./dataset/" + sys.argv[1] + "_bert_data.pt")
print("")

"""to length 512"""
bert_data_512 = []
c = 0
c_ = 0
for data in bert_data:
    c += 1
    ids = data['input_ids']
    BIO = data['BIO_label']
    type_ = data['type_label']
    pos = 0
    count_back = 0
    flag = 0
    while (pos < len(ids)):
        ids_512 = ids[pos : pos+512]
        for i in range(min(511, len(ids_512)-1), 0, -1):
            if (ids_512[i] == 102): # 102 = [SEP]
                count_back += 1
                if (count_back == 1):
                    ids_512 = [101] + ids[pos : pos+i+1] + [0] * (512 - i - 2)
                    seg_512 = [0] * 512
                    att_512 = [1] * (i + 2) + [0] * (512 - i - 2)
                    BIO_512 = [2] + BIO[pos : pos+i+1] + [2] * (512 - i - 2)
                    type_512 = [0] + type_[pos : pos+i+1] + [0] * (512 - i - 2)
                    flag = 1 if (pos+i+1 == len(ids)) else 0

                    pt_dict = {'input_ids':ids_512, "seg":seg_512, "att":att_512,
                            "BIO_label":BIO_512, "type_label":type_512}
                    bert_data_512.append(pt_dict)
                    c_ += 1
                    print("\rprocessed %d data to length 512" %c_, end="")

                elif (count_back == 3): # overlap n-1 sentences 
                    pos += i
                    count_back = 0
                    break
        if (flag): # read single talk
            break

torch.save(bert_data_512, "./dataset/sample_512_bert_data.pt")

print("")
print("processed %d origin datas to %d datas in length 512"
    % (c, c_))
print("Preprocess Done !!")