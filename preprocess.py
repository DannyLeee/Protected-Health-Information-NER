import json
import numpy as np
import torch
type_dict = {"none":0, "name":1, "location":2, "time":3, "contact":4,
             "ID":5, "profession" : 6, "biomarker":7, "family": 8,
             "clinical_event": 9, "special_skills":10, "unique_treatment":11,
             "account":12, "organization":13, "education":14, "money":15,
             "belonging_mark":16, "med_exam":17, "others":18}
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')

with open ('./train_1.json', 'r') as json_file:
    bert_data = []
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
        .replace("個管師", "[SEP]").replace("護理師：", "[SEP]")
        tokens = tokenizer.tokenize(article)
        BIO_lable = np.full(len(tokens), 2)
        type_label = np.full(len(tokens), 0)
        flag = 0
        begin = 0
        j = 0
        remove_list = []
        for i in range(len(tokens)):
            if (tokens[i] == '_'):
                remove_list.append(i)
                if (flag == 0):
                    begin = i+1
                    flag += 1
                else:
                    BIO_lable[begin] = 0
                    BIO_lable[begin+1 : i] = 1
                    type_label[begin : i] = type_list[j]
                    j += 1
                    flag = 0

        BIO_lable = BIO_lable.tolist()
        type_label = type_label.tolist()

        for i in sorted(remove_list, reverse=True):
            del tokens[i], BIO_lable[i], type_label[i]
        
        tokens[0] = "[CLS]"
        tokens.append("[SEP]")
        BIO_lable.append(2)
        type_label.append(0)

        ids = tokenizer.convert_tokens_to_ids(tokens)

        pt_dict = {'input_ids':ids, "BIO_lable":BIO_lable, "type_lable":type_label}
        bert_data.append(pt_dict)
        c += 1
        print("\rprocessed %d data" %c, end="")
    
    torch.save(bert_data, "train_1_bert_data.pt")