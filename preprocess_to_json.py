import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, choices=['train', 'dev', 'test'], default="train")
parser.add_argument("-sourse_path", type=str, required=True, help="raw data (.txt)")
parser.add_argument("-des_path", type=str, required=True, help="output data (.json)")
args = parser.parse_args()

mode = args.mode
sourse_filename = args.sourse_path
des_filename = args.des_path

result = []
flag = 0
with open(sourse_filename,'r') as f:
	if (mode == "train"):
		data = {
			'article':'',
			'id':0,
			'item':[],
		}
		for line in f:
			line = line.replace('\n','')
			if line=='':
				continue
			if line == '--------------------':
				flag = 0
				result.append(data)
				data['id'] = data['item'][0][0]
				data = {
					'article':'',
					'id':0,
					'item':[],
				}
				continue
			if flag == 0:
				data['article']= line
				flag = 1
			elif flag ==1:
				flag = 2
			elif flag ==2:
				row = line.split('	')
				data['item'].append([int(row[0]),int(row[1]),int(row[2]),row[3],row[4]])
	else:
		data = {
			'id':0,
			'article':''
		}
		for line in f:
			line = line.replace('\n','')
			if (line.find("article_id:") != -1):
				data['id'] = int(line.split()[-1])
			elif (line.find("：") != -1):
				data['article'] = line
			elif (line == '--------------------'):
				result.append(data)
				data = {
					'id':0,
					'article':''
				}

with open(des_filename,'w', encoding='utf8') as f:
	json.dump(result,f,ensure_ascii=False)