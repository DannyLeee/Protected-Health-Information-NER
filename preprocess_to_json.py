import json
import sys

if (len(sys.argv) != 4):
	print("usage: python preprocess_to_json.py {mode} {src_file_path} {des_file_path}")
	exit(-1)

mode = sys.argv[1]
sourse_filename = sys.argv[2]
des_filename = sys.argv[3]

assert mode in ['train', 'dev', 'test']

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