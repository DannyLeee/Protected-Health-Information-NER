import torch
import numpy as np

data_path = "./dataset/train_1_bert_data.pt"
# sample
# Max len: 3350   Min len: 801    Avg: 1590
# BIO count
# 0: 446 1.078493   1: 792 1.915171 2: 40116 97.006336
# [89.94618834 50.65151515  1.        ]

# type count
# 0: 40116 97.006336    1: 54 0.130580  2: 106 0.256323 3: 850 2.055424 4: 0 0.000000
# 5: 0 0.000000     6: 6 0.014509   7: 0 0.000000   8: 0 0.000000   9: 0 0.000000
# 10: 0 0.000000    11: 0 0.000000  12: 0 0.000000  13: 20 0.048363 14: 0 0.000000
# 15: 73 0.176525   16: 0 0.000000  17: 129 0.311941    18: 0 0.000000
# [1.00000000e+00 7.42883207e+02 3.78451358e+02 4.71952716e+01
#  9.70063365e+07 9.70063365e+07 6.68553928e+03 9.70063365e+07
#  9.70063365e+07 9.70063365e+07 9.70063365e+07 9.70063365e+07
#  9.70063365e+07 2.00575855e+03 9.70063365e+07 5.49531139e+02
#  9.70063365e+07 3.10975750e+02 9.70063365e+07]

# train_1
# Max len: 6814   Min len: 490    Avg: 1841
# BIO count
# 0: 2184 0.988446    1: 4009 1.814413    2: 214760 97.197141
# [98.33333333 53.5694687   1.        ]
# type count
# 0: 214760 97.197141 1: 448 0.202758 2: 398 0.180129 3: 4442 2.010382    4: 49 0.022177
# 5: 19 0.008599  6: 43 0.019461  7: 0 0.000000   8: 55 0.024892  9: 8 0.003621
# 10: 3 0.001358  11: 54 0.024440 12: 0 0.000000  13: 2 0.000905  14: 6 0.002716
# 15: 200 0.090517    16: 0 0.000000  17: 466 0.210905    18: 0 0.000000
# [1.00000000e+00 4.79372641e+02 5.39595000e+02 4.83475676e+01
#  4.38265956e+03 1.13018437e+04 4.99416203e+03 9.71971425e+07
#  3.90457045e+03 2.68375880e+04 7.15339819e+04 3.97687436e+03
#  9.71971425e+07 1.07261502e+05 3.57801575e+04 1.07378815e+03
#  9.71971425e+07 4.60856189e+02 9.71971425e+07]

data = torch.load(data_path)

token_len = [0] * len(data)
BIO_count = [0] * 3
type_count = [0] * 19

for i, d in enumerate(data):
    token_len[i] += len(d['input_ids'])
    for B in d['BIO_label']:
        BIO_count[B] += 1
    for T in d['type_label']:
        type_count[T] += 1

token_len = np.array(token_len)
BIO_count = np.array(BIO_count)
type_count = np.array(type_count)

print("Max len: %d\tMin len: %d\tAvg: %d"
        % (token_len.max(), token_len.min(), token_len.mean()))

BIO_dis = []
print("BIO count")
for i, c in enumerate(BIO_count):
    print("%d: %d %f" % (i, c, c/BIO_count.sum() * 100))
    BIO_dis.append(c/BIO_count.sum() * 100)

BIO_dis = np.array(BIO_dis)
print(BIO_dis.max() / BIO_dis)

type_dis = []
print("type count")
for i, c in enumerate(type_count):
    print("%d: %d %f" % (i, c, c/type_count.sum() * 100))
    type_dis.append(c/type_count.sum() * 100)

type_dis = np.array(type_dis)
type_dis += 0.000001
print(type_dis.max() / type_dis)

