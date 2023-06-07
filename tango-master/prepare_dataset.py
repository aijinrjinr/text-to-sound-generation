import pandas as pd
import os
import json

path = r'/home/qywei/TTSS/audiocaps/val.csv'
info = pd.read_csv(path)
idx = info.values[:, 0]
caps = info.values[:, -1]
# st = '['
# for i in range(idx.shape[0]):
#     name = idx[i]
#     st += '{"dataset": "audiocaps", "location": "/home/qywei/TTSS/audiocaps/train/' + str(name) + '.wav"' + ', "captions": "' + caps[i] + '"},'
# # st += ']'
# data = json.loads(st[:-1] + ']')
with open('data/valid_audiocaps_new.json', 'w') as file:
    for i in range(idx.shape[0]):
        name = idx[i]
        st = '{"dataset": "audiocaps", "location": "/home/qywei/TTSS/audiocaps/val/' + str(name) + '.wav"' + ', "captions": "' + caps[i] + '"}\n'
        file.write(st)
file.close()

a = 1