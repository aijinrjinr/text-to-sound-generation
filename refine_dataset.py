import pandas as pd
import numpy as np

path_new = r'/home/wei/TTSS/Diffsound/data_root/audiocaps/new_test.csv'
newtest = []
data2 = pd.read_csv(path_new)
cap2 = list(data2.values[:, 1])
for suffix in ['train', 'val', 'test']:

    path = '/data/wei/audiocaps/val.csv'


    data1 = pd.read_csv(path)


    pid1 = list(data1.values[:, 0])
    cap1 = list(data1.values[:, 3])

    overlap = [i for i, j in enumerate(cap1) if j in cap2]
    newtest.extend([pid1[i] for i in overlap])
a = 1