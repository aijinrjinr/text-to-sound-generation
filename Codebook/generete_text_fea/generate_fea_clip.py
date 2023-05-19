import torch
from pybert.configs.basic_config import config
from pybert.io.bert_processor import BertProcessor
from pybert.model.bert_for_multi_label import BertForMultiLable
import pickle
import numpy as np
import pandas as pd
import clip
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def get_text_features(text):
    text = clip.tokenize(text).to(device) # ["a diagram with dog", "a dog", "a cat"]
    # print('text ',text)
    with torch.no_grad():
        # image_features = model.encode_image(image)
        # print('image_features ',image_features.shape)
        text_features = model.encode_text(text)
        text_features = text_features.squeeze()
        # print('text_features ',text_features.shape)
        # assert 1==2
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        text_features = text_features.detach().cpu().numpy()
    return text_features

if __name__ == "__main__":
    for suffix in ['train', 'test', 'val']:
        # suffix = 'train'
        # data = pd.read_csv('/data/wei/audiocaps/test.csv',sep='\t',usecols=[0,1,2])
        data = pd.read_csv('/data/wei/audiocaps/'+suffix+'.csv',sep=',',usecols=[0,1,3])
        save_ls = []
        for row in data.values:
            text = row[2]#[1]
            # print('text ',text)
            # assert 1==2
            name = str(row[0])#[:-4]
            # text = ''''"FUCK YOUR FILTHY MOTHER IN THE ASS, DRY!"'''
            pre_path = '/data/wei/audiocaps/clip_text/'+suffix+'/cls_token_512/'
            os.makedirs(pre_path, exist_ok=True)
            # probs = main(text,arch,max_seq_length,do_loer_case)
            probs = get_text_features(text)
            # print('probs ',probs.shape)
            # assert 1==2
            # print('probs ',probs.shape)
            print(pre_path+name)
            if pre_path+name not in save_ls:
                save_ls.append(pre_path+name)
                k = 1
                np.savetxt(pre_path+name+str(k)+'.txt',probs)
            else:
                k += 1
                np.savetxt(pre_path+name+str(k)+'.txt',probs)
            # assert 1==2

