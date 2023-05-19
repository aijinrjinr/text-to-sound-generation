from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from sound_synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle
from specvqgan.modules.losses.vggishish.transforms import Crop
import torch
import pandas as pd
import torchvision
import librosa

class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels)

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec

class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)

class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val

class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val

class Multiply(object):
    def __init__(self, val, inverse=False) -> None:
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val

class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10 ** x
        else:
            return np.log10(x)

class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)

class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, :self.max_len]

class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)

TRANSFORMS = torchvision.transforms.Compose([
    MelSpectrogram(sr=22050, nfft=1024, fmin=125, fmax=7600, nmels=80, hoplen=1024//4, spec_power=1),
    LowerThresh(1e-5),
    Log10(),
    Multiply(20),
    Subtract(20),
    Add(100),
    Divide(100),
    Clip(0, 1.0),
    TrimSpec(860)
])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class CropImage(Crop):
    def __init__(self, *crop_args):
        super().__init__(*crop_args)

class CapsDataset(Dataset):
    def __init__(self, data_root, phase = 'train', mel_num=80,
                 spec_len=860, spec_crop_len=848, random_crop=False, im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.caps_feature_path = '/data/wei/audiocaps/features'
        if phase=='train':
            tmp_phase='train'
        else:
            tmp_phase='val'
        self.image_folder = os.path.join(self.caps_feature_path, tmp_phase, 'melspec_10s_22050hz')
        self.root = os.path.join(data_root, phase)
        tsv = pd.read_csv(self.root + '.csv', sep=',', usecols=[0, 3])  # usecols=[0,1])
        # filenames = train_tsv['file_name']
        self.name_list = tsv['audiocap_id']
        captions = tsv['caption']
        # self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)
        self.num = len(self.name_list)

        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index] # 
            # print('name ',name)
            # this_text_path = os.path.join(data_root, 'clip_text', phase, 'cls_token_512', str(name)+'1.txt')
            # with open(this_text_path, 'r') as f:
            #     caption = f.readlines()
            self.caption_dict[name] = captions[index]#caption

        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.image_folder, str(name)+'_audio.npy')
        spec = TRANSFORMS(np.load(image_path)) # 加载mel spec
        item = {}
        item['input'] = spec
        if self.transforms is not None: # 
            item = self.transforms(item)
        image = 2 * item['input'] - 1 # why --> it also expects inputs in [-1, 1] but specs are in [0, 1]
        # image = load_img(image_path)
        #image = np.array(image).astype(np.uint8)
        # image = self.transform(image = image)['image']
        image = image[None,:,:]
        caption_list = self.caption_dict[name]
        caption = caption_list.replace('\n', '').lower()#random.choice(caption_list).replace('\n', '').lower()
        # print('image ',image.shape)
        # print('caption ',caption)
        # assert 1==2
        data = {
                'image': image.astype(np.float32),
                'text': caption,
        }
        
        return data


class CapsDatasetAll(Dataset):
    def __init__(self, data_root, phase = 'train', mel_num=80,
                 spec_len=860, spec_crop_len=848, random_crop=False, im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.caps_feature_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/features'
        if phase=='train':
            tmp_phase='train'
        else:
            tmp_phase='val'
        self.image_folder = os.path.join(self.caps_feature_path, tmp_phase, 'melspec_10s_22050hz')
        self.root = os.path.join(data_root, phase)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        self.transforms = CropImage([mel_num, spec_crop_len], random_crop)
        self.num = len(self.name_list)

        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index] # 
            # print('name ',name)
            this_text_path = os.path.join(data_root, 'text', phase, name+'.txt')
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption
        h5_path = '/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/data/audiocaps/feature_h5/'
        self.feats_dict = {}
        for i in range(5):
            name = 'train' + str(i+1) + '.pth'
            tmp_data = torch.load(h5_path + name)
            for k,v in tmp_data.items():
                self.feats_dict[k] = v
        
        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        # image_path = os.path.join(self.image_folder, name+'_mel.npy')
        # spec = np.load(image_path) # 加载mel spec
        spec = self.feats_dict[name]
        item = {}
        item['input'] = spec
        if self.transforms is not None: # 
            item = self.transforms(item)
        image = 2 * item['input'] - 1 # why --> it also expects inputs in [-1, 1] but specs are in [0, 1]
        # image = load_img(image_path)
        #image = np.array(image).astype(np.uint8)
        # image = self.transform(image = image)['image']
        image = image[None,:,:]
        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        # print('image ',image.shape)
        # print('caption ',caption)
        # assert 1==2
        data = {
                'image': image.astype(np.float32),
                'text': caption,
        }
        
        return data