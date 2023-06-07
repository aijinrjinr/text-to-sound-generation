import os
from shutil import copyfile
path = r'/home/qywei/TTSS/audiocaps/OUTPUT/caps_train_vgg_pred_CLIP/caps_validation/'
filelist = os.listdir(path)
filelist = [f for f in filelist if f[-1] == 'v']
filelist1 = [f for f in filelist if f[-5] == '0']
filelist2 = [f for f in filelist if f[-5] == '1']
SAVEP1 = r'/home/qywei/tango/outputs/test/Diffsound_sample0/'
SAVEP2 = r'/home/qywei/tango/outputs/test/Diffsound_sample1/'
for f in filelist1:
    copyfile(path + f, SAVEP1 + 'output_' + f[:-17] + '.wav')
for f in filelist2:
    copyfile(path + f, SAVEP2 + 'output_' + f[:-17] + '.wav')

a = 1