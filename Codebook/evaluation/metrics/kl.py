import torch
from pathlib import Path

def path_to_sharedkey(path, dataset_name, classes=None):
    if dataset_name.lower() == 'vggsound':
        # a generic oneliner which extracts the unique filename for the dataset.
        # Works on both FakeFolder and VGGSound* datasets
        sharedkey = Path(path).stem.replace('_mel', '').split('_sample_')[0]
    elif dataset_name.lower() == 'vas':
        # in the case of vas the procedure is a bit more tricky and involves relying on the premise that
        # the folder names (.../VAS_validation/cls_0, .../cls_1 etc) are made after enumerating sorted list
        # of classes.
        classes = sorted(classes)
        target_to_label = {f'cls_{i}': c for i, c in enumerate(classes)}
        # replacing class folder with the name of the class to match the original dataset (cls_2 -> dog)
        for folder_cls_name, label in target_to_label.items():
            path = path.replace(folder_cls_name, label).replace('melspec_10s_22050hz/', '')
        # merging video name with class name to make a unique shared key
        sharedkey = Path(path).parent.stem + '_' + Path(path).stem.replace('_mel', '').split('_sample_')[0]
    elif dataset_name.lower() == 'caps': # stem : 获取/.之间的部分
        sharedkey = Path(path).stem.replace('_mel', '').split('_sample_')[0] # 获得原文件名称
    else:
        raise NotImplementedError
    return sharedkey

def calculate_kl(featuresdict_1, featuresdict_2, feat_layer_name, dataset_name, classes=None):
    test_input(featuresdict_1, featuresdict_2, feat_layer_name, dataset_name, classes)

    print('KL: Assuming that `input2` is "pseudo" target and `input1` is prediction. KL(input2_i||input1_i)')

    EPS = 1e-6

    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]
    # print('features_1 ',features_1.shape) # the predicted (num*10, class_num)
    # print('features_2 ',features_2.shape) # the true
    paths_1 = featuresdict_1['file_path_']
    paths_2 = featuresdict_2['file_path_']
    # print('paths_1 ',len(paths_1)) its path
    # print('paths_2 ',len(paths_2))
    path_to_feats_1 = {p: f for p, f in zip(paths_1, features_1)}
    path_to_feats_2 = {p: f for p, f in zip(paths_2, features_2)}
    # dataset_name: caps
    # in input1 (fakes) can have multiple samples per video, while input2 has only one real
    sharedkey_to_feats_1 = {path_to_sharedkey(p, dataset_name, classes): [] for p in paths_1}
    sharedkey_to_feats_2 = {path_to_sharedkey(p[:-10], dataset_name, classes): path_to_feats_2[p] for p in paths_2}
    # TODO: assert all 1s in 2 and their size is the same

    # grouping samples by vid name
    for path, features in path_to_feats_1.items(): # 同一文件名的sample直接放一个group里面
        sharedkey_to_feats_1[path_to_sharedkey(path, dataset_name, classes)] += [features]

    features_1 = []
    features_2 = []
    for sharedkey, feat_2 in sharedkey_to_feats_2.items():
        features_1.extend(sharedkey_to_feats_1[sharedkey]) # 在将生成的特征，直接放features_1 
        # just replicating the ground truth logits to compare with multiple samples in prediction
        samples_num = len(sharedkey_to_feats_1[sharedkey])
        features_2.extend([feat_2] * samples_num) # [[],[],[]]

    features_1 = torch.stack(features_1, dim=0)
    features_2 = torch.stack(features_2, dim=0)
    # print('features_1 ',features_1.shape)
    # print('features_2 ',features_2.shape)
    # assert 1==2
    #features_1 = features_1.sigmoid()
    features_1 = features_1.softmax(dim=1)
    # features_1 = features_1.unsqueeze(2) # for multiple classification
    # features_1 = torch.cat([features_1,1-features_1],dim=2).view(-1,2)

    features_2 = features_2.softmax(dim=1)

    # features_2 = features_2.unsqueeze(2)
    # features_2 = torch.cat([features_2,1-features_2],dim=2).view(-1,2)

    # KL(features_2_i || features_1_i)
    kl = torch.nn.functional.kl_div((features_1 + EPS).log(), features_2, reduction='sum') / len(features_1)
    return {'kullback_leibler_divergence': float(kl)}

def test_input(featuresdict_1, featuresdict_2, feat_layer_name, dataset_name, classes):
    assert feat_layer_name == 'logits', 'This KL div metric is implemented on logits.'
    assert 'file_path_' in featuresdict_1 and 'file_path_' in featuresdict_2, 'File paths are missing'
    assert len(featuresdict_1) >= len(featuresdict_2), 'There are more samples in input1, than in input2'
    assert len(featuresdict_1) % len(featuresdict_2) == 0, 'Size of input1 is not a multiple of input1 size.'
    if dataset_name == 'vas':
        assert classes is not None, f'Specify classes if you are using vas dataset. Now `classes` – {classes}'
        print('KL: when FakesFolder on VAS is used as a dataset, we assume the original labels were sorted',
              'to produce the target_ids. E.g. `baby` -> `cls_0`; `cough` -> `cls_1`; `dog` -> `cls_2`.')


if __name__ == '__main__':
    # p = torch.tensor([0.25, 0.25, 0.25, 0.25]).view(1, 4)
    # q = torch.tensor([0.25, 0.25, 0.25, 0.25]).view(1, 4)
    # 0.

    p = torch.tensor([0.5, 0.6, 0.7]).view(3, 1)
    p_ = 1-p
    p = torch.cat([p,p_],dim=1).view(-1,2)
    print(p)
    q = torch.tensor([0.5, 0.6, 0.7]).view(3, 1)
    q_ = 1-q
    q = torch.cat([q,q_],dim=1).view(-1,2)
    print(q.shape)
    kl = torch.nn.functional.kl_div(torch.log(q), p, reduction='sum')
    # 0.0853

    print(kl)
