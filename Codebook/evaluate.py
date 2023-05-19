'''
Adapted from `https://github.com/toshas/torch-fidelity`.
Modified by Dongchao Yang, 2022.
'''
import itertools
import multiprocessing
import os
from pathlib import Path
from specvqgan.util import get_ckpt_path

import torch
import torch.distributed as dist
import torchvision
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from evaluation.metrics.fid import calculate_fid
from evaluation.metrics.isc import calculate_isc
from evaluation.metrics.kid import calculate_kid
from evaluation.metrics.kl import calculate_kl
from train import instantiate_from_config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torchvision
import librosa
import numpy as np

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


def patch_cfg_for_new_paths(nested_dict, patch):
    if patch is None:
        print('Nothing to patch')
        return nested_dict
    if not isinstance(nested_dict, (dict, DictConfig)):
        # print('Ignoring the in patching', nested_dict, type(nested_dict))
        return nested_dict
    for key, value in nested_dict.items():
        if isinstance(value, (dict, DictConfig)):
            nested_dict[key] = patch_cfg_for_new_paths(value, patch)
        elif isinstance(value, (list, ListConfig)):
            for i, element in enumerate(value):
                value[i] = patch_cfg_for_new_paths(element, patch)
        else:
            if key in patch:
                print(f'Patched {key}: {nested_dict[key]} --> {patch[key]}')
                nested_dict[key] = patch.get(key, nested_dict[key])
    return nested_dict

def get_dataset_class(dataset_cfg):
    if 'target' in dataset_cfg:
        dataset_class = instantiate_from_config(dataset_cfg)
    elif 'path_to_exp' in dataset_cfg: # 从以前的配置中，获得validation dataset的相关信息
        print('dataset_cfg ', dataset_cfg)
        dataset_class = instantiate_from_config(dataset_cfg.exp_dataset) # train.ConditionedSpectrogramDataModuleFromConfig_caps
        dataset_class.prepare_data()
        dataset_class.setup() # key: validation
        dataset_class = dataset_class.datasets[dataset_cfg.key] # specvqgan.data.caps.VASSpecsCondOnFeatsValidation 
        # print('dataset_class ', dataset_class)
    else:
        raise NotImplementedError
    return dataset_class


def get_featuresdict(feat_extractor, device, dataset_cfg, is_ddp, batch_size, save_cpu_ram):
    # print('dataset_cfg ',dataset_cfg)
    input = get_dataset_class(dataset_cfg)
    # assert 1==2
    # for debugging
    # input.specs_dataset.dataset = input.specs_dataset.dataset[:1000]
    # input.feats_dataset.dataset = input.feats_dataset.dataset[:1000]
    batch_size = min(batch_size, len(input))
    if dataset_cfg.transform_dset_out_to_inception_in is not None:
        transforms = [instantiate_from_config(c) for c in dataset_cfg.transform_dset_out_to_inception_in]
    else:
        transforms = [lambda x: x]
    transforms = torchvision.transforms.Compose(transforms)
    if is_ddp:
        sampler = DistributedSampler(input, dist.get_world_size(), dist.get_rank(), shuffle=False)
        num_workers = 0
    else:
        sampler = None
        num_workers = 0 if save_cpu_ram else min(8, 2 * multiprocessing.cpu_count())

    dataloader = DataLoader(
        input,
        batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device != 'cpu',
    )
    out = None
    out_meta = None
    for batch in tqdm(dataloader):
        # saving batch meta so that we could merge predictions from both datasets when
        # pair-wise KL is calculated
        # comenting out target and label, because those are asigned by folder name, not original labels
        metadict = {
            # 'target': batch['target'].cpu().tolist(),
            # 'label': batch['label'],
            'file_path_': batch.get('file_path_', batch.get('file_path_specs_')),
        }
        batch = transforms(batch)
        batch = batch.to(device)

        with torch.no_grad():
            features = feat_extractor(batch)

        featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}
        # logits_unbiased, 2048, logits

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

        if out_meta is None:
            out_meta = metadict
        else:
            out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    # we need to send the results from all ranks to one of them (gather). Otherwise, the metric is calculated
    # only on the subset of data that one worker had
    if is_ddp:
        for k, v in out.items():
            gather_out = [None for worker in range(dist.get_world_size())]
            dist.all_gather_object(gather_out, v)
            out[k] = torch.cat(gather_out)
        for k, v in out_meta.items():
            gather_out_meta = [None for worker in range(dist.get_world_size())]
            dist.all_gather_object(gather_out_meta, v)
            # just flattens the list
            out_meta[k] = list(itertools.chain(*gather_out_meta))
    # merging both dicts key-wise
    out = {**out, **out_meta}
    return out

def get_featuresdict_audio(feat_extractor, device, dataset_cfg, is_ddp, batch_size, save_cpu_ram):
    # print('dataset_cfg ',dataset_cfg)
    input = get_dataset_class(dataset_cfg)
    # assert 1==2
    # for debugging
    # input.specs_dataset.dataset = input.specs_dataset.dataset[:1000]
    # input.feats_dataset.dataset = input.feats_dataset.dataset[:1000]
    batch_size = min(batch_size, len(input))
    if dataset_cfg.transform_dset_out_to_inception_in is not None:
        transforms = [instantiate_from_config(c) for c in dataset_cfg.transform_dset_out_to_inception_in]
    else:
        transforms = [lambda x: x]
    transforms = torchvision.transforms.Compose(transforms)
    if is_ddp:
        sampler = DistributedSampler(input, dist.get_world_size(), dist.get_rank(), shuffle=False)
        num_workers = 0
    else:
        sampler = None
        num_workers = 0 if save_cpu_ram else min(8, 2 * multiprocessing.cpu_count())

    dataloader = DataLoader(
        input,
        batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device != 'cpu',
    )
    out = None
    out_meta = None
    for batch in tqdm(dataloader):
        # saving batch meta so that we could merge predictions from both datasets when
        # pair-wise KL is calculated
        # comenting out target and label, because those are asigned by folder name, not original labels
        metadict = {
            # 'target': batch['target'].cpu().tolist(),
            # 'label': batch['label'],
            'file_path_': batch.get('file_path_', batch.get('file_path_specs_')),
        }
        batch['image'] = torch.tensor(np.stack([TRANSFORMS(batch['image'][i].numpy()) for i in range(batch['image'].shape[0])]))
        batch = transforms(batch)
        batch = batch.to(device)

        with torch.no_grad():
            features = feat_extractor(batch)

        featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}
        # logits_unbiased, 2048, logits

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

        if out_meta is None:
            out_meta = metadict
        else:
            out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    # we need to send the results from all ranks to one of them (gather). Otherwise, the metric is calculated
    # only on the subset of data that one worker had
    if is_ddp:
        for k, v in out.items():
            gather_out = [None for worker in range(dist.get_world_size())]
            dist.all_gather_object(gather_out, v)
            out[k] = torch.cat(gather_out)
        for k, v in out_meta.items():
            gather_out_meta = [None for worker in range(dist.get_world_size())]
            dist.all_gather_object(gather_out_meta, v)
            # just flattens the list
            out_meta[k] = list(itertools.chain(*gather_out_meta))
    # merging both dicts key-wise
    out = {**out, **out_meta}
    return out

def main():
    torch.manual_seed(0)
    local_rank = os.environ.get('LOCAL_RANK')
    cfg_cli = OmegaConf.from_cli() # 从命令行中得到的参数
    cfg_cli.config = '/home/wei/TTSS/Codebook/evaluation/configs/eval_melception_caps.yaml'
    cfg_eval = OmegaConf.load(cfg_cli.config) # 从config文件中获得的
    EXPERIMENT_PATH = '/home/wei/TTSS/Diffsound'
    SPEC_DIR_PATH = "/data/wei/audiocaps/features/test/"#melspec_10s_22050hz/"
    cls_token_dir_path = "/data/wei/audiocaps/clip_text/test/cls_token_512/"
    # SAMPLES_FOLDER = "caps_validation"
    # # SPLITS="\"[validation, ]\""
    # SPLITS = "[validation]"
    # SAMPLER_BATCHSIZE = 32  # 32 previous
    # SAMPLES_PER_VIDEO = 10
    # TOP_K = 128  # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook, 128 for caps?
    # NOW = `date + "%Y-%m-%dT%H-%M-%S"`
    # DATASET = 'caps'
    # NO_CONDITION = False  # False indicate using condition
    cfg_cli.input2, cfg_cli.patch, cfg_cli.input1, cfg_cli.input1.params = {}, {}, {}, {}
    cfg_cli.input1.path_to_exp = None
    cfg_cli.input2.path_to_exp = EXPERIMENT_PATH
    cfg_cli.patch.specs_dir = SPEC_DIR_PATH
    cfg_cli.patch.spec_dir_path = SPEC_DIR_PATH
    cfg_cli.patch.cls_token_dir_path = cls_token_dir_path
    cfg_cli.input1.params.root = '/data/wei/audiocaps/OUTPUT/caps_train_vgg_pre/Real_vgg_pre_399_samples_2023-05-14T23-55-02/'
    # cfg_cli.input1.train_ids_path = '/data/wei/audiocaps/test.csv'
    # print('cfg_eval ', cfg_eval)
    # the latter arguments are prioritized
    # 'python Codebook/evaluate.py \
    #     config=Codebook/evaluation/configs/eval_melception_${DATASET,,}.yaml \
    #     cfg_cli.input2.path_to_exp=$EXPERIMENT_PATH \
    #     patch.specs_dir=$SPEC_DIR_PATH \
    #     patch.spec_dir_path=$SPEC_DIR_PATH \
    #     patch.cls_token_dir_path=$cls_token_dir_path \
    #     input1.params.root=$ROOT'
    for dataset in ['input1', 'input2']:
        cli_dataset_cfg = cfg_cli[dataset] # input1.params.root=$ROOT
        # print(cli_dataset_cfg)
        # first I check if the path_to_exp is specified in CLI args
        if cli_dataset_cfg is not None:
            if cli_dataset_cfg.path_to_exp is not None: # input2会通过这里
                cfg_paths = Path(cli_dataset_cfg.path_to_exp).glob('configs/*-project.yaml') # 加载以前的config
                cfgs_dataset = [OmegaConf.load(p) for p in sorted(list(cfg_paths))]
                # cfg_eval[dataset].exp_dataset = OmegaConf.merge(*cfgs_dataset).data # 合并到exp_dataset属性下
                # print('cfg_eval ', cfg_eval[dataset].exp_dataset)
                # assert 1==2
            else:
                assert cli_dataset_cfg.params.root is not None, 'path_to_exp or root should be specified'
        # if not specified in CLI, I will check the default config
        else:
            eval_dataset_cfg = cfg_eval[dataset]
            if eval_dataset_cfg.path_to_exp is not None:
                cfg_paths = Path(eval_dataset_cfg.path_to_exp).glob('configs/*-project.yaml')
                cfgs_dataset = [OmegaConf.load(p) for p in sorted(list(cfg_paths))] 
                cfg_eval[dataset].exp_dataset = OmegaConf.merge(*cfgs_dataset).data
            else:
                assert eval_dataset_cfg.params.root is not None, 'path_to_exp or root should be specified'
    cfg = OmegaConf.merge(cfg_eval, cfg_cli) # 合并
    cfg = patch_cfg_for_new_paths(cfg, cfg.patch)

    assert cfg.have_isc or cfg.have_fid or cfg.have_kid, 'Select at least one metric'
    assert (not cfg.have_fid) and (not cfg.have_kid) or cfg.input2 is not None, 'Two inputs are required'

    if local_rank is not None:
        is_ddp = True
        local_rank = int(local_rank)
        dist.init_process_group(cfg.get('dist_backend', 'nccl'), 'env://')
        print(f'WORLDSIZE {dist.get_world_size()} – RANK {dist.get_rank()}')
        if dist.get_rank() == 0:
            print('MASTER:', os.environ['MASTER_ADDR'], ':', os.environ['MASTER_PORT'])
            print(OmegaConf.to_yaml(cfg))
    else:
        is_ddp = False
        local_rank = cfg.device[-1]  # extracting last elements from e.g. 'cuda:0'[-1]
        print(OmegaConf.to_yaml(cfg))

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # downloading the checkpoint for melception
    #get_ckpt_path('melception', 'evaluation/logs/21-05-10T09-28-40')
    # print('cfg.feature_extractor ',cfg.feature_extractor)
    # assert 1==2
    cfg.input1.transform_dset_out_to_inception_in.pop(1)#[1]['params'].train_ids_path = '/data/wei/audiocaps/test.csv'
    feat_extractor = instantiate_from_config(cfg.feature_extractor) # get feature extractor
    feat_extractor.eval()
    feat_extractor.to(device)

    out = {}

    print('Extracting features from input_1')
    featuresdict_1 = get_featuresdict(feat_extractor, device, cfg.input1, is_ddp, **cfg.extraction_cfg)

    featuresdict_2 = None
    if cfg.input2 not in ['None', None, 'null', 'none']:
        print('Extracting features from input_2')
        cfg.input2.transform_dset_out_to_inception_in.pop(1)  #
        featuresdict_2 = get_featuresdict_audio(feat_extractor, device, cfg.input2, is_ddp, **cfg.extraction_cfg)
    
    # print('featuresdict_1 ', featuresdict_1.keys())
    # print('featuresdict_2 ',featuresdict_2.keys())
    # assert 1==2
    # pickle.dump(featuresdict_1, open('feats1.pkl', 'wb'))
    # pickle.dump(featuresdict_2, open('feats2.pkl', 'wb'))
    # featuresdict_1 is the ground truth
    if cfg.have_kl:
        metric_kl = calculate_kl(featuresdict_1, featuresdict_2, **cfg.kl_cfg)
        out.update(metric_kl)
    if cfg.have_isc:
        metric_isc = calculate_isc(featuresdict_1, **cfg.isc_cfg)
        out.update(metric_isc)
    if cfg.have_fid:
        metric_fid = calculate_fid(featuresdict_1, featuresdict_2, **cfg.fid_cfg)
        out.update(metric_fid)
    if cfg.have_kid:
        metric_kid = calculate_kid(featuresdict_1, featuresdict_2, **cfg.kid_cfg)
        out.update(metric_kid)

    print('\n'.join((f'{k}: {v:.7f}' for k, v in out.items())))

    # just pretty printing of the results, nothing more
    print(
        f'{cfg.input1.get("path_to_exp", Path(cfg.input1.params.root).parent.stem)}:',
        f'KL: {out.get("kullback_leibler_divergence", float("nan")):8.5f};',
        f'ISc: {out.get("inception_score_mean", float("nan")):8.5f} ({out.get("inception_score_std", float("nan")):5f});',
        f'FID: {out.get("frechet_inception_distance", float("nan")):8.5f};',
        f'KID: {out.get("kernel_inception_distance_mean", float("nan")):.5f}',
        f'({out.get("kernel_inception_distance_std", float("nan")):.5f})'
    )


if __name__ == '__main__':
    main()
