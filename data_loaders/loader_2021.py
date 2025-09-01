import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List, Dict
from pathlib import Path

from common.utils import seed_worker



___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def get_database_path(database_path: Path, track: str) -> Dict[str, Path]:
    # Return ASVspoof 2019 database path
    ASVspoof_2019_path = database_path / 'ASVspoof2019'
    ASVspoof_2021_path = database_path / 'ASVspoof2021'

    paths = {}

    # write flac path
    paths["trn_flac_path"] = ASVspoof_2019_path / "ASVspoof2019_LA_train"
    paths["dev_flac_path"] = ASVspoof_2019_path / "ASVspoof2019_LA_dev"
    paths["eval_flac_path"] = ASVspoof_2021_path / f"ASVspoof2021_{track}_eval"

    # write meta path
    paths["trn_meta_path"] = (ASVspoof_2019_path / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt")
    paths["dev_meta_path"] = (ASVspoof_2019_path / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt")
    paths["eval_meta_path"] = (ASVspoof_2021_path / f"ASVspoof2021_{track}_eval" / f"ASVspoof2021.{track}.cm.eval.trl.txt")

    # ASV score path
    paths['asv_score_dev_path'] = (ASVspoof_2019_path / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt")
    paths['asv_keys_path'] = (ASVspoof_2021_path / "keys")

    return paths


def get_loader(
        seed: int,
        trn_meta_path: Path,
        dev_meta_path: Path,
        eval_meta_path: Path,
        trn_flac_path: Path,
        dev_flac_path: Path,
        eval_flac_path: Path,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_meta_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_flac_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_meta_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_dev(list_IDs=file_dev,
                                            base_dir=dev_flac_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_meta_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval,
                                             base_dir=eval_flac_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_dev(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
            return len(self.list_IDs)

    def __getitem__(self, index):
            
            key = self.list_IDs[index]
            X, fs = sf.read(str(self.base_dir)+'/flac/'+key+'.flac')
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp, key
