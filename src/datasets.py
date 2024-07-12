import os
import random
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

import clip
from PIL import Image
import concurrent.futures
import time
from tqdm import tqdm
import gc
from torchvision.transforms import v2


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", device = "cpu", transforms = None) -> None:
        super().__init__()

        self.transforms = transforms
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))

        # チャネル正規化
        mean = self.X.mean(dim=-1, keepdim=True)
        std = self.X.std(dim=-1, keepdim=True)
        self.X = (self.X - mean) / (std + 10**(-6))
        # self.X = (self.X) / (std + 10**(-6))
        
        # self.X = torch.log(torch.abs(torch.fft.rfft(self.X)[:, :, 1:70]) ** 2)

        # # GCN 改善の余地あり
        # mean = self.X.mean(dim=(1, 2), keepdim=True)
        # std = self.X.std(dim=(1, 2), keepdim=True)
        # self.X = (self.X - mean) / (std + 10**(-6))

        # self.X = torch.clamp(self.X, min=-20, max=20)

        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

            if not os.path.exists(os.path.join(data_dir, f"{split}_image_features.pt")):
                self.preprocess_image_features(split, data_dir, device)
            self.image_features = torch.load(os.path.join(data_dir, f"{split}_image_features.pt"))
            self.image_features /= self.image_features.norm(dim=-1, keepdim=True)

        # # 特定の人のデータを用いる
        # subject_id = 0
        # self.X = self.X[self.subject_idxs == subject_id]
        # if split in ["train", "val"]:
        #     self.y = self.y[self.subject_idxs == subject_id]
        # self.subject_idxs = self.subject_idxs[self.subject_idxs == subject_id]

    def preprocess_image_features(self, split, data_dir, device):
        with open(os.path.join(data_dir, f"{split}_image_paths.txt")) as f:
            paths_list = [s.rstrip() for s in f.readlines()]
        for i in range(len(paths_list)):
            tmp = paths_list[i].split('/')
            if len(tmp) == 1:
                paths_list[i] = "_".join(tmp[0].split("_")[:-1]) + "/" + paths_list[i]
        print(len(paths_list))
        print(len(set(paths_list)))

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        def func(image_path):
            return preprocess(Image.open(os.path.join(data_dir, "Images", image_path)))

        image_features = []
        div_n = len(paths_list) // 5000 + 1
        for i in tqdm(range(div_n)):
            read_from = len(paths_list) * i // div_n
            read_to = len(paths_list) * (i + 1) // div_n
            # start = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:  # gpu~5minのはず
                results = executor.map(func, paths_list[read_from:read_to])
                image = [result for result in results]
            # print(time.time() - start)

            image = torch.stack(image, dim=0).to(device)
            with torch.no_grad():
                image_features.append(model.encode_image(image))
        del image; gc.collect()
        image_features = torch.cat(image_features).float().to("cpu")
        torch.save(image_features, os.path.join(data_dir, f"{split}_image_features.pt"))
        print(image_features)
        print(image_features.size())

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            if self.transforms:
                return self.transforms(self.X[i]), self.y[i], self.image_features[i], self.subject_idxs[i]
            else:
                return self.X[i], self.y[i], self.image_features[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    

class TimeShift:
#     def __init__(self, shift_range = None):
#         self.shift_range = shift_range
    
    def __call__(self, tensor):
        nChannels, nSeq = tensor.size()
        # shift = random.randrange(nSeq)
        shift = random.randrange(-4, 5)
        return torch.roll(tensor, shifts=-shift, dims=-1)


class TimeStretch:
    # def __init__(self, ):

    def __call__(self, tensor):
        nChannels, nSeq = tensor.size()
        # seq = random.randrange(int(nSeq*(1-0.3)), int(nSeq*(1+0.3)))
        # seq = np.random.normal(loc=0, scale=0.2)
        # seq = int((1+seq)*nSeq) if seq >= 0 else int(1/(1-seq)*nSeq)
        # seq = max(seq, 1)
        seq = int(2**np.random.normal(scale=0.25) * nSeq)
        # seq = max(int(np.random.normal(loc=1, scale=0.1)*nSeq), 1)
        self.transforms = v2.Resize(size=(271, seq), antialias=True)
        tensor = self.transforms(tensor.view(1, 271, nSeq)).view(271, seq)
        if seq > nSeq:
            start = random.randrange(0, seq-nSeq+1)
            return tensor[:, start:start+nSeq]
        else:
            out = torch.zeros((nChannels, nSeq))
            out[:, :seq] = tensor
            return out


class RandomErasing:
    def __init__(self, p=(0., 0.75)):
        self.p_min = p[0]
        self.p_max = p[1]

    # def __call__(self, tensor):  # pixel毎にランダム消去
    #     nChannels, nSeq = tensor.size()
    #     nErasing = int(nChannels * nSeq * self.p)
    #     indices = np.random.choice(nChannels * nSeq, nErasing, replace=False)
    #     channel_indices = indices // nSeq
    #     seq_indices = indices % nSeq
    #     tensor_copy = tensor.clone()
    #     tensor_copy[channel_indices, seq_indices] = 0.
    #     return tensor_copy
    
    # def __call__(self, tensor):  # 時間連続するランダムpixel消去
    #     n_channels, n_seq = tensor.shape
    #     n_erasing = int(n_seq * self.p)
    #     tensor_copy = tensor.clone()
    #     for c in range(n_channels):
    #         start_idx = np.random.randint(0, n_seq - n_erasing + 1)
    #         tensor_copy[c, start_idx:start_idx + n_erasing] = 0
    #     return tensor_copy

    def __call__(self, tensor):
        p = np.random.uniform(self.p_min, self.p_max)
        n_channels, n_seq = tensor.shape
        n_erasing = int(n_channels * p)
        indices = np.random.choice(n_channels, n_erasing, replace=False)
        tensor_copy = tensor.clone()
        tensor_copy[indices] = 0.
        return tensor_copy
    