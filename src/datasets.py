import os
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


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", device = "cpu") -> None:
        super().__init__()
        
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
            return self.X[i], self.y[i], self.image_features[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]