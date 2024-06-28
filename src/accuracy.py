import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self, device, top_k=10):
        super().__init__()
        self.top_k = top_k

        with open("data/train_image_paths.txt") as f:
            paths_list = [s.rstrip() for s in f.readlines()]
        for i in range(len(paths_list)):
            tmp = paths_list[i].split('/')
            if len(tmp) == 1:
                paths_list[i] = "_".join(tmp[0].split("_")[:-1]) + "/" + paths_list[i]

        loaded = set()
        unique_i = []
        for i, path in enumerate(paths_list):
            if path not in loaded:
                unique_i.append(i)
                loaded.add(path)
        unique_i = torch.tensor(unique_i)

        self.y = torch.load("data/train_y.pt")
        self.y = self.y[unique_i].to(device)
        self.image_features = torch.load("data/train_image_features.pt")
        self.image_features = self.image_features[unique_i].to(device)
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)

    def forward(self, y_pred, y):
        y_pred /= y_pred.norm(dim=-1, keepdim=True)
        # y /= y.norm(dim=-1, keepdim=True)
        similarity = (y_pred @ self.image_features.T)  # .softmax(dim=-1)
        sim_argsort = torch.argsort(-similarity, dim=1)
        pred = torch.gather(self.y.expand(len(sim_argsort), -1), 1, sim_argsort)

        score = 0
        pred = pred.tolist()
        for i in range(min(len(pred), 1000)):
            j = self.top_k
            tmp = set(pred[i][:j])
            while len(tmp) < self.top_k:
                tmp.update(pred[i][j:j+self.top_k-len(tmp)])
                j = j+self.top_k-len(tmp)
            if y[i].item() in tmp:
                score += 1

        return score / len(y_pred)
    