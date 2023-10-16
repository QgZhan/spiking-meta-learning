import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_shot, n_query, ep_per_batch=1, ep_batch_shot = 1):
        self.n_batch = n_batch  # 200
        self.n_cls = n_cls  # 5
        self.n_per = n_shot * ep_batch_shot + n_query  # 5 * 1 + 15 = 20
        self.ep_per_batch = ep_per_batch  # 1

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)  # (20)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)  # 20*5 = 100
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

