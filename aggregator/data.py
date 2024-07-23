from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

import numpy as np
from scipy.stats import describe
import pickle
from collections import defaultdict

class infoDataset(Dataset):

    def __init__(self, predfile=None, infos=None, score_range=2, type='normal'):

        assert predfile != None or infos != None

        if predfile:
            self.infos = self.get_info(predfile)
        else:
            self.infos = infos
        self.score_range = score_range
        self.type = type

    def __getitem__(self, index):
        return self.get_input(index), self.get_target(index)

    def __len__(self):
        return len(self.infos)

    def __iter__(self):
        return (self.__getitem__(i) for i in self.get_indices())

    def get_indices(self):
        # return list(self.videos.groups.keys())
        return [i for i in range(len(self.infos['videos']))]

    def get_input(self, index):
        data = self.infos['videos'][index]['preds']
        scores = []
        for d in data:
            pred_score = d['pred_score']
            scores.append(pred_score)

        return torch.tensor(np.array(scores).transpose(), dtype=torch.float32)

    def get_target(self, index):
        data = self.infos['videos'][index]['label']
        return torch.tensor([data])

    def get_videoname(self, index):
        data = self.infos['videos'][index]['video_name']
        return [data]

    def get_info(self, f):
        '''
        ann:
        {'videos':
            [
                {'video_name': 'malignant_',
                'label': 1,
                'preds': [{'label': 1,
                'img': 'malignant_2c12ff6464cfb1ff_000000.png',
                'pred_label': 1, 'pred_score': 0.42005401849746704}, {...}, {...}]
                }, ..., ...
            ]
        }
        '''
        with open(f, 'rb') as file:
            data = pickle.load(file)

        ann = defaultdict(list)
        for record in data:
            vid = '_'.join(record['img_path'].split('/')[-1].split('_')[0:2])
            if 'malignant' in vid:
                label = 1
            elif 'benign' in vid:
                label = 0

            existing_videos = [item['video_name'] for item in ann['videos']]
            if vid not in existing_videos:
                ann['videos'].append(
                    dict(
                        video_name=vid,
                        label=label
                    ))

        for info in ann['videos']:
            info['preds'] = []
            for d in data:

                if info['video_name'] in d['img_path']:

                    labels = d['pred_instances']['labels'].tolist()
                    scores = d['pred_instances']['scores'].tolist()
                    score_sum = {}
                    for l, s in zip(labels, scores):
                        if l not in score_sum:
                            score_sum[l] = s
                        else:
                            score_sum[l] = max(score_sum[l], s)

                    if -1 in score_sum:
                        score_sum.pop(-1)
                    pred_score = np.array([score_sum[k] for k in sorted(score_sum.keys())])
                    pred_label = np.argmax(pred_score)
                    assert pred_label == int(d['pred_instances']['labels'][0])

                    info['preds'].append(dict(
                        label=label,
                        img=d['img_path'].split('/')[-1],
                        pred_label=pred_label,
                        pred_score=pred_score
                    ))
        return ann
