from collections import defaultdict
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymeshlab
from tqdm import tqdm

if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/IOS/partials/full_dataset/root_partial')

    ms = pymeshlab.MeshSet()
    pair_dists = defaultdict(list)
    num_teeth = defaultdict(int)
    for label_file in tqdm(sorted(root.glob('**/*.json'))):
        with open(label_file, 'r') as f:
            teeth3ds_dict = json.load(f)

        ms.load_new_mesh(str(label_file.parent / label_file.with_suffix('.ply').name))
        vertices = ms.current_mesh().vertex_matrix()

        instances = np.array(teeth3ds_dict['instances'])
        labels = np.array(teeth3ds_dict['labels'])
        fdis, centroids = [], []

        unique = np.unique(instances)[1:]
        for idx in unique:
            mask = instances == idx

            fdi = labels[mask][0]
            centroid = vertices[mask].mean(0)

            fdis.append(fdi)
            centroids.append(centroid)

        fdis = np.array(fdis)
        num_teeth[unique.shape[0]] += 1
        centroids = np.stack(centroids)

        dists = np.linalg.norm(centroids[None] - centroids[:, None], axis=-1)
        for i, fdi1 in enumerate(fdis):
            for j, fdi2 in enumerate(fdis):
                pair_dists[f'{fdi1}_{fdi2}'].append(dists[i, j])
    
    out = {k: np.zeros((49, 49)) for k in ['means', 'stds']}
    for fdis, dists in pair_dists.items():
        fdi1, fdi2 = map(int, fdis.split('_'))
        mean = np.mean(dists)
        gamma_a = 5 + len(dists) / 2
        gamma_b = 1 / (5 + np.sum((mean - dists) ** 2) / 2)
        gamma_mode = (gamma_a - 1) * gamma_b
        std = np.sqrt(1 / gamma_mode)
        
        out['means'][fdi1][fdi2] = mean
        out['stds'][fdi1][fdi2] = std

    with open('pair_dists.pkl', 'wb') as f:
        pickle.dump(out, f)
