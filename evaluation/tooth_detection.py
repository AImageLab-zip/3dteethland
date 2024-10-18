import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d
import pymeshlab
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, jaccard_score
from tqdm import tqdm


from teethland.visualization import palette


def process_scan(pred_label_dict, gt_label_dict, iou_thresh: float=0.5):
    pred_instances = np.array(pred_label_dict['instances'])
    pred_point_idxs = []
    for label in np.unique(pred_instances)[1:]:
        point_idxs = np.nonzero(pred_instances == label)[0]
        pred_point_idxs.append(set(point_idxs.tolist()))

    gt_instances = np.array(gt_label_dict['instances'])
    gt_point_idxs = []
    for label in np.unique(gt_instances)[1:]:
        point_idxs = np.nonzero(gt_instances == label)[0]
        gt_point_idxs.append(set(point_idxs.tolist()))

    tooth_dices = []
    gt_labels, pred_labels = [], []
    ious = np.zeros((max(gt_label_dict['instances']), max(pred_label_dict['instances'])))
    for i in range(max(gt_label_dict['instances'])):
        for j in range(max(pred_label_dict['instances'])):
            inter = len(gt_point_idxs[i] & pred_point_idxs[j])
            union = len(gt_point_idxs[i] | pred_point_idxs[j])
            iou = inter / union
            ious[i, j] = iou

            if iou < iou_thresh:
                continue
            
            dice = 2 * inter / (inter + union)
            tooth_dices.append(dice)
            gt_labels.append(gt_label_dict['labels'][next(iter(gt_point_idxs[i]))])
            pred_labels.append(pred_label_dict['labels'][next(iter(pred_point_idxs[j]))])

    fps = (np.max(ious, axis=0) < iou_thresh).sum()
    fns = (np.max(ious, axis=1) < iou_thresh).sum()
    tps = (max(gt_label_dict['instances']) + max(pred_label_dict['instances']) - fps - fns) / 2

    gt_points = gt_label_dict['labels']
    pred_points = pred_label_dict['labels']

    return tps, fps, fns, tooth_dices, gt_labels, pred_labels, gt_points, pred_points



if __name__ == "__main__":
    gt_dir = Path('/mnt/diag/IOS/3dteethseg/full_dataset/lower_upper')
    gt_dir = Path('/home/mkaailab/Documents/IOS/Brazil/cases')
    pred_dir = Path('mixed_ios')
    TLA, TSA, TIR = [], [], []
    verbose = False

    stats = {
        'names': [], 'fps': [], 'fns': [], 'tps': [], 'dices': [],
        'gt_labels': [], 'pred_labels': [],
        'gt_points': [], 'pred_points': [],
    }

    gt_files = sorted(gt_dir.glob('**/*.json'))

    pred_files = sorted(pred_dir.glob('*'))[:]
    fail_files = []
    i = 0
    for pred_filename in tqdm(pred_files):        
        with open(pred_filename, 'r') as f:
            pred_label_dict = json.load(f)
        pred_label_dict['instances'] = (np.array(pred_label_dict['instances']) + 1).tolist()

        # load ground-truth segmentations
        gt_filename = [f for f in gt_files if f.name == pred_filename.name][0]
        with open(gt_filename, 'r') as f:
            gt_label_dict = json.load(f)

        labels = np.array(gt_label_dict['labels'])
        instances = np.array(gt_label_dict['instances'])
        
        _, instances, counts = np.unique(instances, return_inverse=True, return_counts=True)
        labels[(counts < 100)[instances]] = 0
        instances[(counts < 100)[instances]] = 0
        _, instances = np.unique(instances, return_inverse=True)
        
        gt_label_dict['labels'] = labels.tolist()
        gt_label_dict['instances'] = instances.tolist()

        tps, fps, fns, tooth_dices, gt_labels, pred_labels, gt_points, pred_points = process_scan(pred_label_dict, gt_label_dict)
        # if False:
        if fns:
            fail_files.append(i)
            _, counts = np.unique(pred_label_dict['instances'], return_counts=True)
            print(pred_filename, fps, fns, tps, counts)

            
        if verbose and (fps or fns):
            mesh_file = gt_filename.with_suffix('.obj')

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_file))
            mesh = ms.current_mesh()
            mesh.compact()

            vertices = mesh.vertex_matrix()
            triangles = mesh.face_matrix()

            mesh = open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(vertices),
                triangles=open3d.utility.Vector3iVector(triangles),
            )
            mesh.compute_vertex_normals()
            _, instances = np.unique(pred_label_dict['labels'], return_inverse=True)
            mesh.vertex_colors = open3d.utility.Vector3dVector(palette[instances] / 255)

            open3d.visualization.draw_geometries([mesh], width=1600, height=900)
            
        i += 1
        stats['tps'].append(tps)
        stats['fps'].append(fps)
        stats['fns'].append(fns)
        stats['dices'].extend(tooth_dices)
        stats['gt_labels'].extend(gt_labels)
        stats['pred_labels'].extend(pred_labels)
        stats['gt_points'].extend(gt_points)
        stats['pred_points'].extend(pred_points)
        stats['names'].append(pred_filename.name)

    print('Tooth Precision:', sum(stats['tps']) / (sum(stats['tps']) + sum(stats['fps'])))
    print('Tooth Sensitivity:', sum(stats['tps']) / (sum(stats['tps']) + sum(stats['fns'])))
    print('Tooth F1:', 2 * sum(stats['tps']) / (2 * sum(stats['tps']) + sum(stats['fps']) + sum(stats['fns'])))
    print('Tooth Dice:', np.mean(stats['dices']))

    gt_labels = np.array(stats['gt_labels'])
    pred_labels = np.array(stats['pred_labels'])
    print('Tooth macro-F1:', f1_score(gt_labels, pred_labels, average='macro'))

    macro_iou = jaccard_score(stats['gt_points'], stats['pred_points'], labels=np.unique(stats['gt_points'])[1:], average='macro')
    print('macro-IoU:', macro_iou)

    with open('failures.txt', 'w') as f:
        for idx in fail_files:
            f.write(pred_files[idx].stem + '\n')

    upper_mask = np.isin(gt_labels // 10, np.array([1, 2, 5, 6]))

    cmd = ConfusionMatrixDisplay.from_predictions(gt_labels[upper_mask], pred_labels[upper_mask])
    cmd.plot()
    plt.show(block=True)
    
    cmd = ConfusionMatrixDisplay.from_predictions(gt_labels[~upper_mask], pred_labels[~upper_mask])
    cmd.plot()
    plt.show(block=True)

    k = 3
