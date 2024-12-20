import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import open3d
# from open3d.visualization.tensorboard_plugin import summary  # do not remove
# from open3d.visualization.tensorboard_plugin.util import to_dict_batch
import pymeshlab
import torch

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.getcwd())

# from teethland import PointTensor
from torch import Tensor as PointTensor


palette = torch.tensor([
    [174, 199, 232],
    [152, 223, 138],
    [31, 119, 180],
    [255, 187, 120],
    [188, 189, 34],
    [140, 86, 75],
    [255, 152, 150],
    [214, 39, 40],
    [197, 176, 213],
    [148, 103, 189],
    [196, 156, 148], 
    [23, 190, 207], 
    [247, 182, 210], 
    [219, 219, 141], 
    [255, 127, 14], 
    [158, 218, 229], 
    [44, 160, 44], 
    [112, 128, 144], 
    [227, 119, 194], 
    [82, 84, 163],
    [100, 100, 100],
], dtype=torch.uint8)


def draw_mesh(
    file: Path,
    pt: PointTensor,
) -> None:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file))
    mesh = ms.current_mesh()
    mesh.compact()

    vertices = mesh.vertex_matrix()
    triangles = mesh.face_matrix()

    mesh = open3d.geometry.TriangleMesh(
        vertices=open3d.utility.Vector3dVector(vertices),
        triangles=open3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()

    if pt.has_features:
        pt = pt.new_tensor(features=pt.F - pt.F.amin() - 1)
        mesh.vertex_colors = open3d.utility.Vector3dVector(palette[pt.F] / 255)

    open3d.visualization.draw_geometries([mesh], width=1600, height=900)


def draw_point_clouds(
    pt: PointTensor,
    output_type: Optional[str]=None,
) -> Optional[List[open3d.geometry.PointCloud]]:
    pt = pt.to('cpu')
    length = math.sqrt(pt.batch_size)
    nrows = math.ceil(length)
    ncols = round(length)

    x_range = pt.C[:, 0].amax() - pt.C[:, 0].amin()
    y_range = pt.C[:, 1].amax() - pt.C[:, 1].amin()
    geometries = []
    for row in range(nrows):
        for col in range(ncols):
            batch_idx = ncols * row + col
            if batch_idx >= pt.batch_size:
                continue

            points = pt.batch(batch_idx).C
            points[:, 0] += 1.2 * row * x_range
            points[:, 1] += 1.2 * col * y_range
            pcd = open3d.geometry.PointCloud(
                points=open3d.utility.Vector3dVector(points.cpu()),
            )

            if not pt.has_features or pt.F.dim() != 1:
                geometries.append(pcd)
                continue

            feats = pt.batch(batch_idx).F
            if feats.dtype in [torch.float32, torch.float64]:
                colors = feats.float()
                colors = colors - colors.amin()
                colors /= colors.amax()
                colors = colors.expand(3, -1).T.cpu()
            elif feats.dtype in [torch.bool, torch.int32, torch.int64]:
                feats = feats.long()
                classes = feats - feats.amin() - 1
                classes[classes >= 0] %= palette.shape[0] - 1
                colors = palette[classes] / 255

            pcd.colors = open3d.utility.Vector3dVector(colors.cpu().detach())
            geometries.append(pcd)

    if output_type == 'tensorboard':
        return to_dict_batch(geometries[:1])

    open3d.visualization.draw_geometries(geometries, width=1600, height=900)


def draw_landmarks(
    vertices: np.ndarray,
    landmarks: np.ndarray,
    labels:  Optional[np.ndarray]=None,
    normals: Optional[np.ndarray]=None,
    triangles: Optional[np.ndarray]=None,
    point_size: float=0.025
):
    if triangles is not None:
        geom = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        geom.compute_vertex_normals()
    else:
        geom = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(vertices),
        )
        if normals is not None:
            geom.normals = open3d.utility.Vector3dVector(normals)
        geom.colors = open3d.utility.Vector3dVector(np.full((vertices.shape[0], 3), 100 / 255))

    balls = []
    for landmark in landmarks:
        ball = open3d.geometry.TriangleMesh.create_sphere(radius=point_size)
        ball.translate(landmark[:3])
        if landmark.shape[0] == 3:
            ball.paint_uniform_color([0.0, 0.0, 1.0])
        else:
            ball.paint_uniform_color(palette[int(landmark[-1])].numpy() / 255)
        ball.compute_vertex_normals()
        balls.append(ball)

    open3d.visualization.draw_geometries([geom, *balls], width=1600, height=900)


def check_predictions():
    root = Path('/mnt/diag/IOS/Brazil/Modelberging')
    root = Path('/home/mkaailab/Documents/CBCT/fusion/stls')
    root = Path('/home/mkaailab/Documents/IOS/Katja VOs/transfer_2892173_files_6b1a93dd')
    # root = Path('/home/mkaailab/Documents/IOS/Brazil/cases')
    root = Path('/mnt/diag/IOS/3dteethseg/full_dataset/lower_upper')
    root = Path('/home/mkaailab/Documents/IOS/partials/full_dataset/complete_partial')
    # root = Path('/home/mkaailab/Documents/CBCT/fusion/complete_cadavers/IOS scans')
    # root = Path('/home/mkaailab/Documents/IOS/Katja Vos/AKMA')
    # root = Path('/mnt/diag/IOS/3dteethseg/full_dataset/test')
    mesh_files = sorted(list(root.glob('**/*.obj')) + list(root.glob('**/*.ply')) + list(root.glob('**/*.stl')))
    # root = root.parent / 'last_case'
    # root = Path('/mnt/diag/IOS/3dteethseg/full_dataset/lower_upper')
    root = Path('/home/mkaailab/Documents/IOS/partials/full_dataset/result_realpartials')
    ann_files = sorted(root.glob('**/*er.json'))
    # ann_files = sorted(root.glob('**/*er.json'))
    clean = False

    start_idx = 0
    # files = mesh_files[start_idx:]
    for i, ann_file in enumerate(ann_files[30:]):
        # mesh_file = sorted(root.glob(f'{ann_file.name.split("_")[0]}/{ann_file.stem}*'))[-1]
        # mesh_file = root.parent / 'cases' / ann_file.stem.split('_')[0] / f'{ann_file.stem}.ply'
        # ann_file = Path(f'dentalnetPr/{mesh_file.stem}.json')
        # ann_file = [f for f in ann_files if f.stem == mesh_file.stem][0]
        mesh_file = [f for f in mesh_files if f.stem == ann_file.stem][0]
        # ann_file = next(root.parent.glob(f'**/{mesh_file.stem}.json'))
        if not ann_file.exists():
            continue
        # ann_file = mesh_file.with_suffix('.json')
        # print(i, ':', mesh_file.stem, ann_file)

        # if not mesh_file.stem == '20221229_lower':
        #     ann_file = Path('output/TVSR5QBQ_lower.json')
        #     continue

        # draw_instances(mesh_file, ann_file)
        # draw_proposal(mesh_file, ann_file)
        # draw_point_cloud(mesh_file)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_file))
        # ms.meshing_remove_duplicate_vertices()
        if clean:
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes(maxholesize=130)  # ~20mm boundary
        mesh = ms.current_mesh()
        mesh.compact()

        vertices = mesh.vertex_matrix()
        triangles = mesh.face_matrix()

        with open(ann_file, 'rb') as f:
            ann = json.load(f)

        labels = np.array(ann['labels'])
        classes = np.full_like(labels, fill_value=-1)
        instances = np.array(ann['instances'])

        labels[(71 <= labels) & (labels <= 75)] -= 63
        labels[(31 <= labels) & (labels <= 37)] -= 30
        labels[labels == 38] = 7
        labels[(81 <= labels) & (labels <= 85)] -= 73
        labels[(41 <= labels) & (labels <= 47)] -= 40
        labels[labels == 48] = 7

        _, inverse = np.unique(labels, return_inverse=True)

        print(mesh_file)
        centroids = []
        for idx in np.unique(instances)[1:]:
            centroid = vertices[instances == idx].mean(0)
            centroids.append(centroid)
        if centroids:
            centroids = np.stack(centroids)
            inst_labels = np.array(ann['labels'])[np.unique(instances, return_index=True)[1][1:]]
            print(inst_labels[np.argsort(centroids[:, 0])])

        # instances = np.array(ann['instances'])*0 - 1

        # if not np.any(np.unique(np.array(ann['labels'])) % 10 == 8):
        #     continue

        # labels = np.clip(labels % 10, a_min=0, a_max=7)
        # classes = (instances == 2) - 1

        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        mesh.vertex_colors = open3d.utility.Vector3dVector(palette[instances] / 255)

        open3d.visualization.draw_geometries([mesh], width=1600, height=900)


def check_landmarks():
    seg_root = Path('/home/mkaailab/Documents/IOS/3dteethland/data/unseen')
    seg_root = Path('/mnt/diag/IOS/3dteethseg/full_dataset/lower_upper')
    # seg_root = Path('input')
    # seg_root = Path('/home/mkaailab/Documents/IOS/Brazil/test')
    # seg_root = Path('/home/mkaailab/Documents/CBCT/fusion/stls')
    landmarks_root = Path('/home/mkaailab/Documents/IOS/3dteethland/data/3DTeethLand_landmarks_train')
    landmarks_root = Path('/home/mkaailab/Documents/IOS/3dteethland/code/preds/3dteethland')
    # landmarks_root = seg_root

    for landmarks_file in sorted(landmarks_root.glob('**/*__kpt.json')):
        stem = landmarks_file.stem.split('__')[0]
        ann_file = next(seg_root.glob(f'**/{stem}.json'))
        mesh_file = next(seg_root.glob(f'**/{stem}.obj'))

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_file))

        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()

        with open(ann_file, 'rb') as f:
            ann = json.load(f)

        labels = np.array(ann['labels'])
        _, inverse = np.unique(labels, return_inverse=True)
        print(stem)
        print(_)

        labels = np.clip(labels % 10, a_min=0, a_max=7)

        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        mesh.vertex_colors = open3d.utility.Vector3dVector(palette[inverse - 1] / 255)
        mesh.vertex_colors = open3d.utility.Vector3dVector(np.full((inverse.shape[0], 3), 100 / 255))


        with open(landmarks_file, 'r') as f:
            landmarks = json.load(f)['objects']
        landmark_coords = np.array([landmark['coord'] for landmark in landmarks])
        landmark_scores = np.array([landmark['score'] for landmark in landmarks])
        landmark_classes = np.array([landmark['class'] for landmark in landmarks])
        _, landmark_classes = np.unique(landmark_classes, return_inverse=True)

        mask = landmark_scores >= 0.3
        landmarks = np.column_stack((landmark_coords[mask], landmark_classes[mask]))
        
        balls = []
        for landmark in landmarks:
            ball = open3d.geometry.TriangleMesh.create_sphere(radius=0.03 * 17.3281)
            ball.translate(landmark[:3])
            ball.paint_uniform_color(palette[int(landmark[-1])].numpy() / 255)
            ball.compute_vertex_normals()
            balls.append(ball)

        open3d.visualization.draw_geometries([*balls, mesh])





if __name__ == '__main__':
    import json
    from pathlib import Path

    import numpy as np
    import open3d
    import pymeshlab

    check_predictions()
    # check_landmarks()

    