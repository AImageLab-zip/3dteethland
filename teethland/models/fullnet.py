import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch_scatter import scatter_mean
from torchtyping import TensorType

import teethland
from teethland import PointTensor
import teethland.data.transforms as T
from teethland.cluster import learned_region_cluster
import teethland.nn as nn


def tta_nms(
    clusters: PointTensor,
    labels: PointTensor,
    iou_thresh: float=0.8,
):
    clusters1, clusters2 = clusters.batch(0), clusters.batch(1)

    num_instances = labels.batch_counts.sum()
    ious = torch.zeros(num_instances, num_instances).to(clusters.F.device)
    for i in range(labels.batch_counts[0]):
        mask1 = clusters1.F == i
        for j in range(labels.batch_counts[1]):
            mask2 = clusters2.F == (labels.batch_counts[0] + j)

            inter = (mask1 & mask2).sum()
            union = (mask1 | mask2).sum()
            iou = inter / (union + 1e-6)
            ious[i, labels.batch_counts[0] + j] = iou

    keep = torch.ones(labels.batch_counts.sum()).bool().to(clusters.F.device)
    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = iou >= iou_thresh
        keep = keep & ~condition

    return torch.nonzero(keep)[:, 0]


def instance_nms(
    probs: PointTensor,
    point_idxs: TensorType['K', 'N', torch.int64],
    conf_thresh: float=0.5,
    iou_thresh: float=0.35,
    score_thresh: float=0.7,
):
    fg_point_idxs, scores = [], torch.empty(0).to(probs.F)
    for i in range(probs.batch_size):
        fg_mask = probs.batch(i).F >= conf_thresh
        fg_idxs = point_idxs[i][fg_mask]
        fg_point_idxs.append(set(fg_idxs.cpu().tolist()))

        score = probs.batch(i).F[fg_mask].mean()
        scores = torch.cat((scores, score[None]))

    sort_index = torch.argsort(scores, descending=True)
    fg_point_idxs = [fg_point_idxs[idx.item()] for idx in sort_index]
    
    ious = torch.zeros(probs.batch_size, probs.batch_size).to(probs.F)
    for i in range(probs.batch_size):
        for j in range(i + 1, probs.batch_size):
            inter = len(fg_point_idxs[i] & fg_point_idxs[j])
            union = len(fg_point_idxs[i] | fg_point_idxs[j])
            iou = inter / (union + 1e-6)
            ious[i, j] = iou

    keep = scores[sort_index] >= score_thresh
    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = iou >= iou_thresh
        keep = keep & ~condition

    if not torch.all(keep):
        print('NMS applied!')
        k = 3

    return sort_index[keep].unique()


class FullNet(pl.LightningModule):

    def __init__(
        self,
        align: Dict[str, Any],
        instseg: Dict[str, Any],
        single_tooth: Dict[str, Any],
        proposal_points: int,
        dbscan_cfg: dict[str, Any],
        tta: bool,
        standardize: bool,
        stage2_iters: int,
        post_process: bool,
        out_dir: Path,
        **kwargs,
    ) -> None:
        super().__init__()

        # align stage
        ckpt = align.pop('checkpoint_path')
        self.align_backbone = nn.StratifiedTransformer(
            in_channels=kwargs['in_channels'],
            out_channels=None,
            **instseg,
        )
        self.load_ckpt(self.align_backbone, ckpt)

        self.align_head = nn.MLP(self.align_backbone.enc_channels, 256, 9)
        self.load_ckpt(self.align_head, ckpt)
        
        # instance segmentation stage
        ckpt = instseg.pop('checkpoint_path')
        self.instances_model = nn.StratifiedTransformer(
            in_channels=kwargs['in_channels'],
            out_channels=[6, 1, None],
            **instseg,
        )
        self.load_ckpt(self.instances_model, ckpt)

        self.identify_model = nn.MaskedAveragePooling(
            num_features=self.instances_model.out_channels[-1],
            out_channels=kwargs['num_classes'],
        )
        self.load_ckpt(self.identify_model, ckpt)
        
        # landmark prediction stage
        ckpt = single_tooth.pop('checkpoint_path')
        self.single_tooth_model = nn.StratifiedTransformer(
            in_channels=9,
            **single_tooth,
        )
        self.load_ckpt(self.single_tooth_model, ckpt)

        self.gen_proposals = T.GenerateProposals(proposal_points, max_proposals=40)
        self.tta = tta
        self.standardize = standardize
        self.stage2_iters = stage2_iters
        self.post_process = post_process
        self.dbscan_cfg = dbscan_cfg
        self.only_dentalnet = stage2_iters == 0
        self.out_dir = out_dir
        from teethland.data.datasets import TeethLandDataset
        self.landmark_class2label = {
            v: k for k, v in TeethLandDataset.landmark_classes.items()
        }

    def load_ckpt(self, model, ckpt: str):
        state_dict = model.state_dict()
        if ckpt:
            ckpt = torch.load(ckpt)['state_dict']
            ckpt = {k.split('.', 1)[1]: v for k, v in ckpt.items()}
            ckpt = {k: v for k, v in ckpt.items() if k in state_dict}
            model.load_state_dict(ckpt)
            model.requires_grad_(False)

    def align_stage(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, TensorType[4, 4, torch.float32]]:
        # downsample
        x_down = x[x.cache['instseg_downsample_idxs']]
        
        encoding = self.align_backbone(x_down)
        embeddings = scatter_mean(encoding.F, encoding.batch_indices, dim=0)
        embeddings = PointTensor(
            coordinates=torch.zeros(x.batch_size, 3).to(x.C),
            features=embeddings,
        )
        preds = self.align_head(embeddings)

        dir_up = preds.F[:, :3] / torch.linalg.norm(preds.F[:, :3], dim=-1, keepdim=True)
        dir_fwd = preds.F[:, 3:6] / torch.linalg.norm(preds.F[:, 3:6], dim=-1, keepdim=True)
        trans = preds.F[:, 6:]

        # make two vectors orthogonal
        dots = torch.einsum('bi,bi->b', dir_up, dir_fwd)
        dir_up -= dots[:, None] * dir_fwd

        # determine non-reflecting rotation matrix to standard basis
        pred_right = torch.cross(dir_fwd, dir_up, dim=-1)
        R = torch.stack((pred_right, dir_fwd, dir_up))[:, 0]
        if torch.linalg.det(R) < 0:
            print('Determinant < 0')
            R = torch.tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]).to(R) @ R

        # determine rotation matrix in 3DTeethSeg basis
        R = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]).to(R) @ R

        # apply translation and determine affine matrix
        T = torch.eye(4).to(R)
        T[:3, :3] = R
        T[:3, 3] = -trans @ R.T

        # apply transformation to input
        coords_homo = torch.column_stack((x.C, torch.ones_like(x.C[:, 0])))
        out = x.new_tensor(
            coordinates=(coords_homo @ T.T)[:, :3],
            features=torch.column_stack((
                (coords_homo @ T.T)[:, :3],
                x.F[:, 3:6] @ T[:3, :3].T,
            )),
        )
        out.cache = x.cache

        return out, T

    def instances_stage(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor, PointTensor, TensorType[4, 4, torch.float32]]:
        # downsample
        x_down = x[x.cache['instseg_downsample_idxs']]

        # test-time augmentation by horizontally flipping
        if self.tta:
            x_down_flip = x_down.clone()
            x_down_flip._coordinates[:, 0] *= -1
            x_down_flip.F[:, 0] *= -1
            x_down_flip.F[:, 3] *= -1

            x_down = teethland.stack((x_down, x_down_flip))
        
        # forward pass
        _, (spatial_embeds, seeds, features) = self.instances_model(x_down)

        # cluster
        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])
        clusters = learned_region_cluster(
            offsets, sigmas, seeds,
        )
        clusters._coordinates[clusters.batch_counts[0]:, 0] *= -1

        if not self.standardize:
            _, classes = self.identify_model(features, clusters)
            # classes = classes.new_tensor(features=classes.F.argmax(-1))
            labels = self.trainer.datamodule.teeth_classes_to_labels(classes)

            if self.tta:
                keep_idxs = tta_nms(clusters, labels)
                clusters.F[~torch.any(clusters.F == keep_idxs[:, None], dim=0)] = -1
                labels = labels[keep_idxs]

                if self.only_dentalnet:
                    clusters = clusters.batch(0).new_tensor(
                        features=torch.maximum(clusters.batch(0).F, clusters.batch(1).F),
                    )
                    labels._batch_counts = labels.batch_counts.sum()[None]

                clusters.F = torch.unique(clusters.F, return_inverse=True)[1] - 1

            # interpolate clusters back to original scan
            instances = clusters.interpolate(teethland.stack([x for _ in labels.batch_counts]))

            return x, instances, features, labels, torch.eye(4).to(x.F)


        # determine instance centroids
        centroids = scatter_mean(
            src=clusters.C[clusters.F >= 0],
            index=clusters.F[clusters.F >= 0],
            dim=0,
        )
        centroids = PointTensor(coordinates=centroids)
        
        # rotate intra-oral scan
        affine = self.trainer.datamodule.determine_rot_matrix(centroids)
        coords_rot = x.C @ affine[:3, :3].T
        normals_rot = x.F[:, 3:] @ affine[:3, :3].T
        x_rot = x.new_tensor(
            coordinates=coords_rot,
            features=torch.column_stack((coords_rot, normals_rot)),
        )
        x_rot.cache = x.cache
        x_rot_down = x_rot[x.cache['instseg_downsample_idxs']]

        # second pass
        _, (spatial_embeds, seeds, features) = self.instances_model(x_rot_down)

        # cluster
        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])
        clusters = learned_region_cluster(
            offsets, sigmas, seeds,
        )

        # determine FDI number on rotated scan
        _, classes = self.identify_model(features, clusters)
        # classes = classes.new_tensor(features=classes.F.argmax(-1))
        labels = self.trainer.datamodule.teeth_classes_to_labels(classes)

        # interpolate clusters back to original scan
        instances = clusters.interpolate(x_rot)

        return x_rot, instances, features, labels, affine
    
    def single_tooth_stage(
        self,
        x: PointTensor,
        instances: PointTensor,
        features: PointTensor,
        labels: PointTensor,
        affine: TensorType[4, 4, torch.float32],
    ) -> Tuple[PointTensor, PointTensor, Optional[PointTensor]]:
        x_down = x[x.cache['landmarks_downsample_idxs']]
        instances = teethland.stack(([
            instances.batch(i)[x.cache['landmarks_downsample_idxs']]
            for i in range(instances.batch_size)
        ]))
        
        # generate proposals based on predicted instances
        for _ in range(self.stage2_iters):
            data_dict = {
                'points': x_down.C.cpu().numpy(),
                'instances': instances.F.cpu().numpy(),
                'instance_centroids': labels.C.cpu().numpy(),
                'normals': x_down.F[:, 3:].cpu().numpy(),
            }
            data_dict = self.gen_proposals(**data_dict)
            points = torch.from_numpy(data_dict['points']).to(x.C)
            normals = torch.from_numpy(data_dict['normals']).to(x.C)
            centroids = torch.from_numpy(data_dict['instance_centroids']).to(x.C)
            proposals = PointTensor(
                coordinates=points.reshape(-1, 3),
                features=torch.column_stack((
                    points.reshape(-1, 3),
                    normals.reshape(-1, 3),
                    (points - centroids[:, None]).reshape(-1, 3),
                )),
                batch_counts=torch.tensor([points.shape[1]]*points.shape[0]).to(x.C.device),
            )

            # run the proposals through the models
            _, preds = self.single_tooth_model(proposals)
            seg = preds[0] if isinstance(preds, list) else preds
            probs = seg.new_tensor(features=torch.sigmoid(seg.F[:, 0]))

            # apply non-maximum suppression to remove redundant instances
            point_idxs = torch.from_numpy(data_dict['point_idxs']).to(seg.F.device)
            keep_idxs = instance_nms(probs, point_idxs)
            instances.F[~torch.any(instances.F == keep_idxs[:, None], dim=0)] = -1
            instances.F = torch.unique(instances.F, return_inverse=True)[1] - 1
            labels = labels[keep_idxs]
            probs = probs.batch(keep_idxs)

            # update the centroids for the second run
            labels._coordinates = scatter_mean(
                src=probs.C[probs.F >= 0.5],
                index=probs.batch_indices[probs.F >= 0.5],
                dim=0,
            )

        # interpolate segmentations to original points
        instances = torch.full_like(x.F[:, 0], -1).long()
        max_probs = torch.zeros_like(instances).float()
        for b in range(probs.batch_size):
            interp = probs.batch(b).interpolate(x, dist_thresh=0.03).F
            instances = torch.where(interp > max_probs, b, instances)
            max_probs = torch.maximum(max_probs, interp)
        instances = x.new_tensor(features=instances)
        if self.post_process:
            instances = self.trainer.datamodule.process_instances(instances, max_probs)
        else:
            instances.F = torch.where(max_probs >= 0.5, instances.F, -1)
        
        _, inverse, counts = torch.unique(instances.F, return_inverse=True, return_counts=True)
        instances.F[(counts < 16)[inverse]] = -1
        instances.F = torch.unique(instances.F, return_inverse=True)[1] - 1
        
        clusters = instances[x.cache['instseg_downsample_idxs']]
        _, classes = self.identify_model(features.batch(0), clusters)
        # labels = classes.new_tensor(features=classes.F.argmax(-1))
        labels.F = self.trainer.datamodule.teeth_classes_to_labels(classes).F

        if not isinstance(preds, list):
            return instances, labels, None
        
        print('start landmarks')
        # apply NMS selection
        proposals = proposals.batch(keep_idxs)
        points_offsets = preds[1:]
        points_offsets = [offsets.batch(keep_idxs) for offsets in points_offsets]

        # process point-level landmarks
        landmarks_list = []
        for i, offsets in enumerate(points_offsets):
            kpt_mask = offsets.F[:, 0] < 0.15  # 2.5 mm
            coords = proposals.C + offsets.F[:, 1:]
            dists = torch.clip(offsets.F[:, 0], 0, 0.15)
            weights = (0.15 - dists) / 0.15
            landmarks = PointTensor(
                coordinates=coords[kpt_mask],
                features=weights[kpt_mask],
                batch_counts=torch.bincount(
                    input=proposals.batch_indices[kpt_mask],
                    minlength=proposals.batch_size,
                ),
            )
            landmarks = landmarks.cluster(**self.dbscan_cfg)
            landmarks = landmarks.new_tensor(features=torch.column_stack((landmarks.F, 
                torch.full((landmarks.C.shape[0],), i).to(coords.device),
            )))
            landmarks_list.append(landmarks)
        landmarks = teethland.cat(landmarks_list)

        landmarks = self.trainer.datamodule.process_landmarks(labels, affine, landmarks)

        return instances, labels, landmarks

    def forward(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor]:
        # stage 1
        x, affine = self.align_stage(x)

        # stage 2
        x, instances, features, labels, affine = self.instances_stage(x)
        if torch.all(instances.F == -1) or self.only_dentalnet:
            return instances, labels, None
        
        # stage 3
        instances, labels, landmarks = self.single_tooth_stage(x, instances, features, labels, affine)

        return instances.batch(0), labels.batch(0), landmarks
    
    def predict_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Path,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ):
        instances, classes, landmarks = self(batch)

        self.save_segmentation(instances, classes)
        self.save_landmarks(landmarks)

    def save_segmentation(self, instances: PointTensor, labels: PointTensor):
        if torch.any(instances.F >= 0):
            labels = torch.where(instances.F >= 0, labels.F[instances.F], 0)
        else:
            labels = torch.zeros_like(instances.F)
        instances = torch.unique(instances.F, return_inverse=True)[1] - 1

        out_dict = {
            'instances': instances.cpu().tolist(),
            'labels': labels.cpu().tolist(),
        }
        
        out_name = Path(self.trainer.datamodule.scan_file).with_suffix('.json')
        if self.out_dir.name:
            out_file = self.out_dir / out_name.name
        else:
            out_file = self.trainer.datamodule.root / out_name
        with open(out_file, 'w') as f:
            json.dump(out_dict, f)

    def save_landmarks(self, landmarks: Optional[PointTensor]):
        if landmarks is None:
            return
        
        template = {
            'version': '1.1',
            'description': 'landmarks',
            'key': self.trainer.datamodule.scan_file,
            'objects': [],
        }
        for i, (coords, (score, cls), instance) in enumerate(zip(
            landmarks.C, landmarks.F, landmarks.batch_indices,
        )):
            landmark = {
                'key': f'uuid_{i}',
                'score': score.cpu().item(),
                'class': self.landmark_class2label[cls.cpu().item()],
                'coord': coords.cpu().tolist(),
                'instance_id': instance.cpu().item()
            }
            template['objects'].append(landmark)

        out_name = Path(template['key'][:-4] + '__kpt.json')
        if self.out_dir.name:
            out_file = self.out_dir / out_name.name
        else:
            out_file = self.trainer.datamodule.root / out_name
        with open(out_file, 'w') as f:
            json.dump(template, f, indent=2)
