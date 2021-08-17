import copy
import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .custom_3d import Custom3DDataset
from .pipelines import Compose
import json
from  more_itertools import unique_everseen

@DATASETS.register_module()
class DeeprouteDataset(Custom3DDataset):
    r"""Deeproute Dataset.

    This class serves as the API for experiments on the `Deeproute Dataset

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        est_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [0, -40, -3, 70.4, 40, 0.0].
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 valid_mode=False,
                 pcd_limit_range=[-80, -80, -5, 80, 80, 3.0]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            valid_mode=valid_mode)
        self.valid_mode = valid_mode 
        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix
        self.CLASSES_EVAL = ('car','van','truck','big_truck','bus','pedestrian', 'cyclist','tricycle','cone')
        self.CLASSES_MAP = {'CAR': 0,
                   'VAN':1,
                   'TRUCK':2,
                   'BIG_TRUCK':3,
                   'BUS':4,
                   'PEDESTRIAN':5, 
                   'CYCLIST':6,
                   'TRICYCLE':7,
                   'CONE':8}


    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        if self.test_mode and not self.valid_mode:
            mode = 'testing'
            pcd_list = sorted(os.listdir(self.data_root + mode + '/pointcloud'))
        elif self.valid_mode:
            mode = 'training'
            pcd_list = sorted(os.listdir(self.data_root + mode + '/pointcloud' ))[18000:]
        else:
            mode = 'training'
            pcd_list = sorted(os.listdir(self.data_root + mode + '/pointcloud' ))[:100]
        return pcd_list



    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info[:-4]
        pts_filename = osp.join(self.root_split, 'pointcloud', sample_idx+'.bin')
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            )
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if not annos:
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index][:-4]
        mode = self.split
        annos_path='./data/deeproute_competition/'+ mode +'/groundtruth/' + info + '.txt'
        with open(annos_path, 'r') as f:
            objects = json.load(f)['objects']
        gt_bboxes_3d = []
        gt_names = []
        gt_labels = []
        for i,obj in enumerate(objects):        
            loc = [obj['position']['x'],obj['position']['y'],obj['position']['z']] 
            dims = [obj['bounding_box']['width'], obj['bounding_box']['length'], obj['bounding_box']['height']] 
            rots = [np.pi/2-obj['heading']]
            gt_bboxes_3d.append(loc+dims+rots)
            gt_names.append(obj['type'])
        if len(gt_bboxes_3d)==0:
            return None
        gt_bboxes_3d = np.array(gt_bboxes_3d)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                       origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        for cat in gt_names:
            if cat in self.CLASSES_MAP.keys():
                gt_labels.append(self.CLASSES_MAP[cat])
            else:
                gt_labels.append(-1)
        gt_labels_3d = np.array(gt_labels)
        anns_results = dict( 
                gt_bboxes_3d=gt_bboxes_3d, 
                gt_labels_3d=gt_labels_3d, 
                gt_names=gt_names,
                )
        return anns_results



    def format_results(self,
                       results,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        class_inverse_map = list(self.CLASSES_MAP)
        for i in range(len(results)):
            frame_result = {}
            object_list = []
            for obj_num in range(len(results[i]['scores_3d'])):
                obj={}
                obj["position"] = {}
                obj["position"]["x"] = results[i]['boxes_3d'].tensor[obj_num, 0].item()
                obj["position"]["y"] = results[i]['boxes_3d'].tensor[obj_num, 1].item()
                obj["position"]["z"] = results[i]['boxes_3d'].tensor[obj_num, 2].item() + (0.5 *results[i]['boxes_3d'].tensor[obj_num, 5].item())
                obj["bounding_box"] = {}
                obj["bounding_box"]["width"] = results[i]['boxes_3d'].tensor[obj_num, 3].item()
                obj["bounding_box"]["length"] = results[i]['boxes_3d'].tensor[obj_num, 4].item()
                obj["bounding_box"]["height"] = results[i]['boxes_3d'].tensor[obj_num, 5].item()
                obj["score"] = results[i]['scores_3d'][obj_num].item()
                obj["type"] =  class_inverse_map[results[i]['labels_3d'][obj_num].item()]
                obj["heading"] = np.pi/2 - results[i]['boxes_3d'].tensor[obj_num, 6].item()
                object_list.append(obj)
            frame_result['objects'] = object_list
            save_folder  =  self.root_split + '/submission/'
            save_file = save_folder + self.get_data_info(i)['sample_idx'] + '.txt'
            with open(save_file, 'w' , encoding="utf8") as f:
                json.dump(frame_result, f)

        #convert to summit format
        return results, save_folder

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=True,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        from mmdet3d.core.evaluation import deeproute_eval
        # load gt_data and tranform deeproute data to kitti format        
        #prepare gt for eval  
        gt_annos = [self.get_ann_info(i) for i in range(len(self.data_infos))] 
        gt_annos = self.deeproute2kitti_gt_format(gt_annos)
        gt_annos_bbox = self.gt_anno2kitti(gt_annos, self.CLASSES_EVAL)
        #NOTE gt_annos eval gt_annos test, need to add some noise to rotation to prevent riou bugs
        #gt_annos_bbox_noise = []
        #for i in range(len(gt_annos_bbox)):
        #    gt_annos_bbox_noise_ = copy.deepcopy(gt_annos_bbox[i])
        #    gt_annos_bbox_noise_['rotation_y'] = gt_annos_bbox_noise_['rotation_y']+0.01
        #    gt_annos_bbox_noise.append(gt_annos_bbox_noise_)
        #prepare dt for eval
        dt_annos = results            
        dt_annos_bbox = self.bbox2result_kitti(dt_annos, self.CLASSES_EVAL)
        ap_result_str, ap_dict = deeproute_eval(gt_annos_bbox, dt_annos_bbox, list(unique_everseen(self.CLASSES_EVAL)))    
        print_log('\n' + ap_result_str, logger=logger)
        return ap_result_str

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.data_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(0.0)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            # save file in pkl format
            pklfile_path = (
                pklfile_prefix[:-4] if pklfile_prefix.endswith(
                    ('.pkl', '.pickle')) else pklfile_prefix)
            mmcv.dump(det_annos, pklfile_path)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.data_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.tensor[:, -1] = box_preds.tensor[:, -1] - np.pi
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        img_shape = info['image']['image_shape']
        P2 = box_preds.tensor.new_tensor(P2)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)



    def deeproute2kitti_gt_format(self, gt_annos):
        #convert gt_anno to kitti format
        deeproute_gt_anno=[]
        for i in range(len(gt_annos)):
            info = {}
            try:
                info['pts_bbox'] = self.get_ann_info(i)
                info['pts_bbox']['boxes_3d'] = info['pts_bbox'].pop('gt_bboxes_3d')
                info['pts_bbox']['labels_3d'] = torch.tensor(info['pts_bbox'].pop('gt_labels_3d'))
                info['pts_bbox']['scores_3d'] = torch.tensor(np.ones(len(info['pts_bbox']['labels_3d'])))        
                deeproute_gt_anno.append(info)
            except:
                info['pts_bbox'] = {}
                info['pts_bbox']['boxes_3d'] = torch.tensor([])
                info['pts_bbox']['labels_3d'] = torch.tensor([])
                info['pts_bbox']['scores_3d'] = torch.tensor([])
                deeproute_gt_anno.append(info)
        return deeproute_gt_anno
    def gt_anno2kitti(self, deeproute_gt_annos, class_names):
        gt_annos = []
        print('\nConverting gt_annos to KITTI format')
        for idx, pred_dicts in enumerate( mmcv.track_iter_progress(deeproute_gt_annos)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info[:-4]
            box_dict = self.gt_convert_valid_bboxes(pred_dicts['pts_bbox'], info)
            if len(box_dict['box3d_lidar'])>0:
                if 'scores' in box_dict:
                    scores = box_dict['scores']
                else:
                    scores = np.ones(len(box_dict['label_preds']))
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                anno = {
                   'name': [],
                   'truncated': [],
                   'occluded': [],
                   'alpha': [],
                   'bbox': [],
                   'dimensions': [],
                   'location': [],
                   'rotation_y': [],
                   'score': [],
                }
                for box_lidar, score, label  in zip(box_preds_lidar, scores,label_preds):
                    #bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    #bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]))
                    anno['bbox'].append(np.zeros(4))
                    anno['dimensions'].append(box_lidar[3:6])
                    anno['location'].append(box_lidar[:3])
                    anno['rotation_y'].append(box_lidar[6])
                    anno['score'].append(score)
                    #anno['pts_in_box'].append(pts_in_box)
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                annos.append({
                'name': np.array([]), 
                'truncated': np.array([]),
                'occluded': np.array([]),
                'alpha': np.array([]),
                'bbox': np.zeros([0,4]),
                'dimensions': np.zeros([0,3]),
                'location': np.zeros([0,3]),
                'rotation_y': np.array([]),
                'score': np.array([]), 
                'pts_in_box':np.array([]) 
                })
            #annos[-1]['sample_idx'] = np.array([idx] * len(annos[-1]['score']), dtype=np.int64)
            gt_annos += annos
        return gt_annos

    def bbox2result_kitti(self, 
                          net_outputs, 
                          class_names, 
                          pklfile_prefix=None, 
                          submission_prefix=None):
        assert len(net_outputs) == len(self.data_infos)
        det_annos = []
       
        print('\nConverting prediction to KITTI format') 
        for idx, pred_dicts in enumerate(mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info[:-4]
            if 'pts_bbox' in pred_dicts.keys():
                pred_dicts = pred_dicts['pts_bbox']


            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            
            if len(box_dict['box3d_lidar'])>0:     
                if 'scores' in box_dict:      
                    scores = box_dict['scores']
                else:
                    scores = np.ones(len(box_dict['label_preds']))
                box_preds_lidar = box_dict['box3d_lidar']  
                label_preds = box_dict['label_preds'] 
                anno = {
                   'name': [], 
                   'truncated': [],
                   'occluded': [],
                   'alpha': [], 
                   'bbox': [],
                   'dimensions': [],
                   'location': [], 
                   'rotation_y': [],
                   'score': [] 
                
                }
                for box_lidar, score, label in zip(box_preds_lidar, scores,label_preds): 
                    anno['name'].append(class_names[int(label)]) 
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0) 
                    anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0])) 
                    anno['bbox'].append(np.zeros(4))
                    anno['dimensions'].append(box_lidar[3:6])
                    anno['location'].append(box_lidar[:3])
                    anno['rotation_y'].append(box_lidar[6]) 
                    anno['score'].append(score)
                    
                anno = {k: np.stack(v) for k, v in anno.items()} 
                annos.append(anno)
            else:
                annos.append({
                     'name': np.array([]), 
                     'truncated': np.array([]),
                     'occluded': np.array([]),
                     'alpha': np.array([]), 
                     'bbox': np.zeros([0, 4]),
                     'dimensions': np.zeros([0, 3]), 
                     'location': np.zeros([0, 3]), 
                     'rotation_y': np.array([]),
                     'score': np.array([]),
                })
            
            annos[-1]['sample_idx'] = np.array([idx] * len(annos[-1]['score']), dtype=np.int64)   
            det_annos += annos
        return det_annos


    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        #print("convert_valid_bbox")
        
   
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info[:-4]
        # TODO: remove the hack of yaw

        if len(box_preds) == 0:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # Post-processing
        if box_preds.tensor.shape[0]>0:
            return dict(
                box3d_lidar=box_preds.tensor.numpy(),
                scores=scores.numpy(),
                label_preds=labels.numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )
    def gt_convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        #print("convert_valid_bbox")


        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info[:-4]

        if len(box_preds) == 0:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        # Post-processing
       
        if box_preds.tensor.shape[0] > 0:
            return dict(
                box3d_lidar=box_preds.tensor.numpy(),
                scores=scores.numpy(),
                label_preds=labels.numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )
