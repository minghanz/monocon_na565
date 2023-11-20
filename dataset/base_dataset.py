import os
import sys
import cv2
import json
import numpy as np

from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.kitti_eval import kitti_eval
from utils.data_classes import KITTICalibration, KITTIMultiObjects

# Fixed Root
IMAGESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ImageSets_final_prj')


class BaseKITTIMono3DDataset(Dataset):
    def __init__(self, 
                 base_root: str, 
                 split: str,
                 pad_divisor: int = 32,
                 preload_gt_info: bool = False):
        
        super().__init__()
        
        assert os.path.isdir(base_root)
        self.base_root = base_root
        
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        
        with open(os.path.join(IMAGESET_DIR, f'{split}.txt')) as f:
            file_prefix = f.readlines()
        self.file_prefix = [fp.replace('\n', '') for fp in file_prefix]
        
        sub_root = 'testing' if (split == 'test') else 'training'
        
        # Image Files
        self.image_dir = os.path.join(base_root, sub_root, 'image_2')
        self.image_files = [os.path.join(self.image_dir, f'{fp}.png') for fp in self.file_prefix]
        
        # Image Meta Files
        self.img_meta_dir = os.path.join(base_root, sub_root, 'img_meta')
        self.img_meta_files = [os.path.join(self.img_meta_dir, f'{fp}.txt') for fp in self.file_prefix]

        # Calibration Files
        self.calib_dir = os.path.join(base_root, sub_root, 'calib')
        self.calib_files = [os.path.join(self.calib_dir, f'{fp}.txt') for fp in self.file_prefix]
        
        # Label Files
        self.label_dir, self.label_files = None, []
        if (split != 'test'):
            self.label_dir = os.path.join(base_root, sub_root, 'label_2')
            self.label_files = [os.path.join(self.label_dir, f'{fp}.txt') for fp in self.file_prefix]
        
        self.pad_divisor = pad_divisor
        
        self.gt_annos = None
        if preload_gt_info:
            gt_infos = self.collect_gt_infos()
            self.gt_annos = [gt_info['annos'] for gt_info in gt_infos]
            
    def __len__(self):
        return len(self.file_prefix)
    
    def __getitem__(self, idx: int):
        raise NotImplementedError
    
    def load_image(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        image_arr = cv2.imread(self.image_files[idx])
        image_data = cv2.cvtColor(image_arr, code=cv2.COLOR_BGR2RGB)
        
        img_metas = {
            'idx': idx,
            'split': self.split,
            'sample_idx': int(os.path.basename(self.image_files[idx]).split('.')[0]),
            'image_path': self.image_files[idx],
            'ori_shape': image_data.shape[:2]}
        return (image_data, img_metas)
    
    def load_img_meta(self, idx: int) -> Dict[str, Any]:
        img_size_file = self.img_meta_files[idx]
        with open(img_size_file, 'r') as f:
            img_size = f.read().strip().split(' ')
        img_size = [int(size) for size in img_size]

        img_metas = {
            'idx': idx,
            'split': self.split,
            'sample_idx': int(os.path.basename(img_size_file).split('.')[0]),
            'image_path': self.image_files[idx],
            'ori_shape': img_size}
        return img_metas

    def load_calib(self, idx: int) -> KITTICalibration:
        return KITTICalibration(self.calib_files[idx])
    
    def load_label(self, idx: int) -> KITTIMultiObjects:
        calib = self.load_calib(idx)
        return KITTIMultiObjects.get_objects_from_label(self.label_files[idx], calib)
    
    def collect_gt_infos(self, verbose: bool = False) -> List[Dict[str, Any]]:
        
        # Entire objects which include 'DontCare' class are required for evaluation.
        # If 'ignored_flag' is True, Filtered objects are converted to the original objects.
        ignored_flag = False
        if self.load_label(0).ignore_dontcare:
            ignored_flag = True
            
        results = []
        num_samples = len(self)
        
        iter_ = range(num_samples)
        if verbose:
            iter_ = tqdm(iter_, desc="Collecting GT Infos...")
        
        for idx in iter_:
            
            if os.path.exists(self.img_meta_files[idx]):
                img_metas = self.load_img_meta(idx)
            else:
                _, img_metas = self.load_image(idx)
            
            calib = self.load_calib(idx)
            calib_dict = calib.get_info_dict()
            
            obj_cls = self.load_label(idx)
            if ignored_flag:
                obj_cls = obj_cls.original_objects
            obj_dict = obj_cls.info_dict
            
            results.append(
                {'image': img_metas,
                 'calib': calib_dict,
                 'annos': obj_dict})
        return results  
    
    def evaluate(self, 
                 kitti_format_results: Dict[str, Any],
                 eval_classes: List[str] = ['Pedestrian', 'Cyclist', 'Car'],
                 eval_types: List[str] = ['bbox', 'bev', '3d'],
                 verbose: bool = True,
                 save_path: str = None) -> Dict[str, float]:
        
        if self.gt_annos is None:
            gt_infos = self.collect_gt_infos(verbose=verbose)
            gt_annos = [info['annos'] for info in gt_infos]
            sample_idxs = [info['image']['sample_idx'] for info in gt_infos]
            
            self.gt_annos = gt_annos

        ap_dict = dict()
        
        for name, result in kitti_format_results.items():
            ### When reading detections from files, the length of dt_annos might be shorter than gt_annos. 
            ### We want them to have the same length.
            if len(result) != len(self.gt_annos):
                print(f"Length of {name} is not same as the length of ground truth.")
                print(f"Length of {name}: {len(result)}, Length of ground truth: {len(self.gt_annos)}")
                print(f"Aligning {name} with ground truth...")
                idx_in_gt = 0
                result_aligned = []
                for frame_result in result:
                    idx_result = frame_result['sample_idx'][0]
                    while sample_idxs[idx_in_gt] < idx_result:
                        anno_dummy = {
                            'sample_idx': np.array([sample_idxs[idx_in_gt]]),
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
                        result_aligned.append(anno_dummy)
                        idx_in_gt += 1
                    assert sample_idxs[idx_in_gt] == idx_result, f"error: idx_in_gt: {idx_in_gt}, sample_idxs[idx_in_gt]: {sample_idxs[idx_in_gt]} idx_result: {idx_result}"
                    result_aligned.append(frame_result)
                    idx_in_gt += 1
                while idx_in_gt < len(sample_idxs):
                    anno_dummy = {
                        'sample_idx': np.array([sample_idxs[idx_in_gt]]),
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
                    result_aligned.append(anno_dummy)
                    idx_in_gt += 1
                    
                result = result_aligned
            else:
                print(f"Length of {name} is same as the length of ground truth.")
                print(f"Length of {name}: {len(result)}, Length of ground truth: {len(self.gt_annos)}")

            
            if '2d' in name:
                eval_types=['bbox']
            result_string, result_dict = kitti_eval(
                gt_annos=self.gt_annos,
                dt_annos=result,
                current_classes=eval_classes,
                eval_types=eval_types)
            
            for ap_type, ap_value in result_dict.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap_value:.4f}')

            if verbose and ('2d' not in name):
                print(result_string)
        
        if save_path is not None:
            with open(save_path, 'w') as make_json:
                json.dump(ap_dict, make_json)
        return ap_dict
