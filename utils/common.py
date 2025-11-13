# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Common utility functions used across multiple modules.
This module consolidates duplicated code to improve maintainability.
"""

import os
import numpy as np
import torch
import cv2
import mitsuba
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser


def save_image(image, path, colormap=False, crop_even=False):
    """
    Save an image to disk with optional colormap and cropping.
    
    Args:
        image: Image tensor or numpy array to save
        path: Output file path
        colormap: If True, apply MAGMA colormap
        crop_even: If True, crop to even dimensions (for video encoding)
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image*255).astype(np.uint8)
    if colormap:
        image = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_even:
        h, w = image.shape[:2]
        image = image[:h-h%2, :w-w%2]
    image = Image.fromarray(image)
    image.save(path)
    if crop_even:
        return np.array(image)


def add_model_specific_args(parent_parser, default_options):
    """
    Add model-specific arguments to the argument parser.
    
    Args:
        parent_parser: Parent ArgumentParser to extend
        default_options: Dictionary of default configuration options
        
    Returns:
        ArgumentParser with added model-specific arguments
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    for name, args in default_options.items():
        if args['type'] == bool:
            parser.add_argument('--{}'.format(name), type=eval, 
                              choices=[True, False], 
                              default=str(args.get('default')))
        else:
            parser.add_argument('--{}'.format(name), **args)
    return parser


def load_mesh(dataset_name, dataset_path, scene=None):
    """
    Load scene geometry mesh based on dataset type.
    
    Args:
        dataset_name: Name of the dataset ('synthetic', 'real', or 'scannetpp')
        dataset_path: Root path to the dataset
        scene: Scene name (required for scannetpp)
        
    Returns:
        tuple: (mitsuba_scene, mesh_path, mesh_type)
    """
    if dataset_name in ['synthetic', 'real']:
        mesh_path = os.path.join(dataset_path, 'scene.obj')
        mesh_type = 'obj'
    elif dataset_name == 'scannetpp':
        if scene is None:
            raise ValueError("scene parameter is required for scannetpp dataset")
        mesh_path = os.path.join(dataset_path, 'data', scene, 'scans', 'scene.ply')
        mesh_type = 'ply'
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    assert Path(mesh_path).exists(), f'mesh not found: {mesh_path}'
    
    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id': {
            'type': mesh_type,
            'filename': mesh_path
        }
    })
    
    return scene, mesh_path, mesh_type


def load_checkpoint_weights(checkpoint_path, prefix_filter=None):
    """
    Load weights from a checkpoint file with optional prefix filtering.
    
    Args:
        checkpoint_path: Path to checkpoint file
        prefix_filter: If provided, only load weights with keys containing this prefix
                      (prefix will be removed from the key names)
        
    Returns:
        dict: Filtered state dictionary
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    
    if prefix_filter is None:
        return state_dict
    
    weight = {}
    for k, v in state_dict.items():
        if prefix_filter in k:
            weight[k.replace(prefix_filter, '')] = v
    
    return weight


def load_dataset(dataset_name, dataset_path, img_dir, split, pixel, scene=None,
                ray_diff=False, batch_size=None, has_part=False, cache_dir=None,
                res_scale=1, val_frame=None, load_traj=False):
    """
    Load dataset based on dataset type and configuration.
    
    Args:
        dataset_name: Name of the dataset ('synthetic', 'real', or 'scannetpp')
        dataset_path: Root path to the dataset
        img_dir: Image directory name
        split: Data split ('train', 'val', 'test')
        pixel: Whether to use pixel-level sampling
        scene: Scene name (required for scannetpp)
        ray_diff: Whether to compute ray differentials
        batch_size: Batch size for training
        has_part: Whether to use part segmentation
        cache_dir: Cache directory path
        res_scale: Resolution scale factor
        val_frame: Validation frame index
        load_traj: Whether to load trajectory
        
    Returns:
        Dataset object
    """
    from utils.dataset import (InvRealDatasetLDR, RealDatasetLDR,
                               InvSyntheticDatasetLDR, SyntheticDatasetLDR)
    from utils.dataset.scannetpp.dataset import Scannetpp, InvScannetpp
    
    # Determine if we need inverse dataset (for training)
    use_inv = pixel and batch_size is not None
    
    if dataset_name == 'synthetic':
        if use_inv:
            dataset = InvSyntheticDatasetLDR(
                dataset_path, img_dir=img_dir, pixel=pixel, split=split,
                batch_size=batch_size, has_part=has_part, cache_dir=cache_dir)
        else:
            dataset = SyntheticDatasetLDR(
                dataset_path, img_dir=img_dir, pixel=pixel, split=split,
                ray_diff=ray_diff, val_frame=val_frame, load_traj=load_traj,
                res_scale=res_scale)
    elif dataset_name == 'real':
        if use_inv:
            dataset = InvRealDatasetLDR(
                dataset_path, img_dir=img_dir, pixel=pixel, split=split,
                batch_size=batch_size, cache_dir=cache_dir)
        else:
            dataset = RealDatasetLDR(
                dataset_path, img_dir=img_dir, pixel=pixel, split=split,
                ray_diff=ray_diff, val_frame=val_frame, load_traj=load_traj,
                res_scale=res_scale)
    elif dataset_name == 'scannetpp':
        if scene is None:
            raise ValueError("scene parameter is required for scannetpp dataset")
        if use_inv:
            dataset = InvScannetpp(
                dataset_path, scene, pixel=pixel, split=split,
                batch_size=batch_size, cache_dir=cache_dir, res_scale=res_scale)
        else:
            dataset = Scannetpp(
                dataset_path, scene, pixel=pixel, split=split,
                ray_diff=ray_diff, val_frame=val_frame, load_traj=load_traj,
                res_scale=res_scale)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return dataset
