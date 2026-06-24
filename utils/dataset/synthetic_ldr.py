# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from PIL import Image
import cv2
import math
from const import GAMMA, set_random_seed

set_random_seed()


def get_ray_directions(H, W, focal):
    """get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    x: left, y: up, z: forward
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = torch.stack(
        [-(i - W / 2) / focal, -(j - H / 2) / focal, torch.ones_like(i)], -1
    )

    return directions


def get_rays(directions, c2w, focal=None):
    """world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:, :3]
    rays_d = directions @ R.T
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0 / focal, 0, 0])[None, None].expand_as(directions) @ R.T
        dydv = torch.tensor([0, 1.0 / focal, 0])[None, None].expand_as(directions) @ R.T
        dxdu = dxdu.view(-1, 3)
        dydv = dydv.view(-1, 3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d


def open_exr(file, img_hw):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    else:
        img = img[..., [2, 1, 0]]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    img = torch.from_numpy(img.astype(np.float32))
    return img


def open_exr_mask(file, img_hw):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img = img[..., 0]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    return torch.from_numpy(img.astype(np.float32))


def open_png(file, img_hw, gamma=None):
    img = cv2.imread(str(file))[..., [2, 1, 0]]
    hs, ws, _ = img.shape
    ht, wt = img_hw
    if (ht != hs) or (wt != ws):
        img = cv2.resize(img, (wt, ht))
    # img = cv2.resize(img,img_hw,cv2.INTER_LANCZOS4)
    img = torch.from_numpy((img / 255).astype(np.float32))
    # convert to linear color space, the saturated pixel intensity is incorrect
    if gamma:
        img = img.pow(gamma)
    return img


def _first_existing(*paths):
    for path in paths:
        if path is not None and os.path.exists(path):
            return path
    return None


def _required_existing(description, *paths):
    path = _first_existing(*paths)
    if path is None:
        candidates = "\n".join(str(path) for path in paths if path is not None)
        raise FileNotFoundError(f"Missing {description}. Checked:\n{candidates}")
    return path


def _frame_path(directory, frame_index, *patterns, required=True):
    candidates = [
        os.path.join(directory, pattern.format(frame=frame_index))
        for pattern in patterns
        if directory is not None
    ]
    path = _first_existing(*candidates)
    if path is None and required:
        raise FileNotFoundError(
            "Missing frame {} under {}. Checked:\n{}".format(
                frame_index, directory, "\n".join(candidates)
            )
        )
    return path


class SyntheticSplitLayout:
    """Resolves canonical synthetic paths, with legacy FIPT fallback for migration."""

    def __init__(self, scene_root, split_root, img_dir=None):
        img_dir = img_dir or None
        self.scene_root = scene_root
        self.split_root = split_root
        self.img_dir_arg = img_dir

        requested_input = (
            os.path.join(split_root, img_dir) if img_dir is not None else None
        )
        self.ldr_dir = _required_existing(
            "LDR input image directory",
            requested_input,
            os.path.join(split_root, "inputs", "ldr"),
            os.path.join(split_root, "rgb"),
            os.path.join(split_root, "Image"),
        )
        self.hdr_dir = _first_existing(
            os.path.join(split_root, "inputs", "hdr"),
            os.path.join(split_root, "aovs", "rgb"),
            os.path.join(split_root, "rgb"),
            os.path.join(split_root, "Image"),
        )
        self.camera_dir = _first_existing(
            os.path.join(split_root, "cameras"),
            os.path.join(self.ldr_dir, "cam"),
        )
        self.transforms_path = _required_existing(
            "camera transforms.json",
            os.path.join(split_root, "cameras", "transforms.json"),
            os.path.join(split_root, "transforms.json"),
        )
        self.exposure_path = _first_existing(
            os.path.join(split_root, "cameras", "exposure.npy"),
            os.path.join(self.ldr_dir, "cam", "exposure.npy"),
        )
        self.crf_path = _first_existing(
            os.path.join(split_root, "cameras", "crf.npy"),
            os.path.join(self.ldr_dir, "cam", "crf.npy"),
        )

        self.aov_dirs = {
            "kd": _first_existing(
                os.path.join(split_root, "aovs", "kd"),
                os.path.join(split_root, "kd"),
                os.path.join(split_root, "DiffCol"),
            ),
            "albedo": _first_existing(
                os.path.join(split_root, "aovs", "albedo"),
                os.path.join(split_root, "albedo_pure"),
            ),
            "a_prime": _first_existing(
                os.path.join(split_root, "aovs", "a_prime"),
                os.path.join(split_root, "a_prime"),
                os.path.join(split_root, "albedo"),
            ),
            "roughness": _first_existing(
                os.path.join(split_root, "aovs", "roughness"),
                os.path.join(split_root, "roughness"),
                os.path.join(split_root, "Roughness"),
            ),
            "emission": _first_existing(
                os.path.join(split_root, "aovs", "emission"),
                os.path.join(split_root, "emission"),
                os.path.join(split_root, "Emit"),
            ),
        }
        self.label_dirs = {
            "material_id": _first_existing(
                os.path.join(split_root, "labels", "material_id"),
                os.path.join(split_root, "material_id"),
                os.path.join(split_root, "IndexMA"),
            ),
            "segmentation": _first_existing(
                os.path.join(split_root, "labels", "segmentation"),
                os.path.join(split_root, "segmentation"),
            ),
        }
        self.prior_dirs = {
            "albedo": _first_existing(
                os.path.join(split_root, "priors", "albedo"),
                os.path.join(self.ldr_dir, "albedo"),
                os.path.join(split_root, "irisformer", "albedo"),
            )
        }
        self.metallic_path = _first_existing(os.path.join(split_root, "metallic.npy"))

    def has_exposure(self):
        return self.exposure_path is not None and self.crf_path is not None

    def aov_dir(self, name, required=True):
        path = self.aov_dirs.get(name)
        if path is None and required:
            raise FileNotFoundError(f"Missing required synthetic AOV directory: {name}")
        return path

    def label_dir(self, name, required=True):
        path = self.label_dirs.get(name)
        if path is None and required:
            raise FileNotFoundError(
                f"Missing required synthetic label directory: {name}"
            )
        return path

    def prior_dir(self, name, required=True):
        path = self.prior_dirs.get(name)
        if path is None and required:
            raise FileNotFoundError(
                f"Missing required synthetic prior directory: {name}"
            )
        return path

    def ldr_path(self, frame_index):
        return _frame_path(self.ldr_dir, frame_index, "{frame:03d}_0001.png")

    def hdr_probe_path(self):
        return _frame_path(
            self.hdr_dir, 0, "{frame:03d}_0001.exr", "{frame:03d}_0001.png"
        )

    def aov_path(self, name, frame_index, required=True):
        directory = self.aov_dir(name, required=required)
        if directory is None:
            return None
        if name in ("albedo", "a_prime"):
            return _frame_path(
                directory,
                frame_index,
                "{frame:03d}.exr",
                "{frame:03d}_0001.exr",
                required=required,
            )
        return _frame_path(
            directory, frame_index, "{frame:03d}_0001.exr", required=required
        )

    def label_path(self, name, frame_index, required=True):
        directory = self.label_dir(name, required=required)
        if directory is None:
            return None
        patterns = (
            ("{frame:03d}.exr", "{frame:03d}_0001.exr")
            if name == "segmentation"
            else ("{frame:03d}_0001.exr",)
        )
        return _frame_path(directory, frame_index, *patterns, required=required)

    def albedo_prior_path(self, frame_index):
        return _frame_path(
            self.prior_dir("albedo"),
            frame_index,
            "{frame:03d}_0001.png",
            "{frame:03d}.png",
        )


def _load_optional_exr(path, img_hw):
    if path is None:
        return None
    return open_exr(path, img_hw)


class SyntheticDatasetLDR(Dataset):
    """Synthetic FIPT split with canonical AOV names.

    Scene/
        {split}/
            cameras/{transforms.json, exposure.npy, crf.npy}
            inputs/ldr/{:03d}_0001.png
            inputs/hdr/{:03d}_0001.exr
            aovs/{kd, albedo, a_prime, roughness, emission}/...
            labels/material_id/{:03d}_0001.exr
            labels/segmentation/{:03d}.exr when available
            priors/albedo/{:03d}_0001.png
    """

    def __init__(
        self,
        root_dir,
        img_dir=None,
        split="train",
        pixel=True,
        ray_diff=False,
        load_traj=False,
        val_frame=0,
        res_scale=1.0,
    ):
        """
        Args:
            root_dir: dataset root folder
            split: train or val
            pixel: whether load every camera pixel
            ray_diff: whether load ray differentials
        """
        self.root_dir = (
            os.path.join(root_dir, split)
            if split != "relight"
            else os.path.join(root_dir, "val")
        )
        self.layout = SyntheticSplitLayout(root_dir, self.root_dir, img_dir=img_dir)
        if self.layout.has_exposure():
            self.exposures = np.load(self.layout.exposure_path)
            self.crfs = np.load(self.layout.crf_path)  # (3, 1024)
            self.multi_exposure = True
            self.gamma = None
        else:
            self.exposures = None
            self.crfs = None
            self.multi_exposure = False
            self.gamma = GAMMA
        self.pixel = pixel
        self.split = split
        self.has_segmentation = (
            self.layout.label_dir("segmentation", required=False) is not None
        )
        self.has_albedo_gt = self.layout.aov_dir("albedo", required=False) is not None
        self.has_a_prime_gt = self.layout.aov_dir("a_prime", required=False) is not None

        train_layout = SyntheticSplitLayout(
            root_dir, os.path.join(root_dir, "train"), img_dir=img_dir
        )
        h, w = cv2.imread(train_layout.hdr_probe_path(), -1).shape[:2]
        self.img_hw = (int(h * res_scale), int(w * res_scale))

        self.ray_diff = ray_diff
        self.val_frame = val_frame

        with open(self.layout.transforms_path, "r") as f:
            self.meta = json.load(f)

        # camera focal length and ray directions
        h, w = self.img_hw
        self.focal = (0.5 * w / np.tan(0.5 * self.meta["camera_angle_x"])).item()
        self.directions = get_ray_directions(h, w, self.focal)

        # load every camera pixels
        if self.pixel:
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for cur_idx in range(len(self.meta["frames"])):
                frame = self.meta["frames"][cur_idx]
                pose = np.array(frame["transform_matrix"])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = self.layout.ldr_path(cur_idx)
                img = open_png(image_path, self.img_hw, self.gamma).reshape(-1, 3)

                # load ground truth BRDF
                kd = open_exr(self.layout.aov_path("kd", cur_idx), self.img_hw).reshape(
                    -1, 3
                )
                a_prime_gt = _load_optional_exr(
                    self.layout.aov_path("a_prime", cur_idx, required=False),
                    self.img_hw,
                )
                a_prime = (
                    a_prime_gt.reshape(-1, 3)
                    if a_prime_gt is not None
                    else torch.zeros_like(kd)
                )
                albedo_gt = _load_optional_exr(
                    self.layout.aov_path("albedo", cur_idx, required=False), self.img_hw
                )
                albedo = (
                    albedo_gt.reshape(-1, 3)
                    if albedo_gt is not None
                    else torch.zeros_like(a_prime)
                )
                roughness = open_exr(
                    self.layout.aov_path("roughness", cur_idx), self.img_hw
                ).reshape(-1, 3)[..., :1]
                emission = open_exr(
                    self.layout.aov_path("emission", cur_idx), self.img_hw
                ).reshape(-1, 3)
                material_id = open_exr_mask(
                    self.layout.label_path("material_id", cur_idx),
                    self.img_hw,
                ).reshape(-1, 1)
                segmentation = None
                if self.has_segmentation:
                    segmentation = open_exr_mask(
                        self.layout.label_path("segmentation", cur_idx),
                        self.img_hw,
                    ).reshape(-1, 1)

                self.all_rgbs += [img]

                if not self.ray_diff:
                    rays_o, rays_d = get_rays(self.directions, c2w)

                    ray_tensors = [
                        rays_o,
                        rays_d,
                        kd,
                        albedo,
                        a_prime,
                        roughness,
                        emission,
                        material_id,
                    ]
                    if segmentation is not None:
                        ray_tensors.append(segmentation)
                    self.all_rays += [torch.cat(ray_tensors, 1)]
                else:
                    rays_o, rays_d, dxdu, dydv = get_rays(
                        self.directions, c2w, focal=self.focal
                    )
                    ray_tensors = [
                        rays_o,
                        rays_d,
                        dxdu,
                        dydv,
                        kd,
                        albedo,
                        a_prime,
                        roughness,
                        emission,
                        material_id,
                    ]
                    if segmentation is not None:
                        ray_tensors.append(segmentation)
                    self.all_rays += [torch.cat(ray_tensors, 1)]

            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            if self.multi_exposure:
                pixel_num = self.img_hw[0] * self.img_hw[1]
                self.exposures = (
                    torch.tensor(self.exposures)
                    .view(-1, 1, 1)
                    .repeat(1, pixel_num, 1)
                    .view(-1, 1)
                )

        if load_traj:
            render_traj = np.load(os.path.join(root_dir, "render_traj.npy"))
            self.render_traj_c2w = render_traj
            self.render_traj_rays = []
            for i in range(len(render_traj)):
                c2w = torch.FloatTensor(render_traj[i])
                rays_o, rays_d, dxdu, dydv = get_rays(
                    self.directions, c2w, focal=self.focal
                )
                self.render_traj_rays += [torch.cat([rays_o, rays_d, dxdu, dydv], 1)]

    def __len__(self):
        if self.pixel:
            return len(self.all_rays)
        # if self.split == 'val':
        #     # only show 8 images for reconstruction validation
        #     return 8
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        exposure = None
        if self.pixel:
            if self.multi_exposure:
                exposure = self.exposures[idx]
            tmp = self.all_rays[idx]
            if not self.ray_diff:
                sample = {
                    "rays": tmp[:8],
                    "rgbs": self.all_rgbs[idx],
                    "kd": tmp[8:11],
                    "albedo": tmp[11:14],
                    "a_prime": tmp[14:17],
                    "roughness": tmp[17],
                    "emission": tmp[18:21],
                    "material_id": tmp[21],
                    "exposure": exposure,
                }
                if self.has_segmentation:
                    sample["segmentation"] = tmp[22]
                sample["has_albedo_gt"] = self.has_albedo_gt
                sample["has_a_prime_gt"] = self.has_a_prime_gt
            else:
                sample = {
                    "rays": tmp[:12],
                    "rgbs": self.all_rgbs[idx],
                    "kd": tmp[12:15],
                    "albedo": tmp[15:18],
                    "a_prime": tmp[18:21],
                    "roughness": tmp[21],
                    "emission": tmp[22:25],
                    "material_id": tmp[25],
                    "exposure": exposure,
                }
                if self.has_segmentation:
                    sample["segmentation"] = tmp[26]
                sample["has_albedo_gt"] = self.has_albedo_gt
                sample["has_a_prime_gt"] = self.has_a_prime_gt

        else:
            frame = self.meta["frames"][idx]
            c2w = torch.FloatTensor(frame["transform_matrix"])[:3, :4]

            cur_idx = idx

            image_path = self.layout.ldr_path(cur_idx)
            img = open_png(image_path, self.img_hw, self.gamma).reshape(-1, 3)

            kd = open_exr(self.layout.aov_path("kd", cur_idx), self.img_hw).reshape(
                -1, 3
            )
            a_prime_gt = _load_optional_exr(
                self.layout.aov_path("a_prime", cur_idx, required=False), self.img_hw
            )
            a_prime = (
                a_prime_gt.reshape(-1, 3)
                if a_prime_gt is not None
                else torch.zeros_like(kd)
            )
            albedo_gt = _load_optional_exr(
                self.layout.aov_path("albedo", cur_idx, required=False), self.img_hw
            )
            albedo = (
                albedo_gt.reshape(-1, 3)
                if albedo_gt is not None
                else torch.zeros_like(a_prime)
            )
            roughness = open_exr(
                self.layout.aov_path("roughness", cur_idx), self.img_hw
            ).reshape(-1, 3)[..., 0]
            emission = open_exr(
                self.layout.aov_path("emission", cur_idx), self.img_hw
            ).reshape(-1, 3)
            material_id = open_exr_mask(
                self.layout.label_path("material_id", cur_idx),
                self.img_hw,
            ).reshape(-1)
            segmentation = None
            if self.has_segmentation:
                segmentation = open_exr_mask(
                    self.layout.label_path("segmentation", cur_idx),
                    self.img_hw,
                ).reshape(-1)

            if not self.ray_diff:
                rays_o, rays_d = get_rays(self.directions, c2w)

                rays = torch.cat([rays_o, rays_d], 1)
            else:
                rays_o, rays_d, dxdu, dydv = get_rays(
                    self.directions, c2w, focal=self.focal
                )
                rays = torch.cat([rays_o, rays_d, dxdu, dydv], 1)
            if self.multi_exposure:
                exposure = self.exposures[idx]
            sample = {
                "rays": rays,
                "rgbs": img,
                "c2w": c2w,
                "kd": kd,
                "albedo": albedo,
                "a_prime": a_prime,
                "roughness": roughness,
                "emission": emission,
                "material_id": material_id,
                "has_albedo_gt": self.has_albedo_gt,
                "has_a_prime_gt": self.has_a_prime_gt,
                "exposure": exposure,
            }
            if segmentation is not None:
                sample["segmentation"] = segmentation

        return sample


class InvSyntheticDatasetLDR(Dataset):
    """Synthetic inverse-rendering dataset on the canonical synthetic layout.

    Input AOVs follow the same split contract as SyntheticDatasetLDR:
        cameras/, inputs/, aovs/, labels/, and priors/.

    Optional cached shadings:
        diffuse/{:03d}.exr diffuse shading (L_d)
        specular/{:03d}_i_j.exr specular shading samples (L_s^i(\sigma_j))
    """

    def __init__(
        self,
        root_dir,
        img_dir=None,
        batch_size=None,
        split="train",
        pixel=True,
        has_part=False,
        load_metallic=False,
        val_frame=0,
        cache_dir=None,
    ):
        """
        Args:
            root_dir: dataset root folder
            cache_dir: shadings folder
            split: train or val
            pixel: whether load every camera pixel
            batch_size: size of each ray batch if pixel==True
            has_part: whether use material_id supervision instead of semantic segmentation
        """
        self.root_dir = os.path.join(root_dir, split)
        self.cache_dir = cache_dir
        self.layout = SyntheticSplitLayout(root_dir, self.root_dir, img_dir=img_dir)
        if self.layout.has_exposure():
            self.exposures = np.load(self.layout.exposure_path)
            self.crfs = np.load(self.layout.crf_path)  # (3, 1024)
            self.multi_exposure = True
            self.gamma = None
        else:
            self.exposures = None
            self.crfs = None
            self.multi_exposure = False
            self.gamma = GAMMA
        self.pixel = pixel
        self.split = split
        self.batch_size = batch_size
        self.has_part = has_part
        self.has_albedo_gt = self.layout.aov_dir("albedo", required=False) is not None
        self.has_a_prime_gt = self.layout.aov_dir("a_prime", required=False) is not None
        self.load_metallic = load_metallic
        self.val_frame = val_frame
        # approximate roughness channel by interpolating 6 samples
        self.roughness_level = 6

        train_layout = SyntheticSplitLayout(
            root_dir, os.path.join(root_dir, "train"), img_dir=img_dir
        )
        self.img_hw = cv2.imread(train_layout.hdr_probe_path(), -1).shape[:2]

        with open(self.layout.transforms_path, "r") as f:
            self.meta = json.load(f)

        h, w = self.img_hw
        self.focal = (0.5 * w / np.tan(0.5 * self.meta["camera_angle_x"])).item()
        self.directions = get_ray_directions(h, w, self.focal)

        if self.pixel:
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_intrinsic = []
            self.all_cache = []
            for cur_idx in range(len(self.meta["frames"])):
                frame = self.meta["frames"][cur_idx]
                pose = np.array(frame["transform_matrix"])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = self.layout.ldr_path(cur_idx)
                img = open_png(image_path, self.img_hw, self.gamma).reshape(-1, 3)

                # ground truth brdf-emission
                kd = open_exr(self.layout.aov_path("kd", cur_idx), self.img_hw).reshape(
                    -1, 3
                )
                a_prime_gt = _load_optional_exr(
                    self.layout.aov_path("a_prime", cur_idx, required=False),
                    self.img_hw,
                )
                a_prime = (
                    a_prime_gt.reshape(-1, 3)
                    if a_prime_gt is not None
                    else torch.zeros_like(kd)
                )
                albedo_gt = _load_optional_exr(
                    self.layout.aov_path("albedo", cur_idx, required=False), self.img_hw
                )
                albedo = (
                    albedo_gt.reshape(-1, 3)
                    if albedo_gt is not None
                    else torch.zeros_like(a_prime)
                )
                roughness = open_exr(
                    self.layout.aov_path("roughness", cur_idx), self.img_hw
                ).reshape(-1, 3)[:, :1]
                emission = open_exr(
                    self.layout.aov_path("emission", cur_idx), self.img_hw
                ).reshape(-1, 3)

                material_id = None
                segmentation = None
                if self.has_part:
                    material_id = open_exr_mask(
                        self.layout.label_path("material_id", cur_idx), self.img_hw
                    ).reshape(-1, 1)
                else:
                    segmentation = open_exr_mask(
                        self.layout.label_path("segmentation", cur_idx), self.img_hw
                    ).reshape(-1, 1)

                self.all_rgbs += [img]
                rays_o, rays_d, dxdu, dydv = get_rays(
                    self.directions, c2w, focal=self.focal
                )  # both (h*w, 3)

                self.all_rays += [
                    torch.cat(
                        [
                            rays_o,
                            rays_d,
                            dxdu,
                            dydv,
                            kd,
                            albedo,
                            a_prime,
                            roughness,
                            emission,
                            material_id if material_id is not None else segmentation,
                        ],
                        1,
                    )
                ]

                albedo_prior_path = self.layout.albedo_prior_path(cur_idx)
                albedo_prior = np.array(Image.open(albedo_prior_path)).reshape(-1, 3)
                albedo_prior = torch.from_numpy(albedo_prior / 255.0).float()
                self.all_intrinsic += [albedo_prior]

                if self.cache_dir is not None:
                    # load shadings
                    diffuse = open_exr(
                        os.path.join(
                            self.cache_dir, "diffuse", "{:03d}.exr".format(cur_idx)
                        ),
                        self.img_hw,
                    ).reshape(-1, 3)
                    speculars0, speculars1 = [], []
                    for r_idx in range(self.roughness_level):
                        specular0 = open_exr(
                            os.path.join(
                                self.cache_dir,
                                "specular",
                                "{:03d}_0_{}.exr".format(cur_idx, r_idx),
                            ),
                            self.img_hw,
                        ).float()
                        specular0 = specular0.reshape(-1, 3)
                        speculars0.append(specular0)
                        specular1 = open_exr(
                            os.path.join(
                                self.cache_dir,
                                "specular",
                                "{:03d}_1_{}.exr".format(cur_idx, r_idx),
                            ),
                            self.img_hw,
                        ).float()
                        specular1 = specular1.reshape(-1, 3)
                        speculars1.append(specular1)
                    speculars0 = torch.cat(speculars0, -1)
                    speculars1 = torch.cat(speculars1, -1)
                    self.all_cache += [
                        torch.cat(
                            [
                                diffuse,
                                speculars0,
                                speculars1,
                            ],
                            1,
                        )
                    ]

            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_intrinsic = torch.cat(self.all_intrinsic, 0)
            if self.cache_dir is not None:
                self.all_cache = torch.cat(self.all_cache, 0)

            # number of camera ray batches
            self.batch_num = math.ceil(len(self.all_rays) * 1.0 / self.batch_size)
            self.idxs = torch.randperm(len(self.all_rays))
            if self.multi_exposure:
                pixel_num = self.img_hw[0] * self.img_hw[1]
                self.exposures = (
                    torch.tensor(self.exposures)
                    .view(-1, 1, 1)
                    .repeat(1, pixel_num, 1)
                    .view(-1, 1)
                )
            if load_metallic:
                metallic_all = np.load(os.path.join(self.root_dir, "metallic.npy"))
                self.metallic_all = metallic_all.reshape(-1)

    def resample(
        self,
    ):
        # resample camera ray batches
        self.idxs = torch.randperm(len(self.all_rays))

    def __len__(self):
        if self.pixel:
            return self.batch_num
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        exposure = None
        if self.pixel:
            b0 = idx * self.batch_size
            b1 = min(b0 + self.batch_size, len(self.all_rays))

            # find camera ray indices in the batch
            idx = self.idxs[b0:b1]
            tmp = self.all_rays[idx]
            if self.multi_exposure:
                exposure = self.exposures[idx]

            diffuse, specular0, specular1 = None, None, None
            if self.cache_dir is not None:
                cache = self.all_cache[idx]
                diffuse = cache[..., :3]
                specular0 = cache[..., 3:21].reshape(b1 - b0, -1, 3)
                specular1 = cache[..., 21:39].reshape(b1 - b0, -1, 3)

            sample = {
                "rays": tmp[..., :12],
                "kd": tmp[..., 12:15],
                "albedo": tmp[..., 15:18],
                "a_prime": tmp[..., 18:21],
                "roughness": tmp[..., 21],
                "emission": tmp[..., 22:25],
                "rgbs": self.all_rgbs[idx],
                "albedo_prior": self.all_intrinsic[idx],
                "has_albedo_gt": self.has_albedo_gt,
                "has_a_prime_gt": self.has_a_prime_gt,
                "exposure": exposure,
                "diffuse": diffuse,
                "specular0": specular0,
                "specular1": specular1,
            }
            if self.has_part:
                sample["material_id"] = tmp[..., 25]
            else:
                sample["segmentation"] = tmp[..., 25]
            if self.load_metallic:
                sample["metallic"] = self.metallic_all[idx]

        else:
            frame = self.meta["frames"][idx]
            c2w = torch.FloatTensor(frame["transform_matrix"])[:3, :4]
            cur_idx = idx

            image_path = self.layout.ldr_path(cur_idx)
            img = open_png(image_path, self.img_hw, self.gamma).reshape(-1, 3)

            kd = open_exr(self.layout.aov_path("kd", cur_idx), self.img_hw).reshape(
                -1, 3
            )
            a_prime_gt = _load_optional_exr(
                self.layout.aov_path("a_prime", cur_idx, required=False), self.img_hw
            )
            a_prime = (
                a_prime_gt.reshape(-1, 3)
                if a_prime_gt is not None
                else torch.zeros_like(kd)
            )
            albedo_gt = _load_optional_exr(
                self.layout.aov_path("albedo", cur_idx, required=False), self.img_hw
            )
            albedo = (
                albedo_gt.reshape(-1, 3)
                if albedo_gt is not None
                else torch.zeros_like(a_prime)
            )
            roughness = open_exr(
                self.layout.aov_path("roughness", cur_idx), self.img_hw
            ).reshape(-1, 3)[:, 0]
            emission = open_exr(
                self.layout.aov_path("emission", cur_idx), self.img_hw
            ).reshape(-1, 3)

            material_id = None
            segmentation = None
            if self.has_part:
                material_id = open_exr_mask(
                    self.layout.label_path("material_id", cur_idx), self.img_hw
                ).reshape(-1)
            else:
                segmentation = open_exr_mask(
                    self.layout.label_path("segmentation", cur_idx), self.img_hw
                ).reshape(-1)

            rays_o, rays_d, dxdu, dydv = get_rays(
                self.directions, c2w, focal=self.focal
            )

            rays = torch.cat([rays_o, rays_d, dxdu, dydv], -1)

            albedo_prior_path = self.layout.albedo_prior_path(cur_idx)
            albedo_prior = np.array(Image.open(albedo_prior_path)).reshape(-1, 3)
            albedo_prior = torch.from_numpy(albedo_prior / 255.0).float()
            if self.multi_exposure:
                exposure = self.exposures[idx]

            diffuse, speculars0, speculars1 = None, None, None
            if self.cache_dir is not None:
                diffuse = open_exr(
                    os.path.join(
                        self.cache_dir, "diffuse", "{:03d}.exr".format(cur_idx)
                    ),
                    self.img_hw,
                ).reshape(-1, 3)
                speculars0, speculars1 = [], []
                for r_idx in range(self.roughness_level):
                    specular0 = open_exr(
                        os.path.join(
                            self.cache_dir,
                            "specular",
                            "{:03d}_0_{}.exr".format(cur_idx, r_idx),
                        ),
                        self.img_hw,
                    ).float()
                    specular0 = specular0.reshape(-1, 3)
                    speculars0.append(specular0)
                    specular1 = open_exr(
                        os.path.join(
                            self.cache_dir,
                            "specular",
                            "{:03d}_1_{}.exr".format(cur_idx, r_idx),
                        ),
                        self.img_hw,
                    ).float()
                    specular1 = specular1.reshape(-1, 3)
                    speculars1.append(specular1)
                speculars0 = torch.stack(speculars0, -2)
                speculars1 = torch.stack(speculars1, -2)

            sample = {
                "rays": rays,
                "rgbs": img,
                "c2w": c2w,
                "kd": kd,
                "albedo": albedo,
                "a_prime": a_prime,
                "roughness": roughness,
                "emission": emission,
                "albedo_prior": albedo_prior,
                "has_albedo_gt": self.has_albedo_gt,
                "has_a_prime_gt": self.has_a_prime_gt,
                "exposure": exposure,
                "diffuse": diffuse,
                "specular0": speculars0,
                "specular1": speculars1,
            }
            if self.has_part:
                sample["material_id"] = material_id
            else:
                sample["segmentation"] = segmentation
            if self.load_metallic:
                metallic_all = np.load(os.path.join(self.root_dir, "metallic.npy"))
                metallic = metallic_all[cur_idx].reshape(-1)
                sample["metallic"] = metallic
        return sample
