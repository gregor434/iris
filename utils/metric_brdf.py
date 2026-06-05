# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os

import torch
import torch.nn.functional as NF
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BRDF metrics for one or more FIPT synthetic scenes."
    )
    parser.add_argument(
        "scenes",
        nargs="+",
        help="Synthetic FIPT scene names, e.g. bathroom bedroom kitchen livingroom.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/iris/datasets/fipt/indoor_synthetic",
        help="Root directory containing synthetic FIPT scene folders.",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory containing experiment output folders.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Dataset/output split to evaluate. Default is train because val lacks GT albedo EXR.",
    )
    parser.add_argument(
        "--exp-prefix",
        default="fipt_syn_",
        help="Experiment directory prefix under outputs/.",
    )
    return parser.parse_args()


def psnr_from_mse(mse_values):
    if not mse_values:
        return float("nan")
    mse_tensor = torch.tensor(mse_values)
    return (-10 * torch.log10(mse_tensor.clamp_min(1e-10))).mean().item()


def mean_or_nan(values):
    if not values:
        return float("nan")
    return torch.tensor(values).mean().item()


def format_metric(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def evaluate_scene(scene, dataset_root, outputs_root, split, exp_prefix):
    gt_path = os.path.join(dataset_root, scene, split)
    method = f"{exp_prefix}{scene}"
    method_path = os.path.join(outputs_root, method, "output", split)

    required_dirs = [
        os.path.join(gt_path, "Image"),
        os.path.join(gt_path, "albedo"),
        os.path.join(gt_path, "DiffCol"),
        os.path.join(gt_path, "Roughness"),
        os.path.join(gt_path, "Emit"),
        os.path.join(method_path, "a_prime"),
        os.path.join(method_path, "diffuse"),
        os.path.join(method_path, "roughness"),
        os.path.join(method_path, "emission"),
    ]
    missing_dirs = [path for path in required_dirs if not os.path.isdir(path)]
    if missing_dirs:
        missing = "\n".join(missing_dirs)
        raise FileNotFoundError(
            f"Missing required directories for scene '{scene}':\n{missing}"
        )

    image_num = len(
        [
            f
            for f in os.listdir(os.path.join(gt_path, "Image"))
            if f[0] != "." and f.endswith(".exr")
        ]
    )

    mse_roughness = []
    mse_albedo = []
    mse_diff = []
    iou_emission = []
    mse_emission = []

    for i in tqdm(range(image_num), desc=scene):
        emission_gt = cv2.imread(
            os.path.join(gt_path, "Emit", "{:03d}_0001.exr".format(i)), -1
        )[..., [2, 1, 0]]
        emission_gt = torch.from_numpy(emission_gt).float()
        emission_mask = emission_gt.sum(-1) > 0

        albedo_gt = cv2.imread(
            os.path.join(gt_path, "albedo", "{:03d}.exr".format(i)), -1
        )[..., [2, 1, 0]]
        albedo_gt = (
            torch.from_numpy(albedo_gt).float().clamp(0, 1).mul(255).long().float() / 255
        )
        albedo_gt[emission_mask] = 0

        kd_gt = cv2.imread(
            os.path.join(gt_path, "DiffCol", "{:03d}_0001.exr".format(i)), -1
        )[..., [2, 1, 0]]
        kd_gt = torch.from_numpy(kd_gt).float().clamp(0, 1).mul(255).long().float() / 255
        kd_gt[emission_mask] = 0

        roughness_gt = cv2.imread(
            os.path.join(gt_path, "Roughness", "{:03d}_0001.exr".format(i)), -1
        )[..., 0]
        roughness_gt = (
            torch.from_numpy(roughness_gt).float().mul(255).long().float() / 255
        ).clamp(0.2, 1)
        roughness_gt[emission_mask] = 0

        diff_mask = roughness_gt == 1
        kd_gt[~diff_mask] = 0

        emission = cv2.imread(
            os.path.join(method_path, "emission", "{:05d}_emission.exr".format(i)), -1
        )[..., [2, 1, 0]]
        emission = torch.from_numpy(emission).float()

        albedo = cv2.imread(
            os.path.join(method_path, "a_prime", "{:05d}_a_prime.png".format(i)), -1
        )[..., [2, 1, 0]]
        albedo = torch.from_numpy(albedo).float() / 255
        albedo[emission_mask] = 0

        kd = cv2.imread(
            os.path.join(method_path, "diffuse", "{:05d}_kd.png".format(i)), -1
        )[..., [2, 1, 0]]
        kd = torch.from_numpy(kd).float() / 255
        kd[emission_mask] = 0
        kd[~diff_mask] = 0

        roughness = cv2.imread(
            os.path.join(method_path, "roughness", "{:05d}_roughness.png".format(i)), -1
        )[:, :, 0]
        roughness = (torch.from_numpy(roughness).float() / 255).clamp(0.2, 1)
        roughness[emission_mask] = 0

        emission_mask_est = emission.sum(-1) > 0
        if emission_mask.any():
            iou = (
                (emission_mask & emission_mask_est).sum() * 1.0
                / (emission_mask | emission_mask_est).sum()
            )
            iou_emission.append(iou)
            mse_emission.append(
                NF.mse_loss(torch.log(emission + 1), torch.log(emission_gt + 1))
            )

        mse_roughness.append(NF.mse_loss(roughness, roughness_gt))
        mse_albedo.append(NF.mse_loss(albedo, albedo_gt))
        mse_diff.append(NF.mse_loss(kd, kd_gt))

    return {
        "scene": scene,
        "method": method,
        "split": split,
        "num_frames": image_num,
        "kd": psnr_from_mse(mse_diff),
        "albedo": psnr_from_mse(mse_albedo),
        "roughness": psnr_from_mse(mse_roughness),
        "emit_iou": mean_or_nan(iou_emission),
        "emit_log_mse": mean_or_nan(mse_emission),
    }


def aggregate_results(results):
    metric_names = ["kd", "albedo", "roughness", "emit_iou", "emit_log_mse"]
    aggregate = {
        "scene": "AVG",
        "split": results[0]["split"],
        "num_frames": sum(result["num_frames"] for result in results),
    }
    for metric_name in metric_names:
        values = [
            result[metric_name]
            for result in results
            if not math.isnan(result[metric_name])
        ]
        aggregate[metric_name] = sum(values) / len(values) if values else float("nan")
    return aggregate


def print_result(result):
    print(result["scene"])
    print(f"  split:        {result['split']}")
    print(f"  frames:       {result['num_frames']}")
    print(f"  kd:           {format_metric(result['kd'])}")
    print(f"  albedo:       {format_metric(result['albedo'])}")
    print(f"  roughness:    {format_metric(result['roughness'])}")
    print(f"  emit_iou:     {format_metric(result['emit_iou'])}")
    print(f"  emit_log_mse: {format_metric(result['emit_log_mse'])}")


def main():
    args = parse_args()
    results = []
    for scene in args.scenes:
        result = evaluate_scene(
            scene=scene,
            dataset_root=args.dataset_root,
            outputs_root=args.outputs_root,
            split=args.split,
            exp_prefix=args.exp_prefix,
        )
        results.append(result)
        print_result(result)

    if len(results) > 1:
        print_result(aggregate_results(results))


if __name__ == "__main__":
    main()
