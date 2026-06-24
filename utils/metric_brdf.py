# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as NF
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


METRIC_COLUMNS = ["kd", "albedo", "a_prime", "roughness", "emit_iou", "emit_log_mse"]
REQUIRED_MANIFEST_COLUMNS = ["gt_path", "pred_path"]
REQUIRED_GT_AOVS = ["kd", "roughness", "emission"]
OPTIONAL_GT_AOVS = ["albedo", "a_prime"]
PRED_SUBDIRS = ["kd", "roughness", "emission"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BRDF metrics from an explicit GT/prediction manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with at least gt_path and pred_path columns.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Manifest columns to aggregate by, e.g. model dataset.",
    )
    parser.add_argument(
        "--split",
        help="Optional split value to filter the manifest by its split column.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for exporting per-row and aggregate metrics as CSV.",
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
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def load_rgb_image(path):
    image = cv2.imread(path, -1)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = image[..., None].repeat(3, axis=-1)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected 2D or 3-channel image at {path}, got shape {image.shape}"
        )
    return image[..., [2, 1, 0]]


def load_scalar_image(path):
    image = cv2.imread(path, -1)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return image[..., 0]
    raise ValueError(
        f"Expected 2D or 3-channel image at {path}, got shape {image.shape}"
    )


def first_existing_dir(*paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    return None


def resolve_gt_aov_dir(gt_path, aov_name, required=True):
    legacy_names = {
        "kd": "DiffCol",
        "a_prime": "albedo",
        "roughness": "Roughness",
        "emission": "Emit",
    }
    candidates = [
        os.path.join(gt_path, "aovs", aov_name),
        os.path.join(gt_path, aov_name),
    ]
    if aov_name == "albedo":
        candidates.append(os.path.join(gt_path, "albedo_pure"))
    legacy_name = legacy_names.get(aov_name)
    if legacy_name is not None:
        candidates.append(os.path.join(gt_path, legacy_name))

    directory = first_existing_dir(*candidates)
    if directory is None and required:
        raise FileNotFoundError(
            f"Missing required GT AOV directory '{aov_name}'. Checked:\n"
            + "\n".join(candidates)
        )
    return directory


def resolve_rgb_count_dir(gt_path):
    directory = first_existing_dir(
        os.path.join(gt_path, "inputs", "hdr"),
        os.path.join(gt_path, "aovs", "rgb"),
        os.path.join(gt_path, "rgb"),
        os.path.join(gt_path, "Image"),
    )
    if directory is None:
        raise FileNotFoundError(f"Missing GT RGB/HDR frame directory under {gt_path}")
    return directory


def resolve_gt_frame_path(directory, frame_index, *patterns):
    candidates = [
        os.path.join(directory, pattern.format(frame_index))
        for pattern in patterns
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Missing GT frame {}. Checked:\n{}".format(
            frame_index,
            "\n".join(candidates),
        )
    )


def read_manifest(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest '{path}' is missing a header row.")

        missing_columns = [
            column
            for column in REQUIRED_MANIFEST_COLUMNS
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Manifest '{path}' is missing required columns: {', '.join(missing_columns)}"
            )

        rows = []
        for row_index, row in enumerate(reader, start=2):
            cleaned_row = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
            if not any(cleaned_row.values()):
                continue
            for column in REQUIRED_MANIFEST_COLUMNS:
                if not cleaned_row.get(column):
                    raise ValueError(
                        f"Manifest '{path}' row {row_index} is missing a value for '{column}'."
                    )
            cleaned_row["manifest_row"] = row_index
            rows.append(cleaned_row)

    if not rows:
        raise ValueError(f"Manifest '{path}' does not contain any evaluation rows.")
    return rows, reader.fieldnames


def filter_manifest_rows(rows, split):
    if split is None:
        return rows
    if "split" not in rows[0]:
        raise ValueError(
            "Cannot use --split because the manifest does not define a 'split' column."
        )
    filtered_rows = [row for row in rows if row.get("split") == split]
    if not filtered_rows:
        raise ValueError(f"Manifest contains no rows with split='{split}'.")
    return filtered_rows


def validate_group_by_columns(group_by, fieldnames):
    unknown_columns = [column for column in group_by if column not in fieldnames]
    if unknown_columns:
        raise ValueError(
            "Unknown group-by columns: "
            + ", ".join(unknown_columns)
            + ". Available columns: "
            + ", ".join(fieldnames)
        )


def build_row_label(row):
    label_parts = [
        row.get("model", ""),
        row.get("dataset", ""),
        row.get("scene", ""),
        row.get("split", ""),
    ]
    label_parts = [part for part in label_parts if part]
    if label_parts:
        return " | ".join(label_parts)
    return f"row {row['manifest_row']}"


def evaluate_row(row):
    gt_path = row["gt_path"]
    pred_path = row["pred_path"]

    gt_dirs = {
        aov_name: resolve_gt_aov_dir(gt_path, aov_name) for aov_name in REQUIRED_GT_AOVS
    }
    gt_dirs.update(
        {
            aov_name: resolve_gt_aov_dir(gt_path, aov_name, required=False)
            for aov_name in OPTIONAL_GT_AOVS
        }
    )

    required_dirs = [os.path.join(pred_path, subdir) for subdir in PRED_SUBDIRS]
    if gt_dirs["albedo"] is not None:
        required_dirs.append(os.path.join(pred_path, "albedo"))
    if gt_dirs["a_prime"] is not None:
        required_dirs.append(os.path.join(pred_path, "a_prime"))
    missing_dirs = [path for path in required_dirs if not os.path.isdir(path)]
    if missing_dirs:
        missing = "\n".join(missing_dirs)
        raise FileNotFoundError(
            f"Missing required directories for manifest row {row['manifest_row']}:\n{missing}"
        )

    image_num = len(
        [
            filename
            for filename in os.listdir(resolve_rgb_count_dir(gt_path))
            if not filename.startswith(".") and filename.endswith(".exr")
        ]
    )

    mse_roughness = []
    mse_albedo = []
    mse_a_prime = []
    mse_kd = []
    iou_emission = []
    mse_emission = []

    for frame_index in tqdm(range(image_num), desc=build_row_label(row)):
        emission_gt = load_rgb_image(
            os.path.join(gt_dirs["emission"], "{:03d}_0001.exr".format(frame_index))
        )
        emission_gt = torch.from_numpy(emission_gt).float()
        emission_mask = emission_gt.sum(-1) > 0

        albedo_gt = None
        if gt_dirs["albedo"] is not None:
            albedo_gt = load_rgb_image(
                resolve_gt_frame_path(
                    gt_dirs["albedo"],
                    frame_index,
                    "{:03d}.exr",
                    "{:03d}_0001.exr",
                )
            )
            albedo_gt = (
                torch.from_numpy(albedo_gt).float().clamp(0, 1).mul(255).long().float()
                / 255
            )
            albedo_gt[emission_mask] = 0

        kd_gt = load_rgb_image(
            os.path.join(gt_dirs["kd"], "{:03d}_0001.exr".format(frame_index))
        )
        kd_gt = (
            torch.from_numpy(kd_gt).float().clamp(0, 1).mul(255).long().float() / 255
        )
        kd_gt[emission_mask] = 0

        a_prime_gt = None
        if gt_dirs["a_prime"] is not None:
            a_prime_gt = load_rgb_image(
                resolve_gt_frame_path(
                    gt_dirs["a_prime"],
                    frame_index,
                    "{:03d}.exr",
                    "{:03d}_0001.exr",
                )
            )
            a_prime_gt = (
                torch.from_numpy(a_prime_gt).float().clamp(0, 1).mul(255).long().float()
                / 255
            )
            a_prime_gt[emission_mask] = 0

        roughness_gt = load_scalar_image(
            os.path.join(gt_dirs["roughness"], "{:03d}_0001.exr".format(frame_index))
        )
        roughness_gt = (
            torch.from_numpy(roughness_gt).float().mul(255).long().float() / 255
        ).clamp(0.2, 1)
        roughness_gt[emission_mask] = 0

        diff_mask = roughness_gt == 1
        kd_gt[~diff_mask] = 0

        emission = load_rgb_image(
            os.path.join(
                pred_path, "emission", "{:05d}_emission.exr".format(frame_index)
            )
        )
        emission = torch.from_numpy(emission).float()

        albedo = None
        if albedo_gt is not None:
            albedo = load_rgb_image(
                os.path.join(
                    pred_path, "albedo", "{:05d}_albedo.png".format(frame_index)
                )
            )
            albedo = torch.from_numpy(albedo).float() / 255
            albedo[emission_mask] = 0

        a_prime = None
        if a_prime_gt is not None:
            a_prime = load_rgb_image(
                os.path.join(
                    pred_path, "a_prime", "{:05d}_a_prime.png".format(frame_index)
                )
            )
            a_prime = torch.from_numpy(a_prime).float() / 255
            a_prime[emission_mask] = 0

        kd = load_rgb_image(
            os.path.join(pred_path, "kd", "{:05d}_kd.png".format(frame_index))
        )
        kd = torch.from_numpy(kd).float() / 255
        kd[emission_mask] = 0
        kd[~diff_mask] = 0

        roughness = load_scalar_image(
            os.path.join(
                pred_path, "roughness", "{:05d}_roughness.png".format(frame_index)
            )
        )
        roughness = (torch.from_numpy(roughness).float() / 255).clamp(0.2, 1)
        roughness[emission_mask] = 0

        emission_mask_est = emission.sum(-1) > 0
        if emission_mask.any():
            iou = (
                (emission_mask & emission_mask_est).sum()
                * 1.0
                / (emission_mask | emission_mask_est).sum()
            )
            iou_emission.append(iou)
            mse_emission.append(
                NF.mse_loss(torch.log(emission + 1), torch.log(emission_gt + 1))
            )

        mse_roughness.append(NF.mse_loss(roughness, roughness_gt))
        if albedo is not None:
            mse_albedo.append(NF.mse_loss(albedo, albedo_gt))
        if a_prime is not None:
            mse_a_prime.append(NF.mse_loss(a_prime, a_prime_gt))
        mse_kd.append(NF.mse_loss(kd, kd_gt))

    result = dict(row)
    result.update(
        {
            "result_type": "row",
            "num_frames": image_num,
            "kd": psnr_from_mse(mse_kd),
            "albedo": psnr_from_mse(mse_albedo),
            "a_prime": psnr_from_mse(mse_a_prime),
            "roughness": psnr_from_mse(mse_roughness),
            "emit_iou": mean_or_nan(iou_emission),
            "emit_log_mse": mean_or_nan(mse_emission),
        }
    )
    return result


def aggregate_results(results, group_values=None, result_type="aggregate_all"):
    if not results:
        raise ValueError("Cannot aggregate an empty result set.")

    aggregate = {
        "result_type": result_type,
        "manifest_row": "",
        "gt_path": "",
        "pred_path": "",
        "num_frames": sum(result["num_frames"] for result in results),
    }
    if group_values:
        aggregate.update(group_values)

    for metric_name in METRIC_COLUMNS:
        values = [
            result[metric_name]
            for result in results
            if not math.isnan(result[metric_name])
        ]
        aggregate[metric_name] = sum(values) / len(values) if values else float("nan")

    return aggregate


def group_results(results, group_by):
    grouped = defaultdict(list)
    for result in results:
        key = tuple(result.get(column, "") for column in group_by)
        grouped[key].append(result)

    aggregates = []
    for key, group in sorted(grouped.items()):
        group_values = dict(zip(group_by, key))
        aggregates.append(
            aggregate_results(
                group,
                group_values=group_values,
                result_type="aggregate_group",
            )
        )
    return aggregates


def format_cell(value, is_metric=False):
    if value is None:
        return ""
    if is_metric:
        return format_metric(value)
    return str(value)


def print_table(title, rows, columns):
    if not rows:
        return

    widths = []
    for column in columns:
        is_metric = column in METRIC_COLUMNS
        cell_values = [
            format_cell(row.get(column, ""), is_metric=is_metric) for row in rows
        ]
        widths.append(max(len(column), *(len(value) for value in cell_values)))

    print(title)
    print(
        "  " + " | ".join(column.ljust(width) for column, width in zip(columns, widths))
    )
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        values = [
            format_cell(
                row.get(column, ""), is_metric=(column in METRIC_COLUMNS)
            ).ljust(width)
            for column, width in zip(columns, widths)
        ]
        print("  " + " | ".join(values))
    print()


def write_csv(path, rows, field_order):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable_row = {}
            for field in field_order:
                value = row.get(field, "")
                if (
                    field in METRIC_COLUMNS
                    and isinstance(value, float)
                    and math.isnan(value)
                ):
                    serializable_row[field] = "nan"
                else:
                    serializable_row[field] = value
            writer.writerow(serializable_row)


def build_result_columns(fieldnames):
    metadata_columns = [
        column for column in fieldnames if column not in REQUIRED_MANIFEST_COLUMNS
    ]
    return ["manifest_row"] + metadata_columns + ["num_frames"] + METRIC_COLUMNS


def build_export_columns(fieldnames):
    return (
        ["result_type", "manifest_row"] + fieldnames + ["num_frames"] + METRIC_COLUMNS
    )


def main():
    args = parse_args()
    try:
        manifest_rows, fieldnames = read_manifest(args.manifest)
        validate_group_by_columns(args.group_by, fieldnames)
        manifest_rows = filter_manifest_rows(manifest_rows, args.split)

        results = [evaluate_row(row) for row in manifest_rows]

        result_columns = build_result_columns(fieldnames)
        print_table("Per-row metrics", results, result_columns)

        overall = aggregate_results(results)
        overall_columns = ["num_frames"] + METRIC_COLUMNS
        print_table("Overall aggregate", [overall], overall_columns)

        grouped_aggregates = []
        if args.group_by:
            grouped_aggregates = group_results(results, args.group_by)
            print_table(
                "Grouped aggregates",
                grouped_aggregates,
                args.group_by + ["num_frames"] + METRIC_COLUMNS,
            )

        if args.output_csv:
            export_columns = build_export_columns(fieldnames)
            export_rows = list(results)
            export_rows.append(overall)
            export_rows.extend(grouped_aggregates)
            write_csv(args.output_csv, export_rows, export_columns)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
