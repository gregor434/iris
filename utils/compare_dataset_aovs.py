import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


AOV_SPECS = {
    "rgb": {
        "subdirs": ("inputs/hdr", "aovs/rgb", "rgb", "Image"),
        "kind": "rgb",
        "patterns": ("{frame:03d}_0001.exr",),
    },
    "kd": {
        "subdirs": ("aovs/kd", "kd", "DiffCol"),
        "kind": "rgb",
        "patterns": ("{frame:03d}_0001.exr",),
    },
    "albedo": {
        "subdirs": ("aovs/albedo", "albedo_pure"),
        "kind": "rgb",
        "patterns": ("{frame:03d}.exr",),
        "required": False,
    },
    "a_prime": {
        "subdirs": ("aovs/a_prime", "a_prime", "albedo"),
        "kind": "rgb",
        "patterns": ("{frame:03d}.exr",),
        "required": False,
    },
    "roughness": {
        "subdirs": ("aovs/roughness", "roughness", "Roughness"),
        "kind": "scalar",
        "patterns": ("{frame:03d}_0001.exr",),
    },
    "emission": {
        "subdirs": ("aovs/emission", "emission", "Emit"),
        "kind": "rgb",
        "patterns": ("{frame:03d}_0001.exr", "{frame:05d}_emission_nm.png"),
    },
}

FRAME_PATTERNS = (
    re.compile(r"^(?P<frame>\d{3})\.exr$"),
    re.compile(r"^(?P<frame>\d{3})_0001\.(?:exr|png)$"),
    re.compile(r"^(?P<frame>\d{5})_emission_nm\.png$"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GT AOVs between two synthetic dataset scene roots."
    )
    parser.add_argument("--dataset-a", required=True, help="First dataset scene root.")
    parser.add_argument("--dataset-b", required=True, help="Second dataset scene root.")
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to compare. Defaults to train.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        help="Optional explicit frame indices to compare.",
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Print per-frame metrics in addition to aggregate metrics.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for exporting per-frame metrics as CSV.",
    )
    return parser.parse_args()


def parse_frame_index(filename):
    for pattern in FRAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            return int(match.group("frame"))
    return None


def resolve_aov_dir(split_dir, spec, required=True):
    for subdir in spec["subdirs"]:
        candidate = split_dir / subdir
        if candidate.is_dir():
            return candidate
    if required:
        checked = "\n".join(str(split_dir / subdir) for subdir in spec["subdirs"])
        raise FileNotFoundError(f"Missing required AOV directory. Checked:\n{checked}")
    return None


def list_available_frames(split_dir, spec):
    aov_dir = resolve_aov_dir(split_dir, spec, required=spec.get("required", True))
    if aov_dir is None:
        return None

    frames = set()
    for path in aov_dir.iterdir():
        if path.name.startswith(".") or not path.is_file():
            continue
        frame_index = parse_frame_index(path.name)
        if frame_index is not None:
            frames.add(frame_index)
    if not frames:
        raise ValueError(f"No frames found in AOV directory: {aov_dir}")
    return frames


def resolve_aov_path(split_dir, spec, frame_index):
    aov_dir = resolve_aov_dir(split_dir, spec, required=spec.get("required", True))
    if aov_dir is None:
        return None
    for pattern in spec["patterns"]:
        candidate = aov_dir / pattern.format(frame=frame_index)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing frame {frame_index} for {spec['subdirs']} under {aov_dir}"
    )


def load_rgb_image(path):
    image = cv2.imread(str(path), -1)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = image[..., None].repeat(3, axis=-1)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 2D or 3-channel image at {path}, got shape {image.shape}")
    return image[..., [2, 1, 0]].astype(np.float32)


def load_scalar_image(path):
    image = cv2.imread(str(path), -1)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        return image[..., 0].astype(np.float32)
    raise ValueError(f"Expected 2D or 3-channel image at {path}, got shape {image.shape}")


def load_aov(path, kind):
    if kind == "rgb":
        return load_rgb_image(path)
    if kind == "scalar":
        return load_scalar_image(path)
    raise ValueError(f"Unsupported image kind: {kind}")


def mse(image_a, image_b):
    diff = image_a - image_b
    return float(np.mean(diff * diff))


def mae(image_a, image_b):
    return float(np.mean(np.abs(image_a - image_b)))


def psnr(mse_value, max_value):
    if mse_value <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((max_value * max_value) / mse_value))


def infer_max_value(image_a, image_b):
    peak = max(float(np.max(image_a)), float(np.max(image_b)))
    return 1.0 if peak <= 1.0 else peak


def format_metric(value):
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def gather_frame_indices(split_dir, requested_frames):
    per_aov_frames = {}
    for aov_name, spec in AOV_SPECS.items():
        frames = list_available_frames(split_dir, spec)
        if frames is not None:
            per_aov_frames[aov_name] = frames
    common_frames = set.intersection(*per_aov_frames.values())
    if not common_frames:
        raise ValueError(f"No common frames across required AOVs in {split_dir}")

    if requested_frames is not None and len(requested_frames) > 0:
        requested = set(requested_frames)
        missing = sorted(requested - common_frames)
        if missing:
            raise ValueError(
                f"Requested frames are missing from dataset split {split_dir}: {missing}"
            )
        return sorted(requested)

    return sorted(common_frames)


def compare_datasets(dataset_a, dataset_b, split, requested_frames):
    split_dir_a = Path(dataset_a) / split
    split_dir_b = Path(dataset_b) / split
    if not split_dir_a.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir_a}")
    if not split_dir_b.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir_b}")

    frames_a = set(gather_frame_indices(split_dir_a, requested_frames))
    frames_b = set(gather_frame_indices(split_dir_b, requested_frames))
    frame_indices = sorted(frames_a & frames_b)
    if requested_frames is not None and len(requested_frames) > 0:
        missing_from_b = sorted(set(requested_frames) - frames_b)
        if missing_from_b:
            raise ValueError(
                f"Requested frames are missing from dataset split {split_dir_b}: {missing_from_b}"
            )
    if not frame_indices:
        raise ValueError(
            f"No common frame indices found between {split_dir_a} and {split_dir_b}"
        )

    rows = []
    for frame_index in frame_indices:
        row = {"frame": frame_index}
        for aov_name, spec in AOV_SPECS.items():
            path_a = resolve_aov_path(split_dir_a, spec, frame_index)
            path_b = resolve_aov_path(split_dir_b, spec, frame_index)
            if path_a is None or path_b is None:
                continue

            image_a = load_aov(path_a, spec["kind"])
            image_b = load_aov(path_b, spec["kind"])
            if image_a.shape != image_b.shape:
                raise ValueError(
                    f"Shape mismatch for {aov_name} frame {frame_index}: "
                    f"{path_a} has shape {image_a.shape}, {path_b} has shape {image_b.shape}"
                )

            mse_value = mse(image_a, image_b)
            mae_value = mae(image_a, image_b)
            peak_value = infer_max_value(image_a, image_b)
            row[f"{aov_name}_mse"] = mse_value
            row[f"{aov_name}_mae"] = mae_value
            row[f"{aov_name}_psnr"] = psnr(mse_value, peak_value)
            row[f"{aov_name}_max_value"] = peak_value
        rows.append(row)
    return rows


def aggregate_rows(rows):
    aggregate = {"num_frames": len(rows)}
    for aov_name in AOV_SPECS:
        mse_key = f"{aov_name}_mse"
        mae_key = f"{aov_name}_mae"
        max_value_key = f"{aov_name}_max_value"
        if mse_key not in rows[0]:
            continue
        aggregate[mse_key] = float(np.mean([row[mse_key] for row in rows]))
        aggregate[mae_key] = float(np.mean([row[mae_key] for row in rows]))
        aggregate[max_value_key] = max(row[max_value_key] for row in rows)
        aggregate[f"{aov_name}_psnr"] = psnr(aggregate[mse_key], aggregate[max_value_key])
    return aggregate


def print_aggregate_table(aggregate):
    print(f"Matched frames: {aggregate['num_frames']}")
    print("")
    print("aov        mse         mae         psnr")
    print("--------   ---------   ---------   ---------")
    for aov_name in AOV_SPECS:
        mse_key = f"{aov_name}_mse"
        mae_key = f"{aov_name}_mae"
        psnr_key = f"{aov_name}_psnr"
        if mse_key not in aggregate:
            continue
        print(
            f"{aov_name:<8}   "
            f"{format_metric(aggregate[mse_key]):>9}   "
            f"{format_metric(aggregate[mae_key]):>9}   "
            f"{format_metric(aggregate[psnr_key]):>9}"
        )


def print_per_frame_rows(rows):
    print("")
    header = ["frame"]
    for aov_name in AOV_SPECS:
        if f"{aov_name}_mse" not in rows[0]:
            continue
        header.extend(
            [f"{aov_name}_mse", f"{aov_name}_mae", f"{aov_name}_psnr"]
        )
    print(",".join(header))
    for row in rows:
        values = [str(row["frame"])]
        for column in header[1:]:
            values.append(format_metric(row[column]))
        print(",".join(values))


def write_csv(rows, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["frame"]
    for aov_name in AOV_SPECS:
        if f"{aov_name}_mse" not in rows[0]:
            continue
        fieldnames.extend(
            [f"{aov_name}_mse", f"{aov_name}_mae", f"{aov_name}_psnr"]
        )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {"frame": row["frame"]}
            for fieldname in fieldnames[1:]:
                csv_row[fieldname] = row[fieldname]
            writer.writerow(csv_row)


def main():
    args = parse_args()
    rows = compare_datasets(
        dataset_a=args.dataset_a,
        dataset_b=args.dataset_b,
        split=args.split,
        requested_frames=args.frames,
    )
    aggregate = aggregate_rows(rows)
    print_aggregate_table(aggregate)
    if args.per_frame:
        print_per_frame_rows(rows)
    if args.output_csv:
        write_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
