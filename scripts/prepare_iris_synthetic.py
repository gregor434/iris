#!/usr/bin/env python3
"""Prepare generated synthetic renders for the IRIS FIPT synthetic loader."""

import argparse
import errno
import json
import math
import os
import random
import shutil
import struct
from pathlib import Path


DEFAULT_SOURCE = Path("data/indoor_synthetic/bathroom_mi/ambient_blender_run_synthetic")
DEFAULT_OUTPUT = Path("data/indoor_synthetic/bathroom_mi/iris_synthetic")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restructure generated Blender AOV renders into IRIS synthetic train/val splits."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Generated render directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output IRIS scene directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of frames assigned to val.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic splitting.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--link", action="store_true", help="Hardlink files instead of copying them.")
    parser.add_argument("--no-shuffle", action="store_true", help="Use the last val-ratio frames for val.")
    parser.add_argument(
        "--init-albedo-source",
        choices=["albedo_ldr", "xrgb", "xrgb_exr"],
        default="albedo_ldr",
        help="Source for initialization albedo PNGs copied to Image/albedo and irisformer/albedo.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def source_file(source, rel_path, suffix=None):
    rel = rel_path
    if suffix and not rel.endswith(suffix):
        rel = rel + suffix
    return source / rel


def copy_or_link(src, dst, link=False):
    if not src.exists():
        raise FileNotFoundError(f"missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    else:
        pass
    shutil.copy2(src, dst)


def split_indices(num_frames, val_ratio, seed, shuffle=True):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be >= 0 and < 1")
    indices = list(range(num_frames))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    val_count = int(round(num_frames * val_ratio))
    if val_ratio > 0 and num_frames > 1:
        val_count = min(max(val_count, 1), num_frames - 1)
    val_set = set(indices[-val_count:]) if val_count else set()
    train = [i for i in range(num_frames) if i not in val_set]
    val = [i for i in range(num_frames) if i in val_set]
    return train, val


def frame_old_id(frame):
    file_path = frame.get("file_path") or frame.get("file_path_blender")
    if not file_path:
        raise KeyError("frame is missing file_path/file_path_blender")
    return Path(file_path).name


def init_albedo_file(source, old_id, init_albedo_source):
    if init_albedo_source == "albedo_ldr":
        return source / "albedo_ldr" / f"{old_id}.png"
    return source / init_albedo_source / old_id / "albedo.png"


def validate_source(source, frames, init_albedo_source):
    required_keys = [
        ("file_path", ".png"),
        ("hdr_path", None),
        ("albedo_path", None),
        ("roughness_path", None),
        ("emission_path", None),
        ("material_index_path", None),
    ]
    missing = []
    for idx, frame in enumerate(frames):
        for key, suffix in required_keys:
            rel = frame.get(key)
            if rel is None:
                missing.append((idx, key, "<missing key>"))
                continue
            path = source_file(source, rel, suffix)
            if not path.exists():
                missing.append((idx, key, str(path)))
        old_id = frame_old_id(frame)
        init_albedo = init_albedo_file(source, old_id, init_albedo_source)
        if not init_albedo.exists():
            missing.append((idx, init_albedo_source, str(init_albedo)))
    if missing:
        preview = "\n".join(f"frame {i}: {key} -> {path}" for i, key, path in missing[:20])
        more = "" if len(missing) <= 20 else f"\n... {len(missing) - 20} more missing entries"
        raise FileNotFoundError(f"source validation failed:\n{preview}{more}")


def metadata_tonemap(source):
    metadata_path = source / "metadata.json"
    if not metadata_path.exists():
        return 1.0, 2.2
    metadata = load_json(metadata_path)
    blender_tonemap = metadata.get("tonemap", {}).get("blender", {})
    exposure = float(blender_tonemap.get("exposure", 1.0))
    gamma = float(blender_tonemap.get("gamma", 2.2))
    return exposure, gamma


def npy_header(shape):
    shape_text = "(" + ", ".join(str(v) for v in shape)
    if len(shape) == 1:
        shape_text += ","
    shape_text += ")"
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_text + ", }"
    header_len = len(header) + 1
    padding = (16 - ((10 + header_len) % 16)) % 16
    return (header + " " * padding + "\n").encode("latin1")


def write_npy_float32(path, shape, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = math.prod(shape)
    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} values, got {len(values)}")
    header = npy_header(shape)
    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header)))
        f.write(header)
        f.write(struct.pack("<" + "f" * len(values), *values))


def write_camera_metadata(split_dir, frame_count, exposure, gamma):
    cam_dir = split_dir / "Image" / "cam"
    write_npy_float32(cam_dir / "exposure.npy", (frame_count,), [exposure] * frame_count)
    xs = [i / 1023.0 for i in range(1024)]
    curve = [x ** (1.0 / gamma) for x in xs]
    write_npy_float32(cam_dir / "crf.npy", (3, 1024), curve * 3)


def write_split(
    source,
    output,
    split_name,
    indices,
    frames,
    camera_angle_x,
    exposure,
    gamma,
    init_albedo_source,
    link=False,
):
    split_dir = output / split_name
    split_frames = []

    for new_idx, old_idx in enumerate(indices):
        frame = frames[old_idx]
        new_stem = f"{new_idx:03d}_0001"
        old_id = frame_old_id(frame)

        copy_or_link(source_file(source, frame["file_path"], ".png"), split_dir / "Image" / f"{new_stem}.png", link)
        copy_or_link(source_file(source, frame["hdr_path"]), split_dir / "Image" / f"{new_stem}.exr", link)
        copy_or_link(source_file(source, frame["albedo_path"]), split_dir / "DiffCol" / f"{new_stem}.exr", link)
        copy_or_link(source_file(source, frame["roughness_path"]), split_dir / "Roughness" / f"{new_stem}.exr", link)
        copy_or_link(source_file(source, frame["emission_path"]), split_dir / "Emit" / f"{new_stem}.exr", link)
        copy_or_link(source_file(source, frame["material_index_path"]), split_dir / "IndexMA" / f"{new_stem}.exr", link)

        init_albedo = init_albedo_file(source, old_id, init_albedo_source)
        copy_or_link(init_albedo, split_dir / "Image" / "albedo" / f"{new_stem}.png", link)
        copy_or_link(init_albedo, split_dir / "irisformer" / "albedo" / f"{new_stem}.png", link)
        if split_name == "train":
            copy_or_link(source_file(source, frame["albedo_path"]), split_dir / "albedo" / f"{new_idx:03d}.exr", link)

        split_frames.append(
            {
                "file_path": f"{split_name}/Image/{new_stem}",
                "transform_matrix": frame["transform_matrix"],
            }
        )

    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": split_frames,
    }
    with (split_dir / "transforms.json").open("w") as f:
        json.dump(transforms, f, indent=4)
        f.write("\n")
    write_camera_metadata(split_dir, len(indices), exposure, gamma)


def parse_ply_header(path):
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        data_start = f.tell()

    if header_lines[0] != "ply":
        raise ValueError(f"{path}: not a PLY file")
    if "format binary_little_endian 1.0" not in header_lines:
        raise ValueError(f"{path}: only binary_little_endian PLY is supported")

    elements = []
    current = None
    for line in header_lines[1:]:
        if line.startswith("element "):
            _, name, count = line.split()
            current = {"name": name, "count": int(count), "properties": []}
            elements.append(current)
        elif line.startswith("property ") and current is not None:
            parts = line.split()
            current["properties"].append(parts[1:])
    return elements, data_start


def scalar_format(ply_type):
    formats = {
        "char": "b",
        "uchar": "B",
        "short": "h",
        "ushort": "H",
        "int": "i",
        "uint": "I",
        "float": "f",
        "double": "d",
    }
    if ply_type not in formats:
        raise ValueError(f"unsupported PLY scalar type: {ply_type}")
    return formats[ply_type]


def convert_ply_to_obj(ply_path, obj_path):
    elements, data_start = parse_ply_header(ply_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    with ply_path.open("rb") as src, obj_path.open("w") as dst:
        src.seek(data_start)
        for element in elements:
            name = element["name"]
            count = element["count"]
            props = element["properties"]

            if name == "vertex":
                scalar_props = [p for p in props if p[0] != "list"]
                fmt = "<" + "".join(scalar_format(p[0]) for p in scalar_props)
                size = struct.calcsize(fmt)
                names = [p[1] for p in scalar_props]
                ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
                for _ in range(count):
                    values = struct.unpack(fmt, src.read(size))
                    dst.write(f"v {values[ix]} {values[iy]} {values[iz]}\n")
            elif name == "face":
                if len(props) != 1 or props[0][0] != "list":
                    raise ValueError(f"{ply_path}: unsupported face properties")
                _, count_type, index_type, _ = props[0]
                count_fmt = scalar_format(count_type)
                index_fmt = scalar_format(index_type)
                count_size = struct.calcsize("<" + count_fmt)
                index_size = struct.calcsize("<" + index_fmt)
                for _ in range(count):
                    n = struct.unpack("<" + count_fmt, src.read(count_size))[0]
                    raw = src.read(index_size * n)
                    indices = struct.unpack("<" + index_fmt * n, raw)
                    obj_indices = [str(i + 1) for i in indices]
                    dst.write("f " + " ".join(obj_indices) + "\n")
            else:
                skip_element(src, element)


def skip_element(src, element):
    for _ in range(element["count"]):
        for prop in element["properties"]:
            if prop[0] == "list":
                _, count_type, item_type, _ = prop
                count_size = struct.calcsize("<" + scalar_format(count_type))
                n = struct.unpack("<" + scalar_format(count_type), src.read(count_size))[0]
                src.seek(struct.calcsize("<" + scalar_format(item_type)) * n, os.SEEK_CUR)
            else:
                src.seek(struct.calcsize("<" + scalar_format(prop[0])), os.SEEK_CUR)


def prepare_output(output, overwrite):
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output} (use --overwrite to replace it)")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def main():
    args = parse_args()
    source = args.source
    output = args.output

    if not source.exists():
        raise FileNotFoundError(f"source directory not found: {source}")
    transforms_path = source / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"missing transforms.json: {transforms_path}")

    meta = load_json(transforms_path)
    frames = meta.get("frames")
    if not frames:
        raise ValueError(f"{transforms_path}: no frames found")
    camera_angle_x = meta.get("camera_angle_x")
    if camera_angle_x is None:
        raise ValueError(f"{transforms_path}: missing camera_angle_x")

    validate_source(source, frames, args.init_albedo_source)
    train_indices, val_indices = split_indices(
        len(frames), args.val_ratio, args.seed, shuffle=not args.no_shuffle
    )
    exposure, gamma = metadata_tonemap(source)

    prepare_output(output, args.overwrite)
    write_split(
        source,
        output,
        "train",
        train_indices,
        frames,
        camera_angle_x,
        exposure,
        gamma,
        args.init_albedo_source,
        args.link,
    )
    write_split(
        source,
        output,
        "val",
        val_indices,
        frames,
        camera_angle_x,
        exposure,
        gamma,
        args.init_albedo_source,
        args.link,
    )

    ply_path = source / "scene.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"missing source mesh: {ply_path}")
    convert_ply_to_obj(ply_path, output / "scene.obj")

    print(f"Wrote IRIS synthetic scene: {output}")
    print(f"train frames: {len(train_indices)}")
    print(f"val frames: {len(val_indices)}")
    print(f"init albedo source: {args.init_albedo_source}")
    print(f"exposure: {exposure}")
    print(f"gamma CRF: {gamma}")


if __name__ == "__main__":
    main()
