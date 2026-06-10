#!/usr/bin/env python3
"""Visualize IRIS camera transforms and scene geometry with viser."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation


IRIS_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])
SCANNETPP_JSON_TO_CONSUMED = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize train/val transforms.json camera poses and scene.obj. "
            "By default this matches SyntheticDatasetLDR: transform_matrix is "
            "consumed unchanged as IRIS camera axes (+X image-left, +Y image-up, "
            "+Z forward)."
        )
    )
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Dataset scene directory containing scene.obj and train/val/transforms.json.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="train",
        help="Which split transforms.json to visualize.",
    )
    parser.add_argument(
        "--consumer",
        choices=("synthetic", "scannetpp", "raw"),
        default="synthetic",
        help=(
            "Pose convention to mimic. synthetic/raw use transform_matrix unchanged; "
            "scannetpp flips JSON Y/Z columns like utils.dataset.scannetpp.dataset."
        ),
    )
    parser.add_argument(
        "--transforms",
        type=Path,
        action="append",
        default=None,
        help="Explicit transforms.json path. Can be passed multiple times; overrides --split.",
    )
    parser.add_argument(
        "--scene-obj",
        type=Path,
        default=None,
        help="Explicit scene mesh path. Defaults to scene_dir/scene.obj.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Draw every Nth camera. Use this for dense trajectories.",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=None,
        help="Frustum size. Defaults to 3%% of the scene/camera extent.",
    )
    parser.add_argument(
        "--axes-scale",
        type=float,
        default=None,
        help="Camera axes size. Defaults to 60%% of frustum scale.",
    )
    parser.add_argument("--mesh-opacity", type=float, default=0.35)
    parser.add_argument("--mesh-wireframe", action="store_true")
    parser.add_argument(
        "--max-faces",
        type=int,
        default=200_000,
        help="Simplify the displayed scene mesh to this face count. Set <=0 to disable.",
    )
    parser.add_argument(
        "--image-dir",
        default="Image",
        help="Image directory inside each split, used only to infer aspect if JSON lacks h/w.",
    )
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also draw raw JSON pose axes in gray when they differ from consumed poses.",
    )
    parser.add_argument(
        "--label-stride",
        type=int,
        default=10,
        help="Label every Nth drawn camera. Set 0 to disable labels.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def transforms_paths(args: argparse.Namespace) -> list[Path]:
    if args.transforms:
        return [path.resolve() for path in args.transforms]
    splits = ["train", "val"] if args.split == "both" else [args.split]
    return [(args.scene_dir / split / "transforms.json").resolve() for split in splits]


def as_c2w_4x4(transform_matrix: Iterable[Iterable[float]]) -> np.ndarray:
    mat = np.asarray(transform_matrix, dtype=np.float64)
    if mat.shape == (3, 4):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :4] = mat
        return c2w
    if mat.shape == (4, 4):
        return mat.copy()
    raise ValueError(f"Expected transform_matrix shape (3, 4) or (4, 4), got {mat.shape}")


def consumed_c2w(raw_c2w: np.ndarray, consumer: str) -> np.ndarray:
    c2w = raw_c2w.copy()
    if consumer == "scannetpp":
        c2w[:3, :3] = c2w[:3, :3] @ SCANNETPP_JSON_TO_CONSUMED
    return c2w


def frustum_rotation(c2w: np.ndarray, consumer: str) -> np.ndarray:
    rotation = c2w[:3, :3]
    if consumer in {"synthetic", "raw"}:
        # Viser frustums are OpenCV (+X right, +Y down, +Z forward), while
        # SyntheticDatasetLDR consumes IRIS axes (+X left, +Y up, +Z forward).
        return rotation @ IRIS_TO_OPENCV
    return rotation


def wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def infer_image_size(meta: dict, split_dir: Path, image_dir: str) -> tuple[int | None, int | None]:
    width = meta.get("w") or meta.get("width")
    height = meta.get("h") or meta.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    image_root = split_dir / image_dir
    if not image_root.exists():
        return None, None

    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.exr"):
        candidates.extend(sorted(image_root.glob(pattern)))
    if not candidates:
        return None, None

    image_path = candidates[0]
    try:
        if image_path.suffix.lower() == ".exr":
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None, None
            height, width = image.shape[:2]
            return int(width), int(height)

        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception as exc:
        print(f"[warn] Could not infer image size from {image_path}: {exc}")
        return None, None


def camera_fov_and_aspect(
    meta: dict,
    split_dir: Path,
    image_dir: str,
    image_width: int | None,
    image_height: int | None,
) -> tuple[float, float]:
    width = image_width
    height = image_height
    if width is None or height is None:
        inferred_width, inferred_height = infer_image_size(meta, split_dir, image_dir)
        width = width or inferred_width
        height = height or inferred_height

    aspect = float(width) / float(height) if width and height else 1.0

    if meta.get("fl_y") is not None and height:
        fov_y = 2.0 * math.atan(float(height) / (2.0 * float(meta["fl_y"])))
    elif meta.get("camera_angle_y") is not None:
        fov_y = float(meta["camera_angle_y"])
    elif meta.get("camera_angle_x") is not None:
        fov_x = float(meta["camera_angle_x"])
        fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)
    elif meta.get("fl_x") is not None and width:
        fov_x = 2.0 * math.atan(float(width) / (2.0 * float(meta["fl_x"])))
        fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)
    else:
        print("[warn] No camera FOV found; using 60 degrees.")
        fov_y = math.radians(60.0)

    return fov_y, aspect


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    return loaded


def simplify_mesh_for_display(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh

    print(f"[info] Simplifying mesh from {len(mesh.faces):,} to at most {max_faces:,} faces for viser.")
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception as exc:
        print(f"[warn] Quadric simplification failed ({exc}); using deterministic face sampling.")
        face_indices = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=np.int64)
        sampled = mesh.copy()
        sampled.update_faces(face_indices)
        sampled.remove_unreferenced_vertices()
        return sampled


def scene_extent(mesh: trimesh.Trimesh | None, camera_positions: np.ndarray) -> float:
    spans = []
    if mesh is not None and len(mesh.vertices):
        spans.append(float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])))
    if len(camera_positions) > 1:
        spans.append(float(np.linalg.norm(camera_positions.max(axis=0) - camera_positions.min(axis=0))))
    return max([span for span in spans if span > 0.0], default=1.0)


def split_name_from_path(path: Path) -> str:
    if path.name == "transforms.json":
        return path.parent.name
    return path.stem


def color_for_split(split_name: str) -> tuple[int, int, int]:
    if split_name == "train":
        return (30, 120, 255)
    if split_name == "val":
        return (255, 120, 20)
    return (160, 70, 220)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    scene_obj = (args.scene_obj or (args.scene_dir / "scene.obj")).resolve()
    transform_files = transforms_paths(args)
    for path in transform_files:
        if not path.exists():
            raise FileNotFoundError(path)
    if not scene_obj.exists():
        raise FileNotFoundError(scene_obj)

    split_payloads = []
    all_positions = []
    for transforms_path in transform_files:
        meta = load_json(transforms_path)
        raw_poses = [as_c2w_4x4(frame["transform_matrix"]) for frame in meta["frames"]]
        consumed_poses = [consumed_c2w(pose, args.consumer) for pose in raw_poses]
        positions = np.asarray([pose[:3, 3] for pose in consumed_poses], dtype=np.float64)
        all_positions.append(positions)
        split_payloads.append((transforms_path, meta, raw_poses, consumed_poses, positions))

    mesh = simplify_mesh_for_display(load_mesh(scene_obj), args.max_faces)

    camera_positions = np.concatenate(all_positions, axis=0) if all_positions else np.zeros((0, 3))
    extent = scene_extent(mesh, camera_positions)
    frustum_scale = args.frustum_scale or extent * 0.03
    axes_scale = args.axes_scale or frustum_scale * 0.6

    print("Camera convention:")
    if args.consumer in {"synthetic", "raw"}:
        print("  consumer=synthetic/raw: transform_matrix is consumed unchanged.")
        print("  Consumed camera axes: +X image-left, +Y image-up, +Z forward.")
        print("  Viser frustums are converted with diag(-1, -1, 1) for OpenCV display.")
    else:
        print("  consumer=scannetpp: JSON Y/Z columns are flipped before use.")
        print("  Consumed camera axes after conversion: OpenCV +X right, +Y down, +Z forward.")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/grid_xy",
        width=extent,
        height=extent,
        plane="xy",
        cell_size=max(extent / 20.0, 1e-3),
        section_size=max(extent / 5.0, 1e-3),
    )
    server.scene.add_mesh_simple(
        "/scene_obj",
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=(180, 180, 180),
        opacity=args.mesh_opacity,
        wireframe=args.mesh_wireframe,
        side="double",
    )

    for transforms_path, meta, raw_poses, poses, positions in split_payloads:
        split_name = split_name_from_path(transforms_path)
        color = color_for_split(split_name)
        fov_y, aspect = camera_fov_and_aspect(
            meta,
            transforms_path.parent,
            args.image_dir,
            args.image_width,
            args.image_height,
        )

        draw_indices = list(range(0, len(poses), args.stride))
        print(
            f"{split_name}: {len(poses)} poses from {transforms_path} "
            f"(drawing {len(draw_indices)}, fov_y={math.degrees(fov_y):.2f} deg, aspect={aspect:.4f})"
        )

        if len(positions) >= 2:
            server.scene.add_spline_catmull_rom(
                f"/{split_name}/camera_path",
                points=positions.astype(np.float32),
                line_width=2.0,
                color=color,
            )
        server.scene.add_point_cloud(
            f"/{split_name}/camera_centers",
            points=positions.astype(np.float32),
            colors=color,
            point_size=max(frustum_scale * 0.08, 1e-4),
        )

        for out_idx, pose_idx in enumerate(draw_indices):
            pose = poses[pose_idx]
            raw_pose = raw_poses[pose_idx]
            name = f"/{split_name}/cameras/{pose_idx:04d}"
            position = pose[:3, 3]

            server.scene.add_frame(
                f"{name}/consumed_axes",
                axes_length=axes_scale,
                axes_radius=max(axes_scale * 0.025, 1e-5),
                wxyz=wxyz_from_matrix(pose[:3, :3]),
                position=position,
            )
            server.scene.add_camera_frustum(
                f"{name}/frustum",
                fov=fov_y,
                aspect=aspect,
                scale=frustum_scale,
                line_width=1.5,
                color=color,
                wxyz=wxyz_from_matrix(frustum_rotation(pose, args.consumer)),
                position=position,
            )

            if args.show_raw and not np.allclose(raw_pose[:3, :4], pose[:3, :4]):
                server.scene.add_frame(
                    f"{name}/raw_json_axes",
                    axes_length=axes_scale * 0.75,
                    axes_radius=max(axes_scale * 0.018, 1e-5),
                    wxyz=wxyz_from_matrix(raw_pose[:3, :3]),
                    position=raw_pose[:3, 3],
                )

            if args.label_stride and out_idx % args.label_stride == 0:
                server.scene.add_label(
                    f"{name}/label",
                    text=f"{split_name}:{pose_idx}",
                    position=position + np.array([0.0, 0.0, axes_scale]),
                )

    print(f"Viser server running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
