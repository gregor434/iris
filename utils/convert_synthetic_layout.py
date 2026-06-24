import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


SCENE_FILES = (
    "scene.obj",
    "render_traj.npy",
    "train.npy",
    "val.npy",
    "poses.json",
    "idx2mat.pth",
    "test.xml",
    "test-relight.xml",
    "export_report.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert legacy FIPT synthetic scenes to the canonical IRIS layout."
    )
    parser.add_argument("--src", required=True, type=Path, help="Legacy scene root.")
    parser.add_argument("--dst", required=True, type=Path, help="Canonical output scene root.")
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to materialize files. Defaults to copy.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to convert. Defaults to train val.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing destination files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing files.",
    )
    return parser.parse_args()


class Converter:
    def __init__(self, src, dst, mode, force=False, dry_run=False):
        self.src = src
        self.dst = dst
        self.mode = mode
        self.force = force
        self.dry_run = dry_run
        self.operations = 0
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr)

    def ensure_scene_root(self):
        if not self.src.is_dir():
            raise FileNotFoundError(f"Missing source scene root: {self.src}")
        if not (self.src / "scene.obj").is_file():
            raise FileNotFoundError(f"Missing required scene mesh: {self.src / 'scene.obj'}")

    def materialize(self, src, dst):
        if not src.is_file():
            raise FileNotFoundError(f"Missing required source file: {src}")

        self.operations += 1
        if self.dry_run:
            print(f"{self.mode}: {src} -> {dst}")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if not self.force:
                raise FileExistsError(f"Destination already exists: {dst}")
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if self.mode == "copy":
            shutil.copy2(src, dst)
        elif self.mode == "symlink":
            os.symlink(src.resolve(), dst)
        elif self.mode == "hardlink":
            os.link(src, dst)
        else:
            raise ValueError(f"Unsupported materialization mode: {self.mode}")

    def copy_optional_file(self, relative_path):
        src = self.src / relative_path
        if src.is_file():
            self.materialize(src, self.dst / relative_path)

    def frame_indices(self, split):
        transforms_path = self.src / split / "transforms.json"
        if not transforms_path.is_file():
            raise FileNotFoundError(f"Missing required transforms.json: {transforms_path}")
        with transforms_path.open() as handle:
            meta = json.load(handle)
        frames = meta.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"Invalid transforms.json without frames list: {transforms_path}")
        return range(len(frames))

    def copy_frame_series(
        self,
        split,
        src_subdir,
        src_pattern,
        dst_subdir,
        dst_pattern=None,
        required=True,
        warn_extras=False,
    ):
        src_dir = self.src / split / src_subdir
        if not src_dir.is_dir():
            if required:
                raise FileNotFoundError(f"Missing required directory: {src_dir}")
            self.warn(f"Optional directory missing, omitted: {src_dir}")
            return

        dst_pattern = dst_pattern or src_pattern
        expected = set()
        for frame_index in self.frame_indices(split):
            src_name = src_pattern.format(frame=frame_index)
            expected.add(src_name)
            src = src_dir / src_name
            if not src.is_file():
                if required:
                    raise FileNotFoundError(f"Missing required frame file: {src}")
                self.warn(f"Optional frame missing, omitted: {src}")
                continue
            dst_name = dst_pattern.format(frame=frame_index)
            self.materialize(src, self.dst / split / dst_subdir / dst_name)

        if warn_extras:
            actual = {path.name for path in src_dir.iterdir() if path.is_file()}
            extras = sorted(actual - expected)
            if extras:
                preview = ", ".join(extras[:5])
                suffix = "" if len(extras) <= 5 else f", ... ({len(extras)} total)"
                self.warn(f"Ignored extra files in {src_dir}: {preview}{suffix}")

    def copy_split_file(self, split, src_name, dst_relative, required=False):
        src = self.src / split / src_name
        if src.is_file():
            self.materialize(src, self.dst / split / dst_relative)
        elif required:
            raise FileNotFoundError(f"Missing required split file: {src}")
        else:
            self.warn(f"Optional split file missing, omitted: {src}")

    def hash_file(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def resolve_prior_dir(self, split):
        image_prior = self.src / split / "Image" / "albedo"
        irisformer_prior = self.src / split / "irisformer" / "albedo"
        if image_prior.is_dir() and irisformer_prior.is_dir():
            image_names = {path.name for path in image_prior.iterdir() if path.is_file()}
            irisformer_names = {path.name for path in irisformer_prior.iterdir() if path.is_file()}
            if image_names != irisformer_names:
                self.warn(
                    f"Prior filename sets differ for {split}; using {image_prior}"
                )
            else:
                differing = []
                for name in sorted(image_names):
                    if self.hash_file(image_prior / name) != self.hash_file(irisformer_prior / name):
                        differing.append(name)
                        if len(differing) >= 3:
                            break
                if differing:
                    self.warn(
                        f"Prior files differ for {split}; using {image_prior}. "
                        f"Examples: {', '.join(differing)}"
                    )
            return "Image/albedo"
        if image_prior.is_dir():
            return "Image/albedo"
        if irisformer_prior.is_dir():
            return "irisformer/albedo"
        self.warn(f"No albedo prior directory found for split {split}; priors omitted")
        return None

    def convert_scene_files(self):
        for filename in SCENE_FILES:
            self.copy_optional_file(filename)

    def convert_split(self, split):
        split_root = self.src / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing requested split directory: {split_root}")

        self.copy_split_file(split, "transforms.json", "cameras/transforms.json", required=True)
        self.copy_split_file(split, "Image/cam/exposure.npy", "cameras/exposure.npy")
        self.copy_split_file(split, "Image/cam/crf.npy", "cameras/crf.npy")
        self.copy_split_file(split, "metallic.npy", "metallic.npy")
        self.copy_split_file(split, "cam.txt", "cam.txt")

        self.copy_frame_series(split, "Image", "{frame:03d}_0001.png", "inputs/ldr")
        self.copy_frame_series(split, "Image", "{frame:03d}_0001.exr", "inputs/hdr")
        self.copy_frame_series(split, "DiffCol", "{frame:03d}_0001.exr", "aovs/kd")
        self.copy_frame_series(split, "DiffCol", "{frame:03d}_0001.exr", "aovs/albedo")
        self.copy_frame_series(
            split,
            "albedo",
            "{frame:03d}.exr",
            "aovs/a_prime",
            required=False,
            warn_extras=True,
        )
        self.copy_frame_series(split, "Roughness", "{frame:03d}_0001.exr", "aovs/roughness")
        self.copy_frame_series(split, "Emit", "{frame:03d}_0001.exr", "aovs/emission")
        self.copy_frame_series(split, "IndexMA", "{frame:03d}_0001.exr", "labels/material_id")
        self.copy_frame_series(
            split,
            "segmentation",
            "{frame:03d}.exr",
            "labels/segmentation",
            required=False,
        )

        prior_dir = self.resolve_prior_dir(split)
        if prior_dir is not None:
            self.copy_frame_series(
                split,
                prior_dir,
                "{frame:03d}_0001.png",
                "priors/albedo",
                required=False,
            )

    def convert_relight(self):
        relight_dir = self.src / "val_relight"
        if not relight_dir.is_dir():
            return
        for src in sorted(relight_dir.glob("*_0001.png")):
            self.materialize(src, self.dst / "relight" / "inputs" / "ldr" / src.name)
        for src in sorted(relight_dir.glob("*_0001.exr")):
            self.materialize(src, self.dst / "relight" / "inputs" / "hdr" / src.name)

    def convert(self, splits):
        self.ensure_scene_root()
        self.convert_scene_files()
        for split in splits:
            self.convert_split(split)
        self.convert_relight()
        print(f"Planned operations: {self.operations}" if self.dry_run else f"Completed operations: {self.operations}")
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}", file=sys.stderr)


def main():
    args = parse_args()
    converter = Converter(
        src=args.src,
        dst=args.dst,
        mode=args.mode,
        force=args.force,
        dry_run=args.dry_run,
    )
    converter.convert(args.splits)


if __name__ == "__main__":
    main()
