"""Expand only the locked training split with adjacent original-frame pairs.

The 179 training videos and all held-out image/label paths remain unchanged.
Each training video contributes 48 uniformly spaced (a, a + 1) pairs. This is
denser sampling of existing scenes, not an increase in independent scenes.
No GPU, model prediction, new split selection, or source-data writes occur.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

EXP = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = EXP / "data/vision/paper_v1"
DEFAULT_OUTPUT = EXP / "data/vision/stsd_expanded_v1"
SPLITS = ("train", "validation", "calibration", "test")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pair_indexes(frame_count: int, frames_per_video: int) -> tuple[list[int], list[int]]:
    if frames_per_video < 2 or frames_per_video % 2:
        raise ValueError("frames_per_video must be a positive even number >= 2")
    if frame_count < frames_per_video:
        raise ValueError(f"Source has only {frame_count} frames; needs {frames_per_video}")
    anchors = np.linspace(0, frame_count - 2, frames_per_video // 2, dtype=int).tolist()
    if any(b - a < 2 for a, b in zip(anchors, anchors[1:])):
        raise ValueError("Uniform pair anchors overlap")
    selected = [frame for anchor in anchors for frame in (anchor, anchor + 1)]
    if len(set(selected)) != frames_per_video:
        raise ValueError("Sampling did not create exactly the requested unique frames")
    return anchors, selected


def prepare(source: Path, output: Path, frames_per_video: int):
    cv2.setNumThreads(1)
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be a separate dataset directory, outside source")
    locked = read(source / "protocol.json")
    original_manifest = read(source / "split_manifest.json")
    source_protocol_sha = digest(source / "protocol.json")
    source_manifest_sha = digest(source / "split_manifest.json")
    if source_manifest_sha != locked["split_manifest_sha256"]:
        raise ValueError("Locked source split manifest changed")
    if (output / "protocol.json").exists():
        existing = read(output / "protocol.json")
        if (existing.get("source_protocol_sha256") == source_protocol_sha
                and existing.get("frames_per_video", {}).get("train") == frames_per_video
                and existing.get("split_manifest_sha256") == digest(output / "split_manifest.json")):
            print(json.dumps({"status": "already_prepared", "output": str(output), "counts": existing["counts"]}), flush=True)
            return
        raise FileExistsError("Refusing to overwrite a different or modified prepared output")

    assignments = locked["assignments"]
    train_sequences = sorted(name for name, assignment in assignments.items() if assignment["split"] == "train")
    old_audit = {record["sequence"]: record for record in original_manifest["video_audit"]}
    if len(train_sequences) != locked["counts"]["train"]["videos"]:
        raise ValueError("Source train assignment count mismatch")
    for name in train_sequences:
        pair_indexes(old_audit[name]["frames_original"], frames_per_video)

    output.mkdir(parents=True, exist_ok=True)
    for location in ("images/train", "labels/train", "lists", "weights"):
        (output / location).mkdir(parents=True, exist_ok=True)
    from vision_prepare import load_annotation_table

    # Preserve original evaluation records exactly, including source paths/hashes.
    records = [record for record in original_manifest["files"] if record["split"] != "train"]
    for record in records:
        for key, hash_key in (("image", "image_sha256"), ("label", "label_sha256")):
            if digest(Path(record[key])) != record[hash_key]:
                raise ValueError(f"Source held-out {key} changed: {record[key]}")
    video_audit = [record for record in original_manifest["video_audit"] if record["split"] != "train"]
    sampling_audit, written_bytes = [], 0
    for count, name in enumerate(train_sequences, 1):
        assignment = assignments[name]
        video = source / "raw" / f"{name}.mp4"
        annotation = source / "raw" / f"{name}_LABELS.mat"
        source_sha, annotation_sha = digest(video), digest(annotation)
        original = old_audit[name]
        if source_sha != original["source_sha256"] or annotation_sha != original["annotation_sha256"]:
            raise ValueError(f"Source video or annotation changed: {name}")
        table = load_annotation_table(annotation)
        capture = cv2.VideoCapture(str(video))
        frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or frames != original["frames_original"] or len(table["DRONE"]) != frames or fps <= 0:
            capture.release()
            raise ValueError(f"Bad video/MAT metadata: {name}")
        anchors, selected = pair_indexes(frames, frames_per_video)
        selected_set = set(selected)
        positive = 0
        try:
            for index in range(frames):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Source decode failed: {name}:{index}")
                if index not in selected_set:
                    continue
                rows = []
                for x, y, w, h in table["DRONE"][index]:
                    if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width + 1 and y + h <= height + 1):
                        raise ValueError(f"Invalid DRONE box: {name}:{index}")
                    x, y = max(0, x - 1), max(0, y - 1)
                    rows.append(f"0 {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}")
                positive += bool(rows)
                filename = f"{name}_{index:05d}"
                image_path = output / "images/train" / f"{filename}.jpg"
                label_path = output / "labels/train" / f"{filename}.txt"
                ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise ValueError(f"JPEG encode failed: {name}:{index}")
                image_payload = jpg.tobytes()
                label_payload = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
                # Existing partial outputs are accepted only when byte-identical.
                for path, payload in ((image_path, image_payload), (label_path, label_payload)):
                    if path.exists() and path.read_bytes() != payload:
                        raise FileExistsError(f"Refusing to replace nonidentical partial output: {path}")
                    if not path.exists():
                        path.write_bytes(payload)
                written_bytes += len(image_payload) + len(label_payload)
                records.append({"sequence": name, "split": "train", "category": assignment["category"],
                    "frame": index, "image": str(image_path), "label": str(label_path), "positive": bool(rows),
                    "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "image_sha256": hashlib.sha256(image_payload).hexdigest(),
                    "label_sha256": hashlib.sha256(label_payload).hexdigest(), "frame_time_seconds": index / fps})
        finally:
            capture.release()
        video_audit.append({**original, "sampled_frames": frames_per_video, "sampled_positive": positive})
        sampling_audit.append({"sequence": name, "frames_original": frames, "pairs": len(anchors),
            "pair_anchors": anchors, "sampled_frames": selected, "pair_original_frame_gap_min": 1,
            "pair_original_frame_gap_max": 1, "source_sha256": source_sha, "annotation_sha256": annotation_sha})
        if count % 10 == 0 or count == len(train_sequences):
            print(f"EXPAND {count}/{len(train_sequences)} videos; train_frames={count*frames_per_video}; bytes={written_bytes}", flush=True)

    records.sort(key=lambda record: (SPLITS.index(record["split"]), record["sequence"], record["frame"]))
    video_audit.sort(key=lambda record: record["sequence"])
    hash_groups, sequence_splits = defaultdict(list), defaultdict(set)
    for record in records:
        hash_groups[record["pixel_sha256"]].append(record)
        sequence_splits[record["sequence"]].add(record["split"])
    cross = [[{key: record[key] for key in ("sequence", "split", "frame")} for record in group]
             for group in hash_groups.values() if len({record["split"] for record in group}) > 1]
    cross_sequences = {name: sorted(splits) for name, splits in sequence_splits.items() if len(splits) > 1}
    duplicate_audit = {"cross_split_exact_pixel_duplicate_groups": cross, "count": len(cross),
        "cross_split_sequence_ids": cross_sequences,
        "pixel_hash_method": "SHA256 of decoded original BGR pixels; held-out hashes from locked manifest"}
    save(output / "exact_duplicate_audit.json", duplicate_audit)
    if cross or cross_sequences:
        raise ValueError("Cross-split sequence or exact decoded-pixel overlap found; output not locked")
    counts = {split: {"videos": sum(video["split"] == split for video in video_audit),
        "frames": sum(record["split"] == split for record in records),
        "positive_frames": sum(record["split"] == split and record["positive"] for record in records)} for split in SPLITS}
    for split in SPLITS[1:]:
        old_records = [record for record in original_manifest["files"] if record["split"] == split]
        new_records = [record for record in records if record["split"] == split]
        if sorted(old_records, key=lambda item: item["image"]) != sorted(new_records, key=lambda item: item["image"]):
            raise ValueError(f"Held-out records changed: {split}")
    for split in SPLITS:
        (output / "lists" / f"{split}.txt").write_text(
            "\n".join(Path(record["image"]).as_posix() for record in records if record["split"] == split) + "\n", encoding="utf-8")
    for filename, validation_split in (("train.yaml", "validation"), ("calibration.yaml", "calibration")):
        (output / filename).write_text(
            f"path: {output.as_posix()}\ntrain: lists/train.txt\nval: lists/{validation_split}.txt\n"
            "test: lists/test.txt\nnames:\n  0: drone\ntemporal_pair_stride: 2\ntemporal_max_frame_gap: 32\n", encoding="utf-8")
    # Metadata references public initialization in the source directory; no checkpoint copies.
    weight_manifest = read(source / "weights/manifest.json")
    save(output / "weights/manifest.json", {**weight_manifest,
        "reference_only": True, "source_manifest": str(source / "weights/manifest.json"),
        "source_manifest_sha256": digest(source / "weights/manifest.json")})
    save(output / "sampling_audit.json", {"train_frames_per_video": frames_per_video,
        "pairs_per_video": frames_per_video // 2, "train_pairs": len(train_sequences) * frames_per_video // 2,
        "pair_original_frame_gap_min": 1, "pair_original_frame_gap_max": 1,
        "all_train_images_used_once_per_epoch_with_pair_stride_2": True,
        "train_image_and_label_bytes": written_bytes, "videos": sampling_audit})
    save(output / "split_manifest.json", {"files": records, "video_audit": video_audit, "counts": counts})
    if digest(source / "protocol.json") != source_protocol_sha or digest(source / "split_manifest.json") != source_manifest_sha:
        raise ValueError("Source protocol/manifest changed while preparing")
    limitations = [item for item in locked.get("limitations", []) if "no test performance or visualization" not in item]
    limitations += ["Expansion increases sampling density of the same 179 training videos; independent scenes/video IDs do not increase",
        "Held-out video IDs, image paths, image bytes, label bytes, and records remain identical to paper_v1",
        "Existing held-out results were already inspected before this expansion; this is further development, not an untouched benchmark",
        "This is DroneDetectionThesis/Svanstrom RGB data, not official Anti-UAV",
        "Single-seed feasibility experiment; no statistical significance claim"]
    final = {**locked, "version": "vision_stsd_expanded_v1", "stage": "prepared_locked_before_expanded_training",
        "source_dataset": str(source), "source_protocol_sha256": source_protocol_sha,
        "source_split_manifest_sha256": source_manifest_sha,
        "frames_per_video": {**locked["frames_per_video"], "train": frames_per_video},
        "frame_sampling": f"train: {frames_per_video//2} uniform anchors in [0,N-2], each paired with original frame a+1; held-out splits unchanged",
        "independent_training_videos_added": 0, "sampling_density_multiplier": frames_per_video / locked["frames_per_video"]["train"],
        "counts": counts, "split_manifest_sha256": digest(output / "split_manifest.json"),
        "sampling_audit_sha256": digest(output / "sampling_audit.json"), "cross_split_exact_duplicate_groups": 0,
        "weights_manifest_sha256": digest(output / "weights/manifest.json"),
        "initialization_checkpoint": str(source / "weights/yolo11n.pt"),
        "source_checkpoint": str(source / "weights/yolo11n.pt"),
        "weights_manifest_path": str(source / "weights/manifest.json"),
        "source_raw_directory": str(source / "raw"),
        "train_recipe": {**locked["train_recipe"], "architectures": ["yolo11n-stsd", "yolo11n matched pair control"],
            "seeds": [0], "augmentation": "deterministic square letterbox; no random augmentation",
            "temporal_pair_stride": 2, "temporal_max_frame_gap": 32,
            "calibration": "fixed existing calibration images; no seed sweep",
            "test": "fixed existing held-out images/videos; development results, not untouched benchmark"},
        "limitations": limitations, "prepared_pid": os.getpid(), "prepared_utc": datetime.now(timezone.utc).isoformat()}
    save(output / "protocol.json", final)
    print(json.dumps({"status": final["stage"], "output": str(output), "counts": counts,
        "train_pairs": len(train_sequences) * frames_per_video // 2, "source_frame_pair_gap": 1,
        "train_image_and_label_bytes": written_bytes, "held_out_records_unchanged": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-video", type=int, default=96)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        prepare(args.source, args.output, args.frames_per_video)
    finally:
        print(f"PREPARE_ELAPSED_SECONDS {time.monotonic()-started:.2f}", flush=True)
