"""Bounded download and audit of a real drone-body video smoke subset.

No synthetic data. This subset is an engineering smoke test, not a benchmark.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = ROOT / "data" / "vision"
REPO = "DroneDetectionThesis/Drone-detection-dataset"
REVISION = "442d0c6750708cc221636f7a1fd31cbd8cb0aeb8"
BASE = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"
DATA_BUDGET = 500 * 1024 * 1024


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch(url, destination, expected=None, limit=32 * 1024 * 1024, headers=None):
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected is None or destination.stat().st_size == expected):
        payload = destination.read_bytes()
        return {"url": url, "path": str(destination.relative_to(ROOT)),
                "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "cached": True}
    errors = []
    for attempt in range(2):
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 25)) as response:
                response.raise_for_status()
                declared = int(response.headers.get("Content-Length", 0))
                if declared > limit:
                    raise ValueError(f"Declared download {declared} exceeds {limit}")
                digest, count = hashlib.sha256(), 0
                with partial.open("wb") as stream:
                    for block in response.iter_content(128 * 1024):
                        count += len(block)
                        if count > limit:
                            raise ValueError("Download exceeds per-file budget")
                        stream.write(block)
                        digest.update(block)
            if expected is not None and count != expected:
                raise ValueError(f"Size mismatch {count} != {expected}")
            partial.replace(destination)
            print(f"downloaded {destination.name}: {count} bytes", flush=True)
            return {"url": url, "path": str(destination.relative_to(ROOT)),
                    "bytes": count, "sha256": digest.hexdigest(), "cached": False}
        except Exception as error:
            errors.append(f"attempt {attempt + 1}: {type(error).__name__}: {error}")
            if partial.exists():
                partial.unlink()
    raise RuntimeError("; ".join(errors))


def download():
    import requests

    DATA.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1", timeout=25)
    response.raise_for_status()
    tree = response.json()
    save_json(DATA / "source_tree.json", tree)
    by_path = {entry["path"]: entry for entry in tree["tree"]}
    selected = []
    # Deterministically span the available video IDs; do not cherry-pick accuracy.
    for category, count in [("DRONE", 10), ("BIRD", 6), ("AIRPLANE", 4)]:
        paths = sorted(p for p in by_path if p.startswith(f"Data/Video_V/V_{category}_")
                       and p.endswith(".mp4"))
        indexes = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
        for i, index in enumerate(indexes):
            selected.append({"path": paths[index], "category": category,
                             "split": "val" if i % 3 == 2 else "train"})
    jobs = [(BASE + "README.md", DATA / "source_README.md", by_path["README.md"]["size"])]
    for sequence in selected:
        for p in [sequence["path"], sequence["path"].replace(".mp4", "_LABELS.mat")]:
            jobs.append((BASE + p, DATA / "raw" / Path(p).name, by_path[p]["size"]))
    planned = sum(job[2] for job in jobs)
    if planned > DATA_BUDGET:
        raise ValueError(f"Planned data {planned} exceeds {DATA_BUDGET}")
    manifest = {"repository": f"https://github.com/{REPO}", "revision": REVISION,
                "purpose": "real RGB drone-body engineering smoke subset",
                "data_budget_bytes": DATA_BUDGET, "planned_bytes": planned,
                "sequence_split": selected, "files": [], "failures": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, *job): job[0] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                manifest["files"].append(future.result())
            except Exception as error:
                manifest["failures"].append({"url": futures[future], "error": str(error)})
            save_json(DATA / "download_manifest.json", manifest)
    manifest["downloaded_bytes"] = sum(item["bytes"] for item in manifest["files"])
    save_json(DATA / "download_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["planned_bytes", "downloaded_bytes", "failures"]},
                     ensure_ascii=False), flush=True)
    if manifest["failures"]:
        raise RuntimeError("Downloads incomplete; see download_manifest.json")


def load_annotation_table(path):
    """Read the named label table, excluding DataSource metadata.

    The generic mcos-decoder 0.1.0 RGB extractor incorrectly emits a leading
    metadata None, and also misses cells with multiple bird boxes. Use its
    MAT element parser but select the actual struct with named class fields.
    """
    import scipy.io
    from mcos_decoder import MatStream

    fw = scipy.io.loadmat(path)["__function_workspace__"].tobytes()
    candidates = []

    def visit(element):
        children = element.children
        if element.type_id == 14 and len(children) >= 5 and children[3].type_id == 5:
            field_width = children[3].as_array()
            names = children[4].payload
            if b"DRONE\x00" in names and b"Time\x00" in names and len(field_width) == 1:
                width = int(field_width[0])
                fields = [names[i:i + width].rstrip(b"\x00").decode("ascii")
                          for i in range(0, len(names), width)]
                shape = children[1].as_array()
                rows = int(shape[0])
                cells = [c for c in children[5:] if c.type_id == 14]
                if len(cells) != rows * len(fields):
                    raise ValueError("Annotation struct field count mismatch")
                candidates.append((fields, rows, cells))
        for child in children:
            visit(child)

    for top in MatStream(fw, skip_header=8).walk():
        visit(top)
    if len(candidates) != 1:
        raise ValueError(f"Expected one named annotation table, found {len(candidates)}")
    fields, rows, cells = candidates[0]

    def boxes(cell):
        values = []
        def find(element):
            if element.type_id == 9:
                arr = element.as_array()
                if arr is not None and len(arr) > 0 and len(arr) % 4 == 0:
                    # MATLAB bbox matrices use column-major order [x,y,w,h].
                    values.extend(arr.reshape((-1, 4), order="F").tolist())
            for child in element.children:
                find(child)
        find(cell)
        return [tuple(map(float, value)) for value in values]

    return {field: [boxes(cells[row * len(fields) + col]) for row in range(rows)]
            for col, field in enumerate(fields) if field != "Time"}


def prepare(frames_per_video=24):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    manifest = json.loads((DATA / "download_manifest.json").read_text(encoding="utf-8"))
    audit, examples, box_sizes = [], [], []
    for sequence in manifest["sequence_split"]:
        name = Path(sequence["path"]).stem
        annotations = load_annotation_table(DATA / "raw" / f"{name}_LABELS.mat")
        bboxes = annotations["DRONE"]
        capture = cv2.VideoCapture(str(DATA / "raw" / f"{name}.mp4"))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(3)), int(capture.get(4))
        if total != len(bboxes):
            capture.release()
            raise ValueError(f"{name}: {total} video frames != {len(bboxes)} decoded labels")
        indexes = set(np.linspace(0, total - 1, min(frames_per_video, total), dtype=int).tolist())
        split = sequence["split"]
        image_dir, label_dir = DATA / "images" / split, DATA / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        sampled, positives, absent, invalid = 0, 0, 0, 0
        for index in range(total):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Video decode failed {name}:{index}")
            if index not in indexes:
                continue
            filename = f"{name}_{index:05d}"
            target_boxes = bboxes[index]
            label = ""
            # Read the actual DRONE field even in bird/airplane sequences.
            valid_boxes = []
            for x, y, w, h in target_boxes:
                if not (w > 0 and h > 0 and x >= 0 and y >= 0
                        and x + w <= width + 1 and y + h <= height + 1):
                    invalid += 1
                    continue
                # MATLAB image coordinates are 1-based; subtract one pixel at origin.
                x, y = max(0, x - 1), max(0, y - 1)
                label += f"0 {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}\n"
                valid_boxes.append((x, y, w, h))
                box_sizes.append({"sequence": name, "split": split, "width_px": w,
                                  "height_px": h, "min_side_px": min(w, h)})
            if target_boxes and not valid_boxes:
                continue  # Do not turn a frame with invalid drone labels into a negative.
            if valid_boxes:
                positives += 1
            elif sequence["category"] == "DRONE":
                absent += 1
            # cv2.imwrite can fail on Chinese paths on Windows; encode then save with pathlib.
            encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not encoded:
                raise ValueError("JPEG encoding failed")
            (image_dir / f"{filename}.jpg").write_bytes(payload.tobytes())
            (label_dir / f"{filename}.txt").write_text(label, encoding="utf-8")
            sampled += 1
            if len([e for e in examples if e[0] == name]) < 2:
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(rgb)
                for x, y, w, h in annotations[sequence["category"]][index]:
                    draw.rectangle([x - 1, y - 1, x + w - 1, y + h - 1], outline="red", width=2)
                rgb.thumbnail((360, 210))
                examples.append((name, rgb, f"{split} | {sequence['category']} | frame={index}"))
        capture.release()
        audit.append({**sequence, "video_frames": total, "label_frames": len(bboxes),
                      "resolution": [width, height], "sampled": sampled,
                      "drone_positive_frames": positives, "drone_absent_frames": absent,
                      "invalid_drone_boxes_skipped": invalid,
                      "decoder": "named class-field MAT table parser, not generic metadata scan"})
        print(f"prepared {name}: {sampled} frames, {positives} drone positives", flush=True)
    # Relative paths make dataset configuration portable; script sets its cwd itself.
    (DATA / "drone_smoke.yaml").write_text(
        "path: " + DATA.as_posix() + "\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n",
        encoding="utf-8")
    summary = {"purpose": "engineering feasibility only; not formal benchmark accuracy",
               "split_unit": "entire video, not random neighboring frames",
               "original_source_categories": ["DRONE", "BIRD", "AIRPLANE"],
               "target": "drone body, one class; birds/airplanes used as negatives",
               "sequences": audit, "sampled_frames": sum(a["sampled"] for a in audit),
               "positive_frames": sum(a["drone_positive_frames"] for a in audit),
               "negative_frames": sum(a["sampled"] - a["drone_positive_frames"] for a in audit),
               "box_sizes": box_sizes,
               "caveats": ["small selected subset; sequence split may still share capture sessions",
                           "RGB MATLAB decoding checked by frame counts, bounds and visual montage",
                           "no foundation teacher evaluated; no innovation claimed"]}
    if box_sizes:
        summary["box_min_side_percentiles"] = np.percentile(
            [b["min_side_px"] for b in box_sizes], [0, 25, 50, 75, 100]).tolist()
    save_json(DATA / "audit.json", summary)
    chosen = [e for i, e in enumerate(examples) if i % 2 == 0]
    montage = Image.new("RGB", (1080, 250 * ((len(chosen) + 2) // 3)), "white")
    draw = ImageDraw.Draw(montage)
    for i, (name, rgb, description) in enumerate(chosen):
        x, y = (i % 3) * 360, (i // 3) * 250
        montage.paste(rgb, (x, y + 35))
        draw.text((x + 3, y + 2), name, fill="black")
        draw.text((x + 3, y + 16), description, fill="black")
    montage.save(DATA / "annotation_montage.jpg")
    print(json.dumps({k: summary[k] for k in ["sampled_frames", "positive_frames", "negative_frames",
                                               "box_min_side_percentiles"]}), flush=True)


def weights():
    # Official GitHub release API avoids the github.com timeout seen locally.
    result = fetch("https://api.github.com/repos/ultralytics/assets/releases/assets/196201276?download=1",
                   DATA / "weights" / "yolo11n.pt", expected=5613764,
                   limit=1024 * 1024 * 1024, headers={"Accept": "application/octet-stream"})
    save_json(DATA / "weights" / "manifest.json", {"purpose": "COCO initialization only; not a pretrained drone detector",
                                                   "files": [result]})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "prepare", "weights"])
    parser.add_argument("--frames-per-video", type=int, default=24)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        {"download": download, "prepare": lambda: prepare(args.frames_per_video), "weights": weights}[args.action]()
    except Exception as error:
        save_json(DATA / f"{args.action}_failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        print(f"elapsed_seconds={time.monotonic() - started:.2f}", flush=True)
