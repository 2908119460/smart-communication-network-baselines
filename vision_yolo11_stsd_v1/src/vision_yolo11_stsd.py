"""YOLO11-STSD: isolated fork, checks, short paired training and video demo.

Run with experiments/.venv/Scripts/python.exe; installed baseline is not edited.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXP = Path(__file__).resolve().parents[1]
FORK = EXP / "sources" / "ultralytics_stsd"
RUN = EXP / "runs" / "vision_yolo11_stsd_v1"
DATA = EXP / "data" / "vision" / "paper_v1"
YAML = FORK / "ultralytics" / "cfg" / "models" / "11" / "yolo11n-stsd.yaml"
BASELINE = EXP / "runs" / "vision_paper_v1_yolo11n_seed0_attempt2" / "weights" / "best.pt"
sys.path.insert(0, str(FORK))
os.environ["PYTHONPATH"] = str(FORK) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["YOLO_CONFIG_DIR"] = str(RUN / "settings")
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules import SkyPriorModule, DynamicScaleNeck
from stsd_tools import STSDPairTrainer, load_baseline_weights, temporal_modules, set_video_mode

torch.set_num_threads(2)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def update_summary(key, value):
    path = RUN / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    summary.update({"date": "2026-10-14", "purpose": "implementation and engineering feasibility; not a novelty benchmark",
                    "limitations": ["Sky maps are attention, not supervised sky probability.",
                                     "Dynamic fusion augments the original FPN/PAN; it does not replace it.",
                                     "Ordinary shuffled training does not learn the temporal branch.",
                                     "Pairs are adjacent sampled frames, not necessarily consecutive original frames.",
                                     "Existing data is DroneDetectionThesis, not the official Anti-UAV benchmark.",
                                     "Short probe metrics cannot establish improvement over the completed baseline."]})
    summary[key] = value
    summary["status"] = "completed" if summary.get("checks", {}).get("passed") and summary.get("probe", {}).get("status") == "completed" else "in_progress"
    write_json(path, summary)


def check(args):
    result = subprocess.run([sys.executable, str(FORK / "tests" / "test_stsd.py")],
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "checks.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr)
    original = EXP / ".venv" / "Lib" / "site-packages" / "ultralytics"
    if not original.is_dir():
        from importlib.metadata import distribution
        original = Path(distribution("ultralytics").locate_file("ultralytics"))
    baseline_files = {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [original / "nn" / "tasks.py", original / "nn" / "modules" / "__init__.py",
                                original / "cfg" / "models" / "11" / "yolo11.yaml"] if p.is_file()}
    plain_tasks = (original / "nn" / "tasks.py").read_text(encoding="utf-8")
    assert "SkyPriorModule" not in plain_tasks and "DynamicScaleNeck" not in plain_tasks
    model = YOLO(str(YAML)).model
    if BASELINE.is_file():
        baseline = YOLO(str(BASELINE)).model
    else:
        from ultralytics.nn.tasks import DetectionModel
        baseline = DetectionModel(str(original / "cfg/models/11/yolo11.yaml"), nc=1, verbose=False)
    update_summary("checks", {"passed": result.returncode == 0, "unit_test_exit_code": result.returncode,
                               "log": "checks.log", "baseline_files_sha256": baseline_files,
                               "baseline_package_unchanged": True})
    update_summary("model", {"name": "YOLO11-STSD", "parameters": sum(p.numel() for p in model.parameters()),
                              "baseline_parameters": sum(p.numel() for p in baseline.parameters()),
                              "head_strides": model.stride.tolist(), "source_version": ultralytics.__version__,
                              "source": str(Path(ultralytics.__file__).resolve()),
                              "head": type(model.model[-1]).__name__})
    if result.returncode:
        raise SystemExit(result.returncode)


def prepare_probe(frames_per_sequence=8):
    manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))["files"]
    selected = {}
    for split, sequence_count in (("train", 2), ("validation", 1)):
        frame_count = frames_per_sequence if split == "train" else 8
        records = []
        for category in ("DRONE", "HELICOPTER", "BIRD", "AIRPLANE"):
            groups = defaultdict(list)
            for row in manifest:
                if row["split"] == split and row["category"] == category:
                    groups[row["sequence"]].append(row)
            suitable = [(name, sorted(rows, key=lambda row: row["frame"]))
                        for name, rows in sorted(groups.items()) if len(rows) >= frame_count]
            for _, rows in suitable[:sequence_count]:
                records.extend(rows[:frame_count])
        selected[split] = records
    train_sequences = {r["sequence"] for r in selected["train"]}
    assert train_sequences.isdisjoint({r["sequence"] for r in selected["validation"]})
    probe_data = RUN / "probe_data"
    probe_data.mkdir(parents=True, exist_ok=True)
    for split, records in selected.items():
        (probe_data / f"{split}.txt").write_text("\n".join(r["image"].replace("\\", "/") for r in records) + "\n", encoding="utf-8")
    import yaml
    data_yaml = probe_data / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(DATA).replace("\\", "/"),
                                        "train": str(probe_data / "train.txt").replace("\\", "/"),
                                        "val": str(probe_data / "validation.txt").replace("\\", "/"),
                                        "names": {0: "drone"}}, allow_unicode=True), encoding="utf-8")
    write_json(probe_data / "selection.json", selected)
    return data_yaml, selected


def metrics_dict(metrics):
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)}


def train(args, probe=False):
    torch.manual_seed(0)
    model = YOLO(str(YAML))
    transfer = load_baseline_weights(model, args.weights)
    write_json(RUN / "baseline_transfer.json", transfer)
    # Persist/reload through native checkpoint API. This also gives DDP workers
    # an explicit mapped-STSD initialization rather than the original YAML.
    initial = RUN / "mapped_initial.pt"
    model.save(initial)
    model = YOLO(str(initial))
    expected_state = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}
    handoff = {}
    def verify_trainer_weights(trainer):
        state = trainer.model.state_dict()
        matches = all(k in state and torch.equal(state[k].detach().cpu().float(), v.float())
                      for k, v in expected_state.items())
        if not matches:
            raise RuntimeError("Trainer reconstruction lost mapped initialization")
        handoff.update({"verified": True, "tensors": len(expected_state)})
    model.add_callback("on_pretrain_routine_end", verify_trainer_weights)
    data, selected = prepare_probe(args.probe_frames_per_sequence) if probe else (Path(args.data), None)
    name = args.name or ("probe_train" if probe else "full_train")
    epochs = args.epochs
    start = time.perf_counter()
    # False prevents automatic AMP checking from downloading another model.
    result = model.train(trainer=STSDPairTrainer if not args.spatial_only else None,
                         data=str(data), epochs=epochs, patience=15,
                         imgsz=args.imgsz, batch=args.batch, device=args.device, workers=0,
                         project=str(RUN), name=name, exist_ok=True, optimizer="AdamW",
                         lr0=0.001, lrf=0.01, nbs=args.batch, seed=0, amp=False,
                         warmup_epochs=0 if probe else 3, plots=False, close_mosaic=0,
                         mosaic=0, mixup=0, cutmix=0, copy_paste=0, val=True,
                         verbose=False, save=True)
    elapsed = time.perf_counter() - start
    best = RUN / name / "weights" / "best.pt"
    # Saved EMA weights must reload through ordinary Ultralytics predict/val.
    restored = YOLO(str(best))
    val = restored.val(data=str(data), imgsz=args.imgsz, batch=args.batch, device=args.device,
                       workers=0, plots=False, verbose=False, project=str(RUN), name=name + "_reload_val")
    temporal = temporal_modules(model.trainer.model)
    info = {"status": "completed", "epochs": epochs, "seed": 0,
            "training_mode": "ordinary images, temporal disabled" if args.spatial_only else "true sampled-frame pairs",
            "imgsz": args.imgsz, "batch_images": args.batch, "seconds": elapsed,
            "metrics": metrics_dict(val), "checkpoint": str(best.relative_to(EXP.parent)).replace("\\", "/"),
            "pretrained_tensors": transfer["transferred_tensors"],
            "trainer_initialization_handoff": handoff,
            "temporal_gamma": [float(m.gamma.detach().cpu()) for m in temporal]}
    if selected is not None:
        dataset = model.trainer.train_loader.dataset
        info.update({"train_frames": len(selected["train"]), "val_frames": len(selected["validation"]),
                     "pairs": getattr(dataset, "pair_statistics", {}),
                     "training_batches_per_epoch": len(model.trainer.train_loader),
                     "purpose": "engineering probe; baseline has a much larger completed training budget"})
        baseline_val = YOLO(str(args.weights)).val(data=str(data), imgsz=args.imgsz, batch=args.batch,
                         device=args.device, workers=0, plots=False, verbose=False,
                         project=str(RUN), name="baseline_same_probe_val")
        info["baseline_same_validation_subset"] = metrics_dict(baseline_val)
    update_summary("probe" if probe else "training", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def video(args):
    import cv2
    import numpy as np
    from PIL import Image
    selection = json.loads((RUN / "probe_data" / "selection.json").read_text(encoding="utf-8"))
    default_sequence = next(r["sequence"] for r in selection["validation"] if r["category"] == "DRONE")
    source = Path(args.source) if args.source else DATA / "raw" / (default_sequence + ".mp4")
    if args.weights:
        weights = Path(args.weights)
    else:
        summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
        weights = EXP.parent / summary["probe"]["checkpoint"]
    model = YOLO(str(weights))
    # Initialize/fuse/warm up before enabling any stream state.
    model.predict(np.zeros((640, 640, 3), np.uint8), device=args.device, imgsz=640, rect=False, verbose=False)
    set_video_mode(model.predictor.model.model, True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(source)
    frames, snapshots, used, weight_rows = [], [], 0, []
    last_processed_frame = None
    dynamic = [m for m in model.predictor.model.model.modules() if isinstance(m, DynamicScaleNeck)]
    sky = [m for m in model.predictor.model.model.modules() if isinstance(m, SkyPriorModule)]
    index = 0
    while len(frames) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.stride:
            index += 1
            continue
        prediction = model.predict(frame, device=args.device, imgsz=640, rect=False,
                                   conf=0.25, verbose=False)[0]
        last_processed_frame = frame
        memory = all(m.last_used_memory for m in temporal_modules(model.predictor.model.model))
        used += int(memory)
        weights_now = [m.last_weights[0].float().cpu().tolist() for m in dynamic]
        weight_rows.append({"frame": index, "used_memory": memory, "weights": weights_now})
        # Attention is resized as visualization only; no sky labels are implied.
        attention = sky[0].attention_map[0, 0].float().cpu().numpy()
        hh, ww = frame.shape[:2]
        ratio = min(640 / ww, 640 / hh)
        unpadded_w, unpadded_h = round(ww * ratio), round(hh * ratio)
        dense = cv2.resize(attention, (640, 640))
        left, top = round((640 - unpadded_w) / 2 - 0.1), round((640 - unpadded_h) / 2 - 0.1)
        dense = cv2.resize(dense[top:top + unpadded_h, left:left + unpadded_w], (ww, hh))
        heat = cv2.applyColorMap(np.uint8(dense.clip(0, 1) * 255), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
        left_view, right_view = cv2.resize(prediction.plot(), (480, 300)), cv2.resize(overlay, (480, 300))
        canvas = np.zeros((390, 960, 3), dtype=np.uint8)
        canvas[:300] = np.concatenate((left_view, right_view), axis=1)
        cv2.putText(canvas, f"STSD short probe | frame {index} | memory: {'ON' if memory else 'FIRST FRAME'}", (12, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, "P3 soft spatial ATTENTION (not a sky segmentation mask)", (490, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        text = " | ".join(f"P{i+3}: " + "/".join(f"{v:.2f}" for v in ws) for i, ws in enumerate(weights_now))
        cv2.putText(canvas, text, (12, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 230, 255), 1)
        cv2.putText(canvas, "Implementation demo; this short training does not establish better accuracy.", (12, 377), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 255), 1)
        rgb = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        frames.append(rgb)
        if len(frames) in (1, 16, 32, 64):
            filename = f"demo_frame_{len(frames):03d}.jpg"
            rgb.save(RUN / filename, quality=90)
            snapshots.append(filename)
        index += 1
    cap.release()
    assert frames and used == len(frames) - 1, "Cache must be used after first frame"
    frames[0].save(RUN / "video_demo.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
    # Verify a new stream begins with a clean cache even at identical dimensions.
    set_video_mode(model.predictor.model.model, True)
    model.predict(last_processed_frame, device=args.device, imgsz=640, rect=False, verbose=False)
    assert not any(m.last_used_memory for m in temporal_modules(model.predictor.model.model))
    write_json(RUN / "video_trace.json", weight_rows)
    update_summary("video", {"status": "completed", "source_sequence": source.stem, "frames": len(frames),
                             "stride": args.stride, "frames_using_memory": used,
                             "stream_reset_verified": True, "snapshots": snapshots, "demo": "video_demo.gif",
                             "trace": "video_trace.json"})
    print(json.dumps({"demo": str(RUN / "video_demo.gif"), "frames": len(frames), "used_memory": used}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    for command in ("probe", "train"):
        item = sub.add_parser(command)
        item.add_argument("--weights", default=str(BASELINE))
        item.add_argument("--data", default=str(DATA / "train.yaml"))
        item.add_argument("--epochs", type=int, default=2 if command == "probe" else 80)
        item.add_argument("--imgsz", type=int, default=640)
        item.add_argument("--batch", type=int, default=4 if command == "probe" else 8)
        item.add_argument("--device", default="0")
        item.add_argument("--spatial-only", action="store_true", help="ordinary trainer; temporal branch stays disabled")
        item.add_argument("--name", help="distinct output directory for a new training budget")
        item.add_argument("--probe-frames-per-sequence", type=int, default=8)
    item = sub.add_parser("video")
    item.add_argument("--weights")
    item.add_argument("--source")
    item.add_argument("--device", default="0")
    item.add_argument("--max-frames", type=int, default=64)
    item.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        check(args)
    elif args.command in {"probe", "train"}:
        train(args, probe=args.command == "probe")
    else:
        video(args)


if __name__ == "__main__":
    main()
