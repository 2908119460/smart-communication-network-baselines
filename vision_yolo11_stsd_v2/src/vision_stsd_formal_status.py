"""Summarize the running single-seed formal experiment without importing torch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "runs/vision_yolo11_stsd_v2"
DATA = EXP / "data/vision/paper_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def metrics(values):
    return {"map50": values.get("metrics/mAP50(B)"),
            "map50_95": values.get("metrics/mAP50-95(B)"),
            "precision": values.get("metrics/precision(B)"),
            "recall": values.get("metrics/recall(B)")}


def snapshot():
    pipeline = read(ROOT / "formal_pipeline_seed0/manifest.json")
    counts = read(DATA / "protocol.json").get("counts", {})
    sampling = read(DATA / "sampling_audit.json")
    train_frames = counts.get("train", {}).get("frames", 4296)
    val_frames = counts.get("validation", {}).get("frames", 792)
    test_frames = counts.get("test", {}).get("frames", 792)
    runs = []
    for kind, name in (("stsd", "formal_seed0"), ("pair_control", "formal_pair_baseline_seed0")):
        manifest = read(ROOT / name / "run_manifest.json")
        progress = read(ROOT / name / "progress.json")
        heartbeat = read(ROOT / name / "heartbeat.json")
        finished = read(ROOT / name / "training_summary.json")
        history = progress.get("epochs", [])
        latest = history[-1] if history else {}
        best = max(history, key=lambda r: (r.get("fitness", -1), r["epoch"])) if history else {}
        status = finished.get("status", manifest.get("status", "queued"))
        current = progress.get("epoch", 0)
        if status not in {"completed", "failed"}:
            current = max(current, heartbeat.get("epoch", 0))
        runs.append({"id": kind, "name": "YOLO11-STSD v2" if kind == "stsd" else "YOLO11n 配对训练对照",
                     "status": status, "started_at": manifest.get("started_utc"),
                     "current_epoch": current, "completed_epochs": progress.get("epoch", 0),
                     "max_epochs": manifest.get("epochs_max", 80),
                     "best_epoch": finished.get("best_validation_epoch", best.get("epoch")),
                     "latest_val": metrics(latest.get("validation", {})),
                     "best_val": finished.get("validation", metrics(best.get("validation", {}))),
                     "epoch_seconds": latest.get("epoch_seconds"),
                     "observed_budget": heartbeat.get("budget", progress.get("budget", {})),
                     "last_update": progress.get("updated_utc"),
                     "early_stopped": finished.get("early_stopped"),
                     "runtime": read(ROOT / name / "runtime_setup.json")})
    evaluation = read(ROOT / "formal_seed0/evaluation/summary.json")
    evaluation_rows = []
    for name, splits in evaluation.get("detection", {}).items():
        for split, values in splits.items():
            evaluation_rows.append({"name": name, "split": split,
                                    "map50": values["mAP50"], "map50_95": values["mAP50_95"],
                                    "precision": values["precision"], "recall": values["recall"]})
    stages = pipeline.get("steps", [])
    stage = stages[-1].get("stage") if stages else None
    phase = {"stsd": "stsd_training", "control": "pair_control_training", "evaluation": "evaluation"}.get(stage, "queued")
    if pipeline.get("status") == "completed":
        phase = "completed"
    if pipeline.get("status") == "waiting_for_gpu":
        phase = "waiting_for_gpu"
    return {"schema_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": pipeline.get("status", "queued"), "phase": phase,
            "registered_at": pipeline.get("started_utc"),
            "started_at": (None if pipeline.get("status") == "waiting_for_gpu" else
                           pipeline.get("gpu_acquired_utc", pipeline.get("started_utc"))),
            "data": {"name": "DroneDetectionThesis RGB (not official Anti-UAV)",
                     "train_frames": train_frames, "val_frames": val_frames, "test_frames": test_frames,
                     "train_videos": counts.get("train", {}).get("videos"),
                     "pair_original_frame_gap_min": sampling.get("pair_original_frame_gap_min"),
                     "pair_original_frame_gap_max": sampling.get("pair_original_frame_gap_max")},
            "config": {"epochs": pipeline.get("epochs_max_each", 80), "patience": 15, "seed": 0, "batch": 8, "imgsz": 640,
                       "initialization": "public_coco", "pair_stride": 2, "overlapping_pairs": False,
                       "exposures_per_epoch": train_frames, "batches_per_epoch": (train_frames + 7)//8},
            "runs": runs,
            "evaluation": {"status": evaluation.get("status", "queued"), "val_frames": val_frames,
                           "test_frames": test_frames, "full_test_videos": 4,
                           "results": evaluation_rows, "continuous": evaluation.get("continuous", {})},
            "limitations": ["The published status is a timestamped snapshot, not a live GPU connection.",
                            "Running validation metrics are provisional; test runs after checkpoint selection.",
                            "Matched pair control shares the data transforms; historical baseline is a reference."]}


def export(copy_to_pages=False, quiet=False, public_name="formal_status.json"):
    value = snapshot()
    path = ROOT / "formal_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    if copy_to_pages:
        public = EXP.parent / "github_pages/vision_yolo11_stsd_v2"
        public.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, public / public_name)
    if not quiet:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-to-pages", action="store_true", help="Copy snapshot locally; does not push Git")
    parser.add_argument("--watch", action="store_true", help="Refresh local status every 30s until the pipeline stops")
    parser.add_argument("--run-root", default=str(ROOT))
    parser.add_argument("--data-root", default=str(DATA))
    parser.add_argument("--public-name", default="formal_status.json")
    args = parser.parse_args()
    ROOT, DATA = Path(args.run_root).resolve(), Path(args.data_root).resolve()
    while True:
        value = export(args.copy_to_pages, quiet=args.watch, public_name=args.public_name)
        if not args.watch or value["status"] in {"completed", "failed"}:
            break
        time.sleep(30)
