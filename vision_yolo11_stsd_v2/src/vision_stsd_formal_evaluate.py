"""Evaluate trained STSD and YOLO checkpoints on the existing locked RGB split.

Detection mAP uses every prepared validation/test image. Temporal on/off uses
all original frames of four already fixed test videos; these presence metrics
are a temporal diagnostic, not full-test video mAP or an Anti-UAV benchmark.
No threshold search, checkpoint search, or additional training is performed.
Run with experiments/.venv/Scripts/python.exe. --plan-only does not import torch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

EXP = Path(__file__).resolve().parents[1]
FORK = EXP / "sources" / "ultralytics_stsd"
DATA = EXP / "data" / "vision" / "paper_v1"
FORMAL = EXP / "runs" / "vision_yolo11_stsd_v2" / "formal_seed0"
ORIGINAL_BASELINE = EXP / "runs" / "vision_paper_v1_yolo11n_seed0_attempt2" / "weights" / "best.pt"
CATEGORIES = ("DRONE", "BIRD", "HELICOPTER", "AIRPLANE")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def presence(records, threshold):
    tp = sum(bool(row["positive"]) and row["score"] >= threshold for row in records)
    fp = sum(not row["positive"] and row["score"] >= threshold for row in records)
    fn = sum(bool(row["positive"]) and row["score"] < threshold for row in records)
    tn = len(records) - tp - fp - fn
    return {
        "threshold": threshold, "frames": len(records), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "negative_frame_FPR": fp / (fp + tn) if fp + tn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def summarize_presence(records, threshold):
    by_category, by_video = defaultdict(list), defaultdict(list)
    for row in records:
        by_category[row["category"]].append(row)
        by_video[row["sequence"]].append(row)
    return {
        "frame_micro": presence(records, threshold),
        "per_source_category": {key: presence(rows, threshold) for key, rows in sorted(by_category.items())},
        "per_video": {key: presence(rows, threshold) for key, rows in sorted(by_video.items())},
        "interpretation": "Presence means any predicted drone versus any DRONE GT box; no IoU requirement. Categories identify source videos, not a multiclass detector.",
    }


def latency_summary(milliseconds):
    if not milliseconds:
        return {"frames": 0}
    ordered = sorted(milliseconds)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    mean = statistics.mean(milliseconds)
    return {"frames": len(milliseconds), "mean_ms": mean, "median_ms": statistics.median(milliseconds),
            "p95_ms": ordered[index], "throughput_fps": 1000 / mean if mean else None,
            "scope": "Sequential batch=1 predict call incl. preprocessing, GPU synchronization and NMS; excludes file decoding, GT reading and JSON writing; after one synthetic warmup."}


def make_plan(args):
    protocol = read_json(DATA / "protocol.json")
    evaluation = read_json(DATA / "evaluation_protocol.json")
    manifest = read_json(DATA / "split_manifest.json")
    for filename, key in (("protocol.json", "protocol_sha256"), ("split_manifest.json", "split_manifest_sha256")):
        if digest(DATA / filename) != evaluation[key]:
            raise ValueError(f"Existing locked {filename} differs from evaluation_protocol.json")
    audit_by_name = {row["sequence"]: row for row in manifest["video_audit"]}
    selected = []
    for category in CATEGORIES:
        first = min(name for name, row in protocol["assignments"].items()
                    if row["split"] == "test" and row["category"] == category)
        if evaluation["demo_selection"][category] != first:
            raise ValueError(f"Locked video selection differs from lexicographically first {category} test sequence")
        audit = audit_by_name[first]
        if audit["split"] != "test":
            raise ValueError(f"Selected sequence is not test: {first}")
        selected.append({"sequence": first, "category": category, "frames": audit["frames_original"],
                         "fps": audit["fps"], "source_sha256": audit["source_sha256"],
                         "annotation_sha256": audit["annotation_sha256"]})
    checkpoints = {"baseline": str(Path(args.baseline_weights).resolve()),
                   "stsd": str(Path(args.weights).resolve())}
    reference = Path(args.reference_weights).resolve()
    if reference != Path(checkpoints["baseline"]):
        checkpoints["original_reference"] = str(reference)
    return {
        "created_utc": utc_now(), "version": "yolo11_stsd_v2_formal_seed0",
        "dataset": "DroneDetectionThesis / Svanstrom RGB; not official Anti-UAV",
        "data_yaml": str(DATA / "train.yaml"), "checkpoints": checkpoints,
        "protocol_sha256": digest(DATA / "protocol.json"),
        "split_manifest_sha256": digest(DATA / "split_manifest.json"),
        "evaluation_protocol_sha256": digest(DATA / "evaluation_protocol.json"),
        "detection": {"splits": ["val", "test"], "state": "stateless",
                      "imgsz": args.imgsz, "batch": args.batch, "conf": 0.001,
                      "iou": 0.7, "max_det": 300, "half": False, "augment": False,
                      "prepared_images": {"val": sum(row["split"] == "validation" for row in manifest["files"]),
                                          "test": sum(row["split"] == "test" for row in manifest["files"])},
                      "P_R_interpretation": "Ultralytics validation P/R at its PR-curve operating point; not fixed-confidence video P/R."},
        "video": {"selection": selected, "selection_rule": evaluation["demo_selection_rule"],
                  "scope": "All original frames of four fixed test videos only; not all 33 test videos.",
                  "frames_per_method": sum(row["frames"] for row in selected),
                  "presence_threshold": args.presence_conf, "threshold_rule": "Explicit fixed threshold; never selected on test.",
                  "inference_conf": 0.001, "iou": 0.7, "max_det": 300,
                  "imgsz": args.imgsz, "rect": False, "batch": 1, "vid_stride": 1,
                  "half": False, "augment": False,
                  "methods": ["baseline", "stsd_temporal_off", "stsd_temporal_on"] +
                             (["original_reference"] if "original_reference" in checkpoints else [])},
        "limitations": ["Single training seed; no additional seeds or threshold search.",
                        "Temporal on/off is an inference ablation of the same STSD checkpoint, not a no-temporal-training ablation.",
                        "Original reference uses its original recipe; pair-control baseline, if supplied, better isolates architecture from training recipe.",
                        "Continuous-video presence does not enforce box IoU and cannot substitute for detection mAP.",
                        "Four preset videos provide a bounded temporal diagnostic, not a population-wide estimate.",
                        "Adjacent original-frame inference differs from the sampled-frame spacing used for temporal training."],
    }


def configure_runtime(output):
    sys.path.insert(0, str(FORK))
    os.environ["PYTHONPATH"] = str(FORK) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["YOLO_CONFIG_DIR"] = str(output / "settings")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    sys.dont_write_bytecode = True
    import torch
    import ultralytics
    torch.set_num_threads(2)
    if FORK.resolve() not in Path(ultralytics.__file__).resolve().parents:
        raise RuntimeError("Evaluation must use the isolated STSD fork")
    return torch, ultralytics


def detection_evaluation(args, plan, output):
    from ultralytics import YOLO
    from stsd_tools import set_video_mode
    results = {}
    for key, checkpoint in plan["checkpoints"].items():
        model = YOLO(checkpoint)
        set_video_mode(model.model, False)
        results[key] = {}
        for split in plan["detection"]["splits"]:
            print(f"DETECTION_START {key} {split}", flush=True)
            metrics = model.val(data=plan["data_yaml"], split=split, imgsz=args.imgsz, batch=args.batch,
                                workers=0, device=args.device, conf=0.001, iou=0.7, max_det=300,
                                half=False, augment=False, plots=False, verbose=False,
                                project=str(output / "detection"), name=f"{key}_{split}", exist_ok=False)
            values = metrics.results_dict
            results[key][split] = {
                "frames": plan["detection"]["prepared_images"][split],
                "mAP50": float(values["metrics/mAP50(B)"]),
                "mAP50_95": float(values["metrics/mAP50-95(B)"]),
                "precision": float(values["metrics/precision(B)"]),
                "recall": float(values["metrics/recall(B)"]),
                "speed_ms_per_image": {name: float(value) for name, value in metrics.speed.items()},
                "stateless": True,
            }
            write_json(output / "detection_results.json", results)
            print(f"DETECTION_COMPLETE {key} {split} {json.dumps(results[key][split])}", flush=True)
        del model
    return results


def annotation_reader():
    # The existing helper changes cwd at import; retain this caller's cwd.
    original_cwd = Path.cwd()
    scripts_path = str(EXP / "scripts")
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
    try:
        from vision_prepare import load_annotation_table
    finally:
        os.chdir(original_cwd)
    return load_annotation_table


def continuous_evaluation(args, plan, output, torch):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from stsd_tools import set_video_mode, temporal_modules
    load_annotations = annotation_reader()
    annotations = {}
    for selected in plan["video"]["selection"]:
        name = selected["sequence"]
        source, labels = DATA / "raw" / f"{name}.mp4", DATA / "raw" / f"{name}_LABELS.mat"
        if digest(source) != selected["source_sha256"] or digest(labels) != selected["annotation_sha256"]:
            raise ValueError(f"Locked video/annotation bytes changed: {name}")
        annotations[name] = load_annotations(labels)["DRONE"]
        if len(annotations[name]) != selected["frames"]:
            raise ValueError(f"Annotation frame count mismatch: {name}")

    results = {}
    for method in plan["video"]["methods"]:
        checkpoint_key = "stsd" if method.startswith("stsd_") else method
        model = YOLO(plan["checkpoints"][checkpoint_key])
        set_video_mode(model.model, False)
        model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), device=args.device,
                      imgsz=args.imgsz, rect=False, conf=0.001, iou=0.7, max_det=300,
                      half=False, augment=False, verbose=False, save=False)
        native = model.predictor.model.model
        modules = temporal_modules(native)
        use_temporal = method == "stsd_temporal_on"
        if method.startswith("stsd_") and len(modules) != 3:
            raise ValueError(f"Expected three STSD temporal modules, got {len(modules)}")
        all_records, all_latency, per_video_cache = [], [], {}
        for selected in plan["video"]["selection"]:
            name = selected["sequence"]
            print(f"VIDEO_START {method} {name} expected_frames={selected['frames']}", flush=True)
            set_video_mode(native, use_temporal)
            capture = cv2.VideoCapture(str(DATA / "raw" / f"{name}.mp4"))
            if not capture.isOpened():
                raise FileNotFoundError(f"Cannot decode {name}")
            records, elapsed_ms, module_counts = [], [], [0] * len(modules)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    index = len(records)
                    if index >= len(annotations[name]):
                        raise ValueError(f"Video has more frames than annotations: {name}")
                    cuda_device = model.predictor.model.device
                    if cuda_device.type == "cuda":
                        torch.cuda.synchronize(cuda_device)
                    start = time.perf_counter()
                    prediction = model.predict(frame, device=args.device, imgsz=args.imgsz, rect=False,
                                               conf=0.001, iou=0.7, max_det=300, half=False,
                                               augment=False, verbose=False, save=False)[0]
                    if cuda_device.type == "cuda":
                        torch.cuda.synchronize(cuda_device)
                    milliseconds = (time.perf_counter() - start) * 1000
                    flags = [bool(module.last_used_memory) for module in modules]
                    for module_index, used in enumerate(flags):
                        module_counts[module_index] += int(used)
                    if flags and any(flags) != all(flags):
                        raise RuntimeError(f"Temporal pyramid cache disagrees: {method} {name} {index}")
                    confidences = prediction.boxes.conf.cpu().tolist()
                    records.append({"sequence": name, "category": selected["category"], "frame": index,
                                    "time_seconds": index / selected["fps"],
                                    "positive": bool(annotations[name][index]),
                                    "score": max(confidences, default=0.0),
                                    "pred_boxes": prediction.boxes.xyxy.cpu().tolist(), "confidences": confidences,
                                    "used_memory": bool(flags) and all(flags),
                                    "cache_used_by_module": flags, "predict_ms": milliseconds})
                    elapsed_ms.append(milliseconds)
            finally:
                capture.release()
            if len(records) != selected["frames"]:
                raise ValueError(f"Incomplete original video: {name} {len(records)}/{selected['frames']}")
            expected = len(records) - 1 if use_temporal else 0
            if any(count != expected for count in module_counts):
                raise RuntimeError(f"Unexpected cache use: {method} {name}: {module_counts}, expected {expected}")
            per_video_cache[name] = {"frames": len(records), "frames_using_memory": sum(row["used_memory"] for row in records),
                                     "module_cache_use_counts": module_counts, "expected_per_module": expected,
                                     "first_frame_without_memory": not records[0]["used_memory"]}
            write_json(output / "continuous" / method / f"{name}.json", {
                "selected_video": selected, "presence": presence(records, args.presence_conf),
                "latency": latency_summary(elapsed_ms), "cache": per_video_cache[name], "records": records})
            all_records.extend(records)
            all_latency.extend(elapsed_ms)
            print(f"VIDEO_COMPLETE {method} {name} frames={len(records)} cache={module_counts}", flush=True)
        set_video_mode(native, False)
        results[method] = {"checkpoint_key": checkpoint_key, "temporal_enabled": use_temporal,
                           "presence": summarize_presence(all_records, args.presence_conf),
                           "latency": latency_summary(all_latency), "per_video_cache": per_video_cache,
                           "total_frames": len(all_records),
                           "total_frames_using_memory": sum(row["used_memory"] for row in all_records)}
        write_json(output / "continuous_results.json", results)
        del model
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(FORMAL / "weights" / "best.pt"))
    parser.add_argument("--baseline-weights", default=str(ORIGINAL_BASELINE),
                        help="Use the matched-pair YOLO11n control best.pt when available")
    parser.add_argument("--reference-weights", default=str(ORIGINAL_BASELINE),
                        help="Original completed YOLO11n; deduplicated when equal to --baseline-weights")
    parser.add_argument("--output", default=str(FORMAL / "evaluation"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--presence-conf", type=float, default=0.25)
    parser.add_argument("--plan-only", action="store_true", help="Print the locked evaluation plan; no model loading or GPU use")
    args = parser.parse_args()
    if not 0 < args.presence_conf <= 1:
        parser.error("--presence-conf must be in (0, 1]")
    plan = make_plan(args)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    for checkpoint in plan["checkpoints"].values():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output is non-empty; choose a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan["checkpoint_sha256"] = {key: digest(path) for key, path in plan["checkpoints"].items()}
    write_json(output / "evaluation_plan.json", plan)
    summary = {"status": "running", "started_utc": utc_now(), "plan": plan}
    write_json(output / "summary.json", summary)
    start = time.monotonic()
    try:
        torch, ultralytics = configure_runtime(output)
        summary["runtime"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                              "source": str(Path(ultralytics.__file__).resolve()), "device": args.device}
        summary["detection"] = detection_evaluation(args, plan, output)
        write_json(output / "summary.json", summary)
        summary["continuous"] = continuous_evaluation(args, plan, output, torch)
        summary.update(status="completed", completed_utc=utc_now(), elapsed_seconds=time.monotonic() - start)
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                       elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        write_json(output / "summary.json", summary)
    print(f"FORMAL_EVALUATION_COMPLETE {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
