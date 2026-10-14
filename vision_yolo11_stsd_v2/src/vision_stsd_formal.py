"""Full-data, single-seed STSD and matched YOLO11n control training.

The supervisor runs one CUDA worker at a time, then evaluates best validation
checkpoints. Existing runs are never overwritten. No retry or seed sweep.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "runs" / "vision_yolo11_stsd_v2"
FORK = EXP / "sources" / "ultralytics_stsd"
DATA = EXP / "data" / "vision" / "paper_v1"
SOURCE = DATA / "weights" / "yolo11n.pt"
MODEL_YAML = FORK / "ultralytics/cfg/models/11/yolo11n-stsd.yaml"
RUN_NAMES = {"stsd": "formal_seed0", "control": "formal_pair_baseline_seed0"}
sys.path.insert(0, str(FORK))
os.environ["PYTHONPATH"] = str(FORK) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tee:
    def __init__(self, original, file):
        self.original, self.file = original, file

    def write(self, value):
        self.original.write(value)
        self.file.write(value)
        self.file.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def train(kind, epochs):
    out = ROOT / RUN_NAMES[kind]
    if (out / "run_manifest.json").exists() or (out / "weights").exists():
        raise FileExistsError(f"Refusing to replace existing training: {out}")
    out.mkdir(parents=True, exist_ok=True)
    stream = (out / "console.log").open("w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, stream), Tee(sys.stderr, stream)
    settings = out / "settings"
    settings.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings)
    # Native AMP check resolves its reference detector to this verified local file.
    os.chdir(SOURCE.parent)
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from stsd_tools import STSDPairTrainer, load_baseline_weights, temporal_modules

    torch.set_num_threads(2)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this formal run")
    locked = read(DATA / "protocol.json")
    if digest(DATA / "split_manifest.json") != locked["split_manifest_sha256"]:
        raise RuntimeError("Existing data split changed")
    weights_manifest = Path(locked.get("weights_manifest_path", str(DATA / "weights/manifest.json")))
    source_record = next(r for r in read(weights_manifest)["files"]
                         if Path(r["path"]).name == SOURCE.name)
    if digest(SOURCE) != source_record["sha256"]:
        raise RuntimeError("Official initialization checksum mismatch")
    data = yaml.safe_load((DATA / "train.yaml").read_text(encoding="utf-8"))
    data.update(temporal_pair_stride=2, temporal_max_frame_gap=32)
    data_yaml = out / "formal_data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = dict(epochs=epochs, patience=15, imgsz=640, batch=8, nbs=8,
                  optimizer="AdamW", lr0=0.001, lrf=0.01, warmup_epochs=3.0,
                  warmup_bias_lr=0.01, seed=0, deterministic=True, amp=True,
                  device=0, workers=0, cache=False, plots=False, save=True,
                  mosaic=0, mixup=0, cutmix=0, copy_paste=0, close_mosaic=0,
                  verbose=False)
    started = time.monotonic()
    manifest = {"status": "initializing", "kind": kind, "pid": os.getpid(),
                "started_utc": utc(), "epochs_max": epochs, "seed": 0,
                "source_checkpoint": str(SOURCE), "source_sha256": digest(SOURCE),
                "split_manifest_sha256": digest(DATA / "split_manifest.json"),
                "configuration": config, "counts": locked["counts"],
                "model_yaml_sha256": digest(MODEL_YAML) if kind == "stsd" else None,
                "pair_training_source_sha256": digest(FORK / "stsd_tools/training.py"),
                "runner_sha256": digest(Path(__file__)),
                "initialization": "official COCO pretraining, not completed drone baseline",
                "checkpoint_selection": "validation mAP50-95 only; temporal disabled during validation",
                "pair_sampling": "disjoint same-video sampled-frame pairs; one exposure per image per epoch",
                "training_geometry": "deterministic square letterbox; no random augmentation in paired trainer",
                "test_used_for_training_or_selection": False,
                "limitations": ["Current dataset is DroneDetectionThesis, not official Anti-UAV",
                    "Temporal spacing is recorded in runtime_setup.json; sampling is dataset-specific",
                    "Original historical baseline uses different augmentation; paired control is matched",
                    "Single seed; no significance or generalization claim"]}
    save(out / "run_manifest.json", manifest)
    budget = {"batches": 0, "optimizer_updates": 0}
    history, handoff = [], {}
    try:
        if kind == "stsd":
            model = YOLO(str(MODEL_YAML))
            transfer = load_baseline_weights(model, SOURCE)
        else:
            original_yaml = FORK / "ultralytics/cfg/models/11/yolo11.yaml"
            model = YOLO(str(original_yaml))
            model.model = DetectionModel(str(original_yaml), nc=1, verbose=False)
            pretrained = YOLO(str(SOURCE)).model
            source_state = pretrained.state_dict()
            target_state = model.model.state_dict()
            matching = {k: v for k, v in source_state.items()
                        if k in target_state and target_state[k].shape == v.shape}
            model.model.load_state_dict(matching, strict=False)
            transfer = {"checkpoint": str(SOURCE), "transferred_tensors": len(matching),
                        "source_tensors": len(source_state), "skipped_tensors": len(source_state)-len(matching)}
        save(out / "initialization_transfer.json", transfer)
        initial = out / "initialization" / "mapped_initial.pt"
        initial.parent.mkdir(exist_ok=True)
        model.save(initial)
        model = YOLO(str(initial))
        expected = {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}

        def verify_weights(trainer):
            state = trainer.model.state_dict()
            if not all(k in state and torch.equal(state[k].detach().cpu().float(), v.float())
                       for k, v in expected.items()):
                raise RuntimeError("Trainer lost mapped initialization")
            handoff.update(verified=True, tensors=len(expected))
            save(out / "initialization_handoff.json", handoff)

        def runtime(trainer):
            trainer.optimizer.register_step_post_hook(
                lambda *unused: budget.__setitem__("optimizer_updates", budget["optimizer_updates"]+1))
            stats = trainer.train_loader.dataset.pair_statistics
            if stats["unique_frames_used"] != locked["counts"]["train"]["frames"]:
                raise RuntimeError("Disjoint pairing did not preserve all training frames")
            if stats["unique_frames_used"] != 2 * stats["pairs"]:
                raise RuntimeError("Disjoint frame pairs overlap")
            setup = {"amp_actual": bool(trainer.amp), "physical_batch_images": trainer.batch_size,
                     "batches_per_epoch": len(trainer.train_loader), "pairs": stats,
                     "parameters": sum(p.numel() for p in trainer.model.parameters()),
                     "device": str(trainer.device), "gpu": torch.cuda.get_device_name(0)}
            save(out / "runtime_setup.json", setup)
            manifest.update(status="running", runtime_setup=setup)
            save(out / "run_manifest.json", manifest)
            save(out / "progress.json", {"status": "running", "epoch": 0, "epochs_max": epochs,
                 "updated_utc": utc(), "budget": budget, "epochs": history})

        def batch_end(trainer):
            budget["batches"] += 1
            if budget["batches"] % 50 == 0:
                save(out / "heartbeat.json", {"pid": os.getpid(), "updated_utc": utc(),
                     "epoch": int(trainer.epoch)+1, "budget": budget,
                     "elapsed_seconds": time.monotonic()-started})

        def epoch_end(trainer):
            epoch = int(trainer.epoch)+1
            if any(r["epoch"] == epoch for r in history):
                return
            row = {"epoch": epoch, "epoch_seconds": float(trainer.epoch_time),
                   "elapsed_seconds": time.monotonic()-started,
                   "validation": {k: float(v) for k, v in trainer.metrics.items()},
                   "fitness": float(trainer.fitness), "budget": dict(budget)}
            history.append(row)
            save(out / "progress.json", {"status": "running", "epoch": epoch,
                 "epochs_max": epochs, "updated_utc": utc(), "budget": budget, "epochs": history})
            print("FORMAL_EPOCH " + json.dumps(row), flush=True)

        model.add_callback("on_pretrain_routine_end", verify_weights)
        model.add_callback("on_train_start", runtime)
        model.add_callback("on_train_batch_end", batch_end)
        model.add_callback("on_fit_epoch_end", epoch_end)
        torch.cuda.reset_peak_memory_stats()
        model.train(trainer=STSDPairTrainer, data=str(data_yaml), project=str(ROOT),
                    name=RUN_NAMES[kind], exist_ok=True, **config)
        import pandas as pd
        table = pd.read_csv(out / "results.csv")
        best_row = table[table["metrics/mAP50-95(B)"] == table["metrics/mAP50-95(B)"].max()].iloc[-1]
        best = out / "weights/best.pt"
        restored = YOLO(str(best))
        metrics = restored.val(data=str(data_yaml), imgsz=640, batch=8, device=0,
                              workers=0, plots=False, verbose=False, project=str(out), name="reload_val")
        summary = {**manifest, "status": "completed", "finished_utc": utc(),
                   "elapsed_seconds": time.monotonic()-started, "epochs_completed": len(table),
                   "early_stopped": len(table) < epochs, "best_validation_epoch": int(best_row["epoch"]),
                   "validation": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                                  "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)},
                   "checkpoint": str(best), "checkpoint_sha256": digest(best), "budget": budget,
                   "trainer_initialization_handoff": handoff, "transfer": transfer["transferred_tensors"],
                   "temporal_gamma": [float(m.gamma.detach().cpu()) for m in temporal_modules(model.trainer.model)],
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
        save(out / "training_summary.json", summary)
        save(out / "progress.json", {"status": "completed", "epoch": len(table), "epochs_max": epochs,
             "updated_utc": utc(), "budget": budget, "epochs": history})
        manifest.update(status="completed", finished_utc=utc())
        save(out / "run_manifest.json", manifest)
        print("FORMAL_COMPLETED " + json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception:
        manifest.update(status="failed", finished_utc=utc())
        save(out / "run_manifest.json", manifest)
        save(out / "failure.json", {"status": "failed", "budget": budget,
             "elapsed_seconds": time.monotonic()-started, "traceback": traceback.format_exc()})
        raise


def supervise(epochs, wait_pipeline=None):
    out = ROOT / "formal_pipeline_seed0"
    if (out / "manifest.json").exists():
        raise FileExistsError("Formal pipeline already exists; inspect it instead of starting a duplicate")
    out.mkdir(parents=True, exist_ok=True)
    state = {"status": "waiting_for_gpu" if wait_pipeline else "running", "pid": os.getpid(), "started_utc": utc(),
             "epochs_max_each": epochs, "seed": 0, "serial_gpu_workers": True,
             "monitoring": "process poll and heartbeat every 30s; no automatic retries",
             "timeout_monitoring": "elapsed time recorded; 80-epoch/15-patience budget, no time-based truncation",
             "data_root": str(DATA), "run_root": str(ROOT), "source_checkpoint": str(SOURCE),
             "wait_pipeline": str(wait_pipeline) if wait_pipeline else None, "steps": []}
    save(out / "manifest.json", state)
    commands = [(kind, [sys.executable, "-u", str(Path(__file__).resolve()), "train",
                       "--kind", kind, "--epochs", str(epochs),
                       "--data-root", str(DATA), "--run-root", str(ROOT), "--weights", str(SOURCE)])
                for kind in ("stsd", "control")]
    commands.append(("evaluation", [sys.executable, "-u", str(EXP / "scripts/vision_stsd_formal_evaluate.py"),
                       "--weights", str(ROOT / "formal_seed0/weights/best.pt"),
                       "--baseline-weights", str(ROOT / "formal_pair_baseline_seed0/weights/best.pt"),
                       "--output", str(ROOT / "formal_seed0/evaluation"), "--device", "0"]))
    try:
        if wait_pipeline:
            while True:
                previous = read(wait_pipeline)
                state["last_monitor_utc"] = utc()
                state["waiting_pipeline_status"] = previous["status"]
                save(out / "manifest.json", state)
                if previous["status"] == "completed":
                    break
                if previous["status"] == "failed":
                    raise RuntimeError("Earlier GPU pipeline failed; inspect it before starting this queued run")
                time.sleep(30)
            state.update(status="running", gpu_acquired_utc=utc())
            save(out / "manifest.json", state)
        for kind, command in commands:
            with (out / (kind + ".log")).open("w", encoding="utf-8") as stream:
                child = subprocess.Popen(command, cwd=str(EXP.parent), stdout=stream, stderr=subprocess.STDOUT)
                row = {"stage": kind, "pid": child.pid, "status": "running", "started_utc": utc(),
                       "command": command, "log": str(out / (kind + ".log"))}
                state["steps"].append(row)
                save(out / "manifest.json", state)
                while child.poll() is None:
                    state["last_monitor_utc"] = utc()
                    save(out / "manifest.json", state)
                    time.sleep(30)
                row.update(status="completed" if child.returncode == 0 else "failed",
                           exit_code=child.returncode, finished_utc=utc())
                save(out / "manifest.json", state)
                if child.returncode:
                    raise RuntimeError(f"{kind} exited with code {child.returncode}; inspect {row['log']}")
        state.update(status="completed", finished_utc=utc())
        save(out / "manifest.json", state)
    except Exception:
        state.update(status="failed", finished_utc=utc(), error=traceback.format_exc())
        save(out / "manifest.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["all", "train"])
    parser.add_argument("--kind", choices=list(RUN_NAMES), default="stsd")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--data-root", default=str(DATA), help="Independent prepared dataset directory")
    parser.add_argument("--run-root", default=str(ROOT), help="Independent experiment output directory")
    parser.add_argument("--weights", default=str(SOURCE), help="Verified official initialization checkpoint")
    parser.add_argument("--wait-pipeline", type=Path, help="Wait for an existing GPU pipeline to complete")
    args = parser.parse_args()
    DATA, ROOT, SOURCE = Path(args.data_root).resolve(), Path(args.run_root).resolve(), Path(args.weights).resolve()
    train(args.kind, args.epochs) if args.action == "train" else supervise(args.epochs, args.wait_pipeline)
