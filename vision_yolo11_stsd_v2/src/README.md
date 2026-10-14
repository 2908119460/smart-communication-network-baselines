# YOLO11-STSD

Isolated source fork of Ultralytics **8.3.253**, AGPL-3.0. The original installed
baseline package is unchanged. Source modifications are in `ultralytics/nn/modules`,
`ultralytics/nn/tasks.py`, and `ultralytics/cfg/models/11/yolo11n-stsd.yaml`.

Default architecture (v2): original YOLO11n backbone → three soft spatial
attention modules → three dynamic scale fusion branches → three optional temporal
modules → original Detect. Dynamic fusion **replaces** the fixed FPN/PAN.
`yolo11n-stsd.yaml` is the default. `yolo11n-stsd-pan.yaml` retains the earlier v1
additive-PAN prototype for comparison; its results do not describe v2.

Full-data experiments are launched separately from the engineering probe. Run
`python experiments/scripts/vision_stsd_formal.py all --epochs 80`. This serially
trains STSD and a matching YOLO11n control from the same official COCO checkpoint,
then evaluates validation/test and four fixed complete videos. Each run uses
seed 0, batch 8 and validation-only early stopping (patience 15). The formal
dataset sets `temporal_pair_stride: 2`: 2,148 disjoint pairs expose all 4,296
training images once per epoch (537 batches). Both models use the same
deterministic letterbox pipeline. Original augmentation-based baseline results
remain a separate reference. Status lives in `formal_status.json`; logs and
checkpoints stay under `formal_seed0/` and `formal_pair_baseline_seed0/`.

For a 640×640 input, backbone features are `[B,128,80,80]`, `[B,128,40,40]`,
and `[B,256,20,20]`. Sky attention keeps these shapes. Each dynamic branch
projects all three inputs to a common channel count and resizes them to its
target grid. Output features are `[B,64,80,80]`, `[B,128,40,40]`, and
`[B,256,20,20]`, exactly the original nano Detect inputs. Temporal modules keep
these shapes. Three output branches are required because the existing Detect
head consumes a pyramid, rather than only the P3 fused tensor.

## Run in the existing workspace

```powershell
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py check
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py probe --epochs 8 --probe-frames-per-sequence 16 --name probe_train_8ep --device 0
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py train --epochs 80 --batch 8 --device 0
experiments\.venv\Scripts\python.exe experiments\scripts\vision_yolo11_stsd.py video --device 0
```

`train` defaults to the existing dataset YAML and completed single-class baseline
checkpoint, with semantic layer remapping. In v2, only the original backbone
layers 0–10 and Detect layer 23 (now layer 20) are eligible for transfer; original
neck layers 11–22 are deliberately skipped. New modules learn from initialization.
It saves/reloads the mapped initial checkpoint and verifies the actual trainer
weights before optimization. `--model-yaml` and `--run-dir` select other isolated
configurations and output directories.

The optional `STSDPairTrainer` shuffles temporal pairs as units and keeps the
standard BCHW model input, Detect head, loss, optimizer and label format. Both
frames retain their own detection loss. No training memory crosses batches.
`batch` is the total number of images and must be even on each device. The first
version uses deterministic square letterboxing without mosaic or random geometry.
Ordinary training (`--spatial-only`) does not learn temporal fusion.

Supported frame naming: `<sequence>_<number>.jpg` or per-sequence folders containing
numbered frame names such as `000001.jpg`. Only images inside the current split
are paired. `temporal_max_frame_gap: 32` in data YAML controls allowable frame gaps.
`temporal_pair_stride: 1` is the default overlapping-pair sampling used by the
historical probe. Set it to 2 for disjoint pairs as in the formal experiment.
Other conventions can use explicit same-sequence index tuples in
`TemporalPairDataset`. Video cache assumes stable lanes; reset on every new stream.

## Portable source archive

Extract `portableimplementation.zip` preserving its `experiments/` hierarchy.
Create a separate Python environment with PyTorch/torchvision compatible with your
CUDA installation, then install `ultralytics==8.3.253` for its dependencies. The
runner puts this fork first on `sys.path`; installing the fork over the baseline
environment is unnecessary. The archive includes source and tests, not weights,
datasets or a virtual environment.

For the formal experiment on the existing RGB dataset, also install
`pip install -r experiments/sources/ultralytics_stsd/requirements-formal.txt`.
The annotation reader `experiments/scripts/vision_prepare.py` is included;
`mcos-decoder` reads the dataset's MATLAB MCOS ground-truth tables. Formal
evaluation requires the existing split/protocol files and original videos,
which are not distributed in the source archive.

For another dataset, supply actual paths:

```powershell
python experiments/scripts/vision_yolo11_stsd.py train --data path/to/drone.yaml --weights path/to/yolo11n.pt --epochs 80 --batch 8 --device 0
python experiments/scripts/vision_yolo11_stsd.py video --weights path/to/stsd-best.pt --source path/to/video.mp4 --run-dir path/to/demo --device 0
```

Convert Anti-UAV annotations to YOLO detection labels, retain video-disjoint splits,
and preserve chronological frame identifiers. Current engineering probes use the
DroneDetectionThesis RGB dataset, **not** the official Anti-UAV benchmark.

## Interpretation

Sky maps are detector-trained attention, not supervised sky segmentation.
Dynamic weights are per image and sum to one. Temporal fusion uses actual paired
training and an explicit detached video cache; image inference is stateless.
There is no motion alignment or optical flow in this version. ONNX/JIT tracing
uses the image path and does not export the Python video cache.

Eleven module/integration checks cover both YAML configurations, real detection
loss, gradients in all three module types, parameter updates, weight transfer,
same-sequence pairs, video cache and checkpoint reload. Run `check` to record
the current result. The authoritative v2 experiment metrics and training budget
are in `experiments/runs/vision_yolo11_stsd_v2/summary.json` and the published v2
report; the old v1 metrics must not be applied to the default v2 model.

Short probes establish engineering feasibility. They do not demonstrate better
accuracy than a completed baseline with a much larger training budget. Ordinary
image validation disables temporal memory; it does not measure the contribution
of the video branch. Subsequent matched-budget comparisons and module ablations
can evaluate accuracy, false alarms and temporal stability without requiring
multi-seed runs during this exploratory stage.
