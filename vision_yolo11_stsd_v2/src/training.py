"""Adjacent-frame batches with the original Detect head and detection loss.

Pairs are shuffled as units. Images remain BCHW; targets keep the Ultralytics
format. There is no training cache across batches. Each frame retains its loss.
"""
from collections import defaultdict
from copy import copy
from pathlib import Path
import re

from torch.utils.data import Dataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model
from .transfer import temporal_modules

class TemporalPairDataset(Dataset):
    """Same-split chronological pairs from <sequence>_<frame-number>.jpg.

    Parent path also scopes sequence identity. Explicit index tuples in ``pairs``
    support other naming conventions. max_frame_gap bounds sampled-frame gaps.
    """
    def __init__(self, dataset, pairs=None, max_frame_gap=32):
        self.dataset = dataset
        if pairs is None:
            groups = defaultdict(list)
            for index, filename in enumerate(dataset.im_files):
                path = Path(filename)
                match = re.fullmatch(r"(.+)_(\d+)", path.stem)
                if match:
                    sequence, frame = match[1], int(match[2])
                elif path.stem.isdigit():
                    # Anti-UAV-style per-sequence folders with numbered frames.
                    sequence, frame = path.parent.name, int(path.stem)
                else:
                    raise ValueError(f"Cannot infer sequence/frame from {filename}; supply explicit pairs")
                groups[(str(path.parent), sequence)].append((frame, index))
            pairs, gaps = [], []
            for frames in groups.values():
                frames.sort()
                for (previous_frame, previous), (current_frame, current) in zip(frames, frames[1:]):
                    gap = current_frame - previous_frame
                    if 0 < gap <= max_frame_gap:
                        pairs.append((previous, current))
                        gaps.append(gap)
            self.pair_statistics = {"pairs": len(pairs), "sequences": len(groups),
                                    "frame_gap_min": min(gaps) if gaps else None,
                                    "frame_gap_max": max(gaps) if gaps else None,
                                    "max_frame_gap": max_frame_gap}
        else:
            self.pair_statistics = {"pairs": len(pairs), "explicit_pair_indices": True}
        if not pairs:
            raise ValueError("No valid temporal pairs; use sequence filenames or explicit pairs")
        if any(a == b or min(a, b) < 0 or max(a, b) >= len(dataset) for a, b in pairs):
            raise ValueError("Invalid pair indices")
        self.pairs = list(pairs)
        self.rect = False

    @property
    def labels(self):
        return self.dataset.labels

    @property
    def im_files(self):
        return self.dataset.im_files

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        previous, current = self.pairs[index]
        return self.dataset[previous], self.dataset[current]

    @staticmethod
    def collate_fn(pairs):
        from ultralytics.data.dataset import YOLODataset
        return YOLODataset.collate_fn([image for pair in pairs for image in pair])

    def close_mosaic(self, hyp):
        return None

class STSDPairTrainer(DetectionTrainer):
    """Optional temporal trainer. DetectionModel, loss and optimizer unchanged.

    batch=8 means four pairs/eight images per forward. Validation is ordinary
    stateless image evaluation; temporal evaluation uses ordered videos.
    Augmentations are disabled here to retain aligned pair coordinates. This
    differs from baseline augmentation, so a short probe is not a benchmark.
    """
    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        cfg = copy(self.args)
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        # mode=val + rect=False gives square deterministic letterboxing.
        dataset = build_yolo_dataset(cfg, img_path, batch, self.data, mode="val", rect=False, stride=gs)
        paired = TemporalPairDataset(dataset, max_frame_gap=int(self.data.get("temporal_max_frame_gap", 32)))
        LOGGER.info(f"STSD temporal pairs: {paired.pair_statistics}")
        return paired

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Per-device batch must be even and >=2 for temporal pairs")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        return build_dataloader(dataset, batch=batch_size // 2, workers=self.args.workers,
                                shuffle=True, rank=rank, drop_last=False)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        for module in temporal_modules(unwrap_model(self.model)):
            module.set_pair_training(True)
        return batch
