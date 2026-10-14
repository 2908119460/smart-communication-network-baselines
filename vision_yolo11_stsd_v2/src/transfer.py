"""Semantic pretrained transfer and explicit stream-state control."""
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import TemporalFeatureModule

def temporal_modules(model):
    """Accept a YOLO wrapper or a torch DetectionModel."""
    root = model.model if isinstance(model, YOLO) else model
    return [m for m in root.modules() if isinstance(m, TemporalFeatureModule)]

def set_video_mode(model, enabled=True):
    """Start a new stream; image inference remains stateless by default."""
    modules = temporal_modules(model)
    for module in modules:
        module.enable_temporal(enabled)
    return len(modules)

def load_baseline_weights(model, checkpoint):
    """Transfer matching YOLO11 layers using the explicit YAML layer map.

    New modules retain their initialization. Head tensors transfer only where
    shapes match. YOLO.load(baseline.pt) alone cannot map shifted layer indices.
    """
    target = model.model if isinstance(model, YOLO) else model
    source = YOLO(str(checkpoint)).model
    mapping = target.yaml.get("stsd_base_layer_map")
    if not mapping:
        raise ValueError("Missing stsd_base_layer_map in model YAML")
    source_state, target_state = source.state_dict(), target.state_dict()
    transferred, skipped, loaded = {}, [], []
    for key, value in source_state.items():
        parts = key.split(".", 2)
        if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
            continue
        old_index = int(parts[1])
        if old_index not in mapping:
            skipped.append(key)
            continue
        new_key = f"model.{mapping[old_index]}.{parts[2]}"
        if new_key in target_state and target_state[new_key].shape == value.shape:
            transferred[new_key] = value.detach().clone()
            loaded.append({"source": key, "target": new_key})
        else:
            skipped.append(key)
    target.load_state_dict(transferred, strict=False)
    state = target.state_dict()
    assert all(torch.equal(state[k].cpu(), v.cpu().to(state[k].dtype)) for k, v in transferred.items())
    if isinstance(model, YOLO):
        # Model.train() forwards existing weights only when ckpt is truthy.
        # This is an initialized STSD checkpoint, not a baseline resume state.
        model.ckpt = {"model": target, "stsd_transfer_source": str(checkpoint), "epoch": -1}
    return {"checkpoint": str(checkpoint), "transferred_tensors": len(transferred),
            "source_tensors": len(source_state), "skipped_tensors": len(skipped),
            "layer_map": mapping, "loaded": loaded, "skipped": skipped}
