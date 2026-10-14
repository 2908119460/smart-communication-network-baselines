"""CPU-first module and real Ultralytics integration checks for YOLO11-STSD.

Run directly with the workspace vision environment. ``STSD_TEST_DEVICE`` may
select a CUDA device, but all tests default to CPU and use small feature maps.
These checks verify implementation and trainability, not detection accuracy.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

FORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FORK_ROOT.parents[2]
sys.path.insert(0, str(FORK_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE_ROOT / "experiments" / "_readonly_ultra_cfg"))

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Concat, Detect, DynamicScaleNeck, SkyPriorModule, TemporalFeatureModule
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from stsd_tools.training import TemporalPairDataset
from stsd_tools.transfer import load_baseline_weights

torch.set_num_threads(2)
TEST_DEVICE = torch.device(os.environ.get("STSD_TEST_DEVICE", "cpu"))
MODEL_YAML = FORK_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11n-stsd.yaml"
PAN_MODEL_YAML = MODEL_YAML.with_name("yolo11n-stsd-pan.yaml")


class STSDModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_sky_shape_attention_and_gradient(self):
        module = SkyPriorModule(16).to(TEST_DEVICE)
        x = torch.randn(2, 16, 16, 20, device=TEST_DEVICE, requires_grad=True)
        output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(tuple(module.attention_map.shape), (2, 1, 16, 20))
        self.assertTrue(torch.isfinite(module.attention_map).all().item())
        self.assertTrue(((module.attention_map >= 0) & (module.attention_map <= 1)).all().item())
        self.assertNotIn("attention_map", module.state_dict())
        output.square().mean().backward()
        self.assertIsNotNone(module.alpha.grad)
        self.assertGreater(module.alpha.grad.abs().item(), 0)
        self.assertIsNotNone(module.attention[0].weight.grad)
        self.assertGreater(module.attention[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_dynamic_three_targets_weights_and_gradient(self):
        features = [
            torch.randn(2, 16, 24, 24, device=TEST_DEVICE),
            torch.randn(2, 32, 12, 12, device=TEST_DEVICE) + 1,
            torch.randn(2, 64, 6, 6, device=TEST_DEVICE) + 2,
        ]
        for target in range(3):
            with self.subTest(target=target):
                module = DynamicScaleNeck([16, 32, 64], target).to(TEST_DEVICE).eval()
                optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
                output = module(features)
                self.assertEqual(output.shape, features[target].shape)
                self.assertEqual(tuple(module.last_weights.shape), (2, 3))
                torch.testing.assert_close(module.last_weights.sum(1), torch.ones(2, device=TEST_DEVICE))
                torch.testing.assert_close(module.last_weights, torch.full((2, 3), 1 / 3, device=TEST_DEVICE))
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    output = module(features)
                    output.square().mean().backward()
                    self.assertGreater(module.scale_weights[-2].weight.grad.abs().sum().item(), 0)
                    self.assertGreater(module.projections[0].conv.weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                module(features)
                original_weights = module.last_weights.clone()
                shifted = [feature + (i + 1) * 3 for i, feature in enumerate(features)]
                module(shifted)
                self.assertGreater((original_weights - module.last_weights).abs().max().item(), 1e-6)
                self.assertTrue(torch.isfinite(module.last_weights).all().item())
                torch.testing.assert_close(module.last_weights.sum(1), torch.ones(2, device=TEST_DEVICE))
                self.assertNotIn("last_weights", module.state_dict())

    def test_dynamic_backbone_to_original_head_widths(self):
        # The strict neck reads backbone channels [128, 128, 256], whereas
        # the unchanged nano Detect head expects [64, 128, 256].
        features = [
            torch.randn(2, 128, 12, 16, device=TEST_DEVICE),
            torch.randn(2, 128, 6, 8, device=TEST_DEVICE),
            torch.randn(2, 256, 3, 4, device=TEST_DEVICE),
        ]
        for target, width in enumerate((64, 128, 256)):
            with self.subTest(target=target, out_channels=width):
                module = DynamicScaleNeck([128, 128, 256], target, out_channels=width).to(TEST_DEVICE)
                output = module(features)
                self.assertEqual(tuple(output.shape), (2, width, *features[target].shape[-2:]))
                output.square().mean().backward()
                for projection in module.projections:
                    self.assertTrue(torch.isfinite(projection.conv.weight.grad).all().item())
                    self.assertGreater(projection.conv.weight.grad.abs().sum().item(), 0)
                self.assertGreater(module.scale_weights[-2].weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "out_channels"):
            DynamicScaleNeck([128, 128, 256], out_channels=0)

    def test_temporal_pair_gradient_and_shuffled_training(self):
        module = TemporalFeatureModule(16).to(TEST_DEVICE)
        x = torch.randn(4, 16, 12, 12, device=TEST_DEVICE, requires_grad=True)
        self.assertIs(module(x), x)
        self.assertIsNone(module._previous)
        module.set_pair_training(True)
        output = module(x)
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output[0::2], x[0::2])
        self.assertGreater((output[1::2] - x[1::2]).abs().max().item(), 0)
        output.square().mean().backward()
        self.assertGreater(module.fusion.conv.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(module.gamma.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertIsNone(module._previous)
        with self.assertRaisesRegex(ValueError, "even"):
            module(x[:3])

    def test_temporal_video_cache_isolation_and_serialization(self):
        module = TemporalFeatureModule(16).to(TEST_DEVICE).eval()
        x = torch.randn(2, 16, 12, 12, device=TEST_DEVICE)
        self.assertIs(module(x), x)
        unrelated_image = x + 1
        self.assertIs(module(unrelated_image), unrelated_image)
        self.assertIsNone(module._previous)
        module.enable_temporal(True)
        self.assertIs(module(x), x)
        self.assertFalse(module.last_used_memory)
        current = x + 1
        output = module(current)
        self.assertTrue(module.last_used_memory)
        self.assertEqual(output.shape, x.shape)
        self.assertGreater((output - current).abs().max().item(), 0)
        self.assertFalse(module._previous.requires_grad)
        self.assertIsNone(module._previous.grad_fn)
        self.assertNotEqual(module._previous.data_ptr(), current.data_ptr())
        self.assertNotIn("_previous", module.state_dict())
        cloned = copy.deepcopy(module)
        self.assertIsNone(cloned._previous)
        self.assertFalse(cloned.last_used_memory)
        self.assertFalse(cloned.temporal_enabled)
        self.assertFalse(cloned.training_pair_mode)
        self.assertIs(cloned(current), current)
        resized = torch.randn(2, 16, 10, 10, device=TEST_DEVICE)
        self.assertIs(module(resized), resized)
        self.assertFalse(module.last_used_memory)
        # A precision change supplied by the caller also invalidates cache.
        as_double = resized.double()
        self.assertIs(module(as_double), as_double)
        self.assertFalse(module.last_used_memory)
        # Meta device exercises the device guard without requiring a GPU.
        other_device = torch.empty_like(as_double, device="meta")
        self.assertIs(module(other_device), other_device)
        self.assertFalse(module.last_used_memory)
        module.double()
        self.assertIsNone(module._previous)
        module(as_double)
        module.to(TEST_DEVICE)
        self.assertIsNone(module._previous)
        module(as_double)
        module.train()
        self.assertIsNone(module._previous)
        module.eval().enable_temporal(False)
        self.assertIs(module(as_double), as_double)
        self.assertIsNone(module._previous)


class STSDIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_yaml_detect_pyramid_and_real_detection_loss_step(self):
        for yaml_path in (MODEL_YAML, PAN_MODEL_YAML):
            with self.subTest(yaml=yaml_path.name):
                model = DetectionModel(str(yaml_path), nc=1, verbose=False).to(TEST_DEVICE)
                model.args = get_cfg()
                self.assertIsInstance(model.model[-1], Detect)
                torch.testing.assert_close(model.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
                self.assertEqual(sum(isinstance(m, SkyPriorModule) for m in model.modules()), 3)
                self.assertEqual(sum(isinstance(m, DynamicScaleNeck) for m in model.modules()), 3)
                self.assertEqual(sum(isinstance(m, TemporalFeatureModule) for m in model.modules()), 3)
                if yaml_path == MODEL_YAML:
                    # Dynamic fusion must replace the fixed FPN/PAN path,
                    # rather than being appended to it after its C3k2 blocks.
                    self.assertEqual(len(model.model), 21)
                    self.assertFalse(any(isinstance(m, (Concat, torch.nn.Upsample)) for m in model.model))
                    self.assertEqual([m.f for m in model.model[14:17]], [[11, 12, 13]] * 3)
                    self.assertEqual([m.out_channels for m in model.model[14:17]], [64, 128, 256])
                    self.assertEqual(model.model[-1].f, [17, 18, 19])
                else:
                    self.assertEqual(len(model.model), 33)
                    self.assertTrue(any(isinstance(m, Concat) for m in model.model))
                    self.assertTrue(any(isinstance(m, torch.nn.Upsample) for m in model.model))
                model.train()
                for module in model.modules():
                    if isinstance(module, TemporalFeatureModule):
                        module.set_pair_training(True)
                img = torch.rand(2, 3, 96, 96, device=TEST_DEVICE)
                head_inputs = []

                def capture_head_inputs(module, inputs):
                    head_inputs.append([tuple(feature.shape) for feature in inputs[0]])

                handle = model.model[-1].register_forward_pre_hook(capture_head_inputs)
                predictions = model(img)
                handle.remove()
                self.assertEqual(head_inputs, [[(2, 64, 12, 12), (2, 128, 6, 6), (2, 256, 3, 3)]])
                self.assertEqual([tuple(p.shape[-2:]) for p in predictions], [(12, 12), (6, 6), (3, 3)])
                batch = {
                    "img": img,
                    "batch_idx": torch.tensor([0, 1], device=TEST_DEVICE),
                    "cls": torch.zeros((2, 1), device=TEST_DEVICE),
                    "bboxes": torch.tensor([[0.50, 0.50, 0.35, 0.35], [0.54, 0.52, 0.35, 0.35]], device=TEST_DEVICE),
                }
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                sky = next(m for m in model.modules() if isinstance(m, SkyPriorModule))
                before_alpha = sky.alpha.detach().clone()
                loss, loss_items = model.loss(batch, predictions)
                self.assertTrue(torch.isfinite(loss).all().item())
                self.assertTrue(torch.isfinite(loss_items).all().item())
                self.assertGreater(loss.sum().item(), 0)
                optimizer.zero_grad(set_to_none=True)
                loss.sum().backward()
                for module in model.modules():
                    if isinstance(module, SkyPriorModule):
                        self.assertIsNotNone(module.alpha.grad)
                        self.assertTrue(torch.isfinite(module.alpha.grad).all().item())
                        self.assertGreater(module.attention[0].weight.grad.abs().sum().item(), 0)
                    elif isinstance(module, DynamicScaleNeck):
                        self.assertIsNotNone(module.scale_weights[-2].weight.grad)
                        self.assertGreater(module.scale_weights[-2].weight.grad.abs().sum().item(), 0)
                        for projection in module.projections:
                            self.assertTrue(torch.isfinite(projection.conv.weight.grad).all().item())
                            self.assertGreater(projection.conv.weight.grad.abs().sum().item(), 0)
                    elif isinstance(module, TemporalFeatureModule):
                        self.assertIsNotNone(module.fusion.conv.weight.grad)
                        self.assertGreater(module.fusion.conv.weight.grad.abs().sum().item(), 0)
                        self.assertTrue(torch.isfinite(module.gamma.grad).all().item())
                optimizer.step()
                self.assertGreater((sky.alpha.detach() - before_alpha).abs().item(), 0)

    def test_highlevel_checkpoint_save_reload_and_image_predict(self):
        model = YOLO(str(MODEL_YAML), task="detect", verbose=False)
        model.model.names = {0: "drone"}
        model.model.eval()
        for module in model.model.modules():
            if isinstance(module, TemporalFeatureModule):
                module.enable_temporal(True)
                module.set_pair_training(True)
        with torch.no_grad():
            model.model(torch.rand(1, 3, 96, 96))
        # The full model checkpoint must omit the transient video feature.
        with tempfile.TemporaryDirectory(prefix="stsd_checkpoint_", dir=WORKSPACE_ROOT / "experiments" / "runs") as directory:
            checkpoint = Path(directory) / "stsd.pt"
            model.save(checkpoint)
            loaded = YOLO(str(checkpoint), task="detect", verbose=False)
            temporal_modules = [m for m in loaded.model.modules() if isinstance(m, TemporalFeatureModule)]
            self.assertEqual(len(temporal_modules), 3)
            self.assertTrue(all(m._previous is None for m in temporal_modules))
            self.assertTrue(all(not m.temporal_enabled and not m.training_pair_mode for m in temporal_modules))
            results = loaded.predict(np.zeros((96, 96, 3), dtype=np.uint8), imgsz=96, device=str(TEST_DEVICE), verbose=False)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].orig_shape, (96, 96))
            self.assertTrue(all(m._previous is None for m in temporal_modules))

    def test_strict_model_video_memory_and_single_image_reset(self):
        model = DetectionModel(str(MODEL_YAML), nc=1, verbose=False).to(TEST_DEVICE).eval()
        temporal = [m for m in model.modules() if isinstance(m, TemporalFeatureModule)]
        for module in temporal:
            module.enable_temporal(True)
        previous = torch.rand(1, 3, 96, 96, device=TEST_DEVICE)
        current = torch.rand_like(previous)
        with torch.no_grad():
            model(previous)
            self.assertTrue(all(m._previous is not None for m in temporal))
            self.assertTrue(all(not m.last_used_memory for m in temporal))
            video_output = model(current)[0]
            self.assertTrue(all(m.last_used_memory for m in temporal))
            self.assertTrue(all(m._previous.grad_fn is None for m in temporal))
            self.assertTrue(torch.isfinite(video_output).all().item())
            for module in temporal:
                module.enable_temporal(False)
            image_output = model(current)[0]
            repeated_image_output = model(current)[0]
        self.assertTrue(all(m._previous is None and not m.last_used_memory for m in temporal))
        torch.testing.assert_close(image_output, repeated_image_output, rtol=0, atol=0)

    def test_pair_dataset_chronology_split_scope_and_collated_targets(self):
        class SyntheticFrames:
            # Intentionally shuffled; seqA occurs in both split directories.
            im_files = [
                "train/seqA_000100.jpg", "train/seqA_000000.jpg", "train/seqB_000002.jpg",
                "validation/seqA_000003.jpg", "train/seqA_000002.jpg", "train/seqB_000001.jpg",
                "validation/seqA_000004.jpg",
            ]
            labels = [{} for _ in im_files]

            def __len__(self):
                return len(self.im_files)

            def __getitem__(self, index):
                count = 0 if index == 1 else 2 if index == 4 else 1
                return {
                    "img": torch.full((3, 8, 8), index, dtype=torch.uint8),
                    "cls": torch.full((count, 1), float(index)),
                    "bboxes": torch.full((count, 4), 0.25),
                    "batch_idx": torch.zeros(count),
                    "im_file": self.im_files[index],
                }

        base = SyntheticFrames()
        paired = TemporalPairDataset(base, max_frame_gap=8)
        self.assertEqual(set(paired.pairs), {(1, 4), (5, 2), (3, 6)})
        for previous, current in paired.pairs:
            prev_path, current_path = Path(base.im_files[previous]), Path(base.im_files[current])
            self.assertEqual(prev_path.parent, current_path.parent)
            prev_sequence, prev_frame = prev_path.stem.rsplit("_", 1)
            current_sequence, current_frame = current_path.stem.rsplit("_", 1)
            self.assertEqual(prev_sequence, current_sequence)
            self.assertLess(int(prev_frame), int(current_frame))
        samples = [paired[paired.pairs.index((1, 4))], paired[paired.pairs.index((5, 2))]]
        batch = paired.collate_fn(samples)
        self.assertEqual(batch["img"].shape[0] % 2, 0)
        self.assertEqual(batch["img"][:, 0, 0, 0].tolist(), [1, 4, 5, 2])
        self.assertEqual(batch["batch_idx"].tolist(), [1, 1, 2, 3])
        self.assertEqual(batch["cls"].flatten().tolist(), [4, 4, 5, 2])
        self.assertEqual(len(batch["cls"]), len(batch["bboxes"]))
        self.assertEqual(batch["im_file"], tuple(base.im_files[i] for i in [1, 4, 5, 2]))

    def test_disjoint_pair_stride_preserves_one_exposure_per_frame(self):
        class Frames:
            im_files = [f"train/video_{i:06d}.jpg" for i in range(0, 120, 15)]
            labels = [{} for _ in im_files]

            def __len__(self):
                return len(self.im_files)

        paired = TemporalPairDataset(Frames(), pair_stride=2)
        self.assertEqual(paired.pairs, [(0, 1), (2, 3), (4, 5), (6, 7)])
        self.assertEqual(paired.pair_statistics["unique_frames_used"], 8)
        self.assertEqual(len({i for pair in paired.pairs for i in pair}), 2 * len(paired))
        with self.assertRaises(ValueError):
            TemporalPairDataset(Frames(), pair_stride=0)

    def test_explicit_baseline_transfer_and_removed_neck_is_not_reused(self):
        baseline_yaml = MODEL_YAML.with_name("yolo11.yaml")
        baseline = YOLO(str(baseline_yaml), task="detect", verbose=False)
        baseline.model = DetectionModel(str(baseline_yaml), nc=1, verbose=False)
        # Binary fractions survive checkpoint fp16 storage exactly. Different
        # layer markers detect accidental transfer to a shifted neck/head index.
        with torch.no_grad():
            for key, value in baseline.model.state_dict().items():
                layer_index = int(key.split(".")[1])
                value.fill_((layer_index + 1) / 64 if value.is_floating_point() else layer_index + 1)
        with tempfile.TemporaryDirectory(prefix="stsd_transfer_", dir=WORKSPACE_ROOT / "experiments" / "runs") as directory:
            checkpoint = Path(directory) / "baseline.pt"
            baseline.save(checkpoint)
            source_state = YOLO(str(checkpoint), task="detect", verbose=False).model.state_dict()
            for yaml_path in (MODEL_YAML, PAN_MODEL_YAML):
                with self.subTest(yaml=yaml_path.name):
                    target = YOLO(str(yaml_path), task="detect", verbose=False)
                    mapping = target.model.yaml["stsd_base_layer_map"]
                    mapped_layers = set(mapping.values())
                    untouched = {
                        key: value.clone() for key, value in target.model.state_dict().items()
                        if int(key.split(".")[1]) not in mapped_layers
                    }
                    report = load_baseline_weights(target, checkpoint)
                    target_state = target.model.state_dict()
                    self.assertTrue(target.ckpt, "YOLO.train() discards mapped weights when ckpt is falsey.")
                    expected_loaded = {key for key in source_state if int(key.split(".")[1]) in mapping}
                    expected_skipped = set(source_state) - expected_loaded
                    self.assertEqual(report["transferred_tensors"], len(expected_loaded))
                    self.assertEqual(report["skipped_tensors"], len(expected_skipped))
                    self.assertEqual(set(report["skipped"]), expected_skipped)
                    self.assertEqual(mapping[0], 0)  # First backbone layer.
                    if yaml_path == MODEL_YAML:
                        self.assertEqual(mapping, {**{i: i for i in range(11)}, 23: 20})
                        self.assertTrue(expected_skipped)
                        self.assertTrue(all(11 <= int(key.split(".")[1]) <= 22 for key in expected_skipped))
                    else:
                        self.assertEqual(mapping[11], 14)  # Preserved FPN/PAN prototype.
                        self.assertEqual(mapping[23], 32)
                        self.assertFalse(expected_skipped)
                    for key in expected_loaded:
                        value = source_state[key]
                        _, source_index, suffix = key.split(".", 2)
                        target_key = f"model.{mapping[int(source_index)]}.{suffix}"
                        torch.testing.assert_close(target_state[target_key], value.to(target_state[target_key].dtype), rtol=0, atol=0)
                    self.assertTrue(untouched)
                    for key, value in untouched.items():
                        torch.testing.assert_close(target_state[key], value, rtol=0, atol=0)
                    # Exercise the same reconstruction used by YOLO.train(),
                    # without creating a run or reading an image dataset.
                    trainer = DetectionTrainer.__new__(DetectionTrainer)
                    trainer.data = {"nc": 1, "channels": 3}
                    rebuilt = trainer.get_model(cfg=target.model.yaml, weights=target.model if target.ckpt else None, verbose=False)
                    for key, value in target_state.items():
                        torch.testing.assert_close(rebuilt.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
