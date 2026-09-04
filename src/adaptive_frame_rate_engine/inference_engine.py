from typing import Tuple, List, Optional, Dict
import os
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

from .adaptive_controller import AdaptiveController, AdaptiveMetrics
from .metrics import FpsMeter, top1_accuracy, MetricsLogger


class InferenceEngine:
	def __init__(self, model_name: str, num_classes: int, device: torch.device, pretrained: bool = True):
		self.device = device
		self.model, self.input_size = self._build_model(model_name, num_classes, pretrained)
		self.model.to(self.device)
		self.model.eval()
		self.current_precision = "fp32"  # "fp32" or "fp16"

	@staticmethod
	def _build_model(model_name: str, num_classes: int, pretrained: bool) -> Tuple[nn.Module, int]:
		name = model_name.lower()
		if name == "resnet50":
			weights = models.ResNet50_Weights.DEFAULT if pretrained else None
			model = models.resnet50(weights=weights)
			model.fc = nn.Linear(model.fc.in_features, num_classes)
			return model, 224
		elif name in ("mobilenet_v3_small", "mobilenetv3", "mobilenetv3_small"):
			weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
			model = models.mobilenet_v3_small(weights=weights)
			model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
			return model, 224
		else:
			raise ValueError(f"Unsupported model: {model_name}")

	@torch.no_grad()
	def predict(self, batch: torch.Tensor) -> torch.Tensor:
		batch = batch.to(self.device, non_blocking=True)
		if self.current_precision == "fp16" and self.device.type == "cuda":
			with torch.autocast(device_type="cuda", dtype=torch.float16):
				logits = self.model(batch)
				return logits
		else:
			logits = self.model(batch)
			return logits

	@torch.no_grad()
	def predict_labels(self, batch: torch.Tensor) -> torch.Tensor:
		logits = self.predict(batch)
		return torch.argmax(logits, dim=1)

	def _build_transform(self, size: int) -> transforms.Compose:
		return transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((size, size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

	def process_directory_afre(
		self,
		frames_dir: str,
		controller: AdaptiveController,
		target_fps: float = 20.0,
		initial_res_levels: Optional[List[int]] = None,
		batch_size: int = 8,
		label: Optional[int] = None,
		logger: Optional[MetricsLogger] = None,
	) -> Tuple[List[torch.Tensor], Dict[str, float]]:
		"""Run AFRE on a directory of frames with CSV logging."""
		frames = sorted([str(Path(frames_dir) / f) for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
		if not frames:
			raise ValueError("No image frames found in directory.")

		logger = logger or MetricsLogger(mode="afre")
		res_levels = initial_res_levels or [512, 384, 256]
		fps_meter = FpsMeter(window=50)
		processed_batches: List[torch.Tensor] = []
		latencies: List[float] = []
		conf_history: List[float] = []

		frame_interval = 1
		resolution_index = 0
		self.current_precision = "fp32"

		transform = self._build_transform(res_levels[resolution_index])

		def load_frame(path: str):
			img = cv2.imread(path, cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img

		batch_images: List[torch.Tensor] = []
		start_time = time.perf_counter()

		for i, path in enumerate(frames):
			if (i % frame_interval) != 0:
				continue

			img = load_frame(path)
			tensor = transform(img)
			batch_images.append(tensor)

			if len(batch_images) < batch_size and i < len(frames) - 1:
				continue

			batch = torch.stack(batch_images, dim=0)
			batch_images = []

			t0 = time.perf_counter()
			logits = self.predict(batch)
			t1 = time.perf_counter()

			batch_latency_ms = (t1 - t0) * 1000.0
			latencies.append(batch_latency_ms)
			fps_meter.tick()
			current_fps = fps_meter.fps()

			with torch.no_grad():
				probs = torch.softmax(logits, dim=1)
				conf_vals, preds = probs.max(dim=1)
				conf = float(conf_vals.mean().item())

			if torch.cuda.is_available():
				mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
				util = min(100.0, max(0.0, (batch_latency_ms / max(1.0, 1000.0 / max(1.0, target_fps))) * 50.0))
			else:
				mem_mb = 0.0
				util = 0.0

			# Log per-frame entries
			for j in range(logits.shape[0]):
				logger.log_frame(
					batch_idx=i,
					frame_idx=j,
					gpu_mem_mb=float(mem_mb),
					gpu_util=float(util),
					latency_ms=float(batch_latency_ms),
					fps=float(current_fps),
					resolution=int(res_levels[resolution_index]),
					precision=self.current_precision,
					pred_label=int(preds[j].item()),
					confidence=float(conf_vals[j].item()),
				)

			conf_history.append(conf)

			metrics = AdaptiveMetrics(
				measured_fps=current_fps,
				latency_ms=batch_latency_ms,
				gpu_util_percent=util,
				gpu_mem_mb=mem_mb,
				acc_trend=sum(conf_history[-10:]) / max(1, len(conf_history[-10:])),
			)

			decision = controller.update(metrics)

			if decision.changed:
				if decision.new_precision and decision.new_precision != self.current_precision:
					self.current_precision = decision.new_precision
					print(f"[AFRE] precision -> {self.current_precision} ({decision.notes})")
				if decision.new_resolution_index is not None and decision.new_resolution_index != resolution_index:
					resolution_index = decision.new_resolution_index
					transform = self._build_transform(res_levels[resolution_index])
					print(f"[AFRE] resolution -> {res_levels[resolution_index]} ({decision.notes})")
				if decision.new_frame_interval is not None and decision.new_frame_interval != frame_interval:
					frame_interval = decision.new_frame_interval
					print(f"[AFRE] frame_interval -> {frame_interval} ({decision.notes})")
			else:
				print(f"[AFRE] stable | fps={current_fps:.1f} util~{util:.0f}% lat={batch_latency_ms:.1f}ms conf={conf:.2f}")

			processed_batches.append(logits.detach().cpu())

		total_time = time.perf_counter() - start_time
		num_frames_processed = sum(b.shape[0] for b in processed_batches)
		avg_fps = num_frames_processed / total_time if total_time > 0 else 0.0
		avg_latency = sum(latencies) / max(1, len(latencies))
		avg_conf = sum(conf_history) / max(1, len(conf_history))

		summary = {
			"avg_fps": float(avg_fps),
			"avg_latency_ms": float(avg_latency),
			"avg_confidence": float(avg_conf),
			"final_precision": self.current_precision,
			"final_resolution": float(res_levels[resolution_index]),
			"final_frame_interval": float(frame_interval),
			"csv_path": str(logger.path()),
		}
		return processed_batches, summary

	@torch.no_grad()
	def process_directory_baseline(
		self,
		frames_dir: str,
		resolution: int = 512,
		precision: str = "fp32",
		batch_size: int = 8,
		logger: Optional[MetricsLogger] = None,
	) -> Tuple[List[torch.Tensor], Dict[str, float]]:
		"""Static baseline pipeline: fixed resolution and precision, with CSV logging."""
		frames = sorted([str(Path(frames_dir) / f) for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
		if not frames:
			raise ValueError("No image frames found in directory.")

		logger = logger or MetricsLogger(mode="baseline")
		transform = self._build_transform(resolution)
		self.current_precision = precision

		fps_meter = FpsMeter(window=50)
		processed_batches: List[torch.Tensor] = []
		latencies: List[float] = []
		conf_history: List[float] = []

		def load_frame(path: str):
			img = cv2.imread(path, cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img

		batch_images: List[torch.Tensor] = []
		start_time = time.perf_counter()

		for i, path in enumerate(frames):
			img = load_frame(path)
			tensor = transform(img)
			batch_images.append(tensor)

			if len(batch_images) < batch_size and i < len(frames) - 1:
				continue

			batch = torch.stack(batch_images, dim=0)
			batch_images = []

			t0 = time.perf_counter()
			logits = self.predict(batch)
			t1 = time.perf_counter()

			batch_latency_ms = (t1 - t0) * 1000.0
			latencies.append(batch_latency_ms)
			fps_meter.tick()
			current_fps = fps_meter.fps()

			probs = torch.softmax(logits, dim=1)
			conf_vals, preds = probs.max(dim=1)
			conf = float(conf_vals.mean().item())

			if torch.cuda.is_available():
				mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
				util = min(100.0, max(0.0, (batch_latency_ms / 50.0)))
			else:
				mem_mb = 0.0
				util = 0.0

			for j in range(logits.shape[0]):
				logger.log_frame(
					batch_idx=i,
					frame_idx=j,
					gpu_mem_mb=float(mem_mb),
					gpu_util=float(util),
					latency_ms=float(batch_latency_ms),
					fps=float(current_fps),
					resolution=int(resolution),
					precision=self.current_precision,
					pred_label=int(preds[j].item()),
					confidence=float(conf_vals[j].item()),
				)

			conf_history.append(conf)
			processed_batches.append(logits.detach().cpu())

		total_time = time.perf_counter() - start_time
		num_frames_processed = sum(b.shape[0] for b in processed_batches)
		avg_fps = num_frames_processed / total_time if total_time > 0 else 0.0
		avg_latency = sum(latencies) / max(1, len(latencies))
		avg_conf = sum(conf_history) / max(1, len(conf_history))

		summary = {
			"avg_fps": float(avg_fps),
			"avg_latency_ms": float(avg_latency),
			"avg_confidence": float(avg_conf),
			"final_precision": self.current_precision,
			"final_resolution": float(resolution),
			"csv_path": str(logger.path()),
		}
		return processed_batches, summary
