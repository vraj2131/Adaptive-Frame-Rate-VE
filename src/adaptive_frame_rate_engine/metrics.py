import os
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Iterable

import torch


@dataclass
class FpsMeter:
	window: int = 50

	def __post_init__(self):
		self.timestamps = []

	def tick(self):
		self.timestamps.append(time.perf_counter())
		if len(self.timestamps) > self.window:
			self.timestamps.pop(0)

	def fps(self) -> float:
		if len(self.timestamps) < 2:
			return 0.0
		delta = self.timestamps[-1] - self.timestamps[0]
		if delta <= 0:
			return 0.0
		return (len(self.timestamps) - 1) / delta


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
	if logits.numel() == 0:
		return 0.0
	pred = torch.argmax(logits, dim=1)
	correct = (pred == labels.to(logits.device)).float().sum().item()
	return float(correct) / float(labels.shape[0])


class MetricsLogger:
	"""CSV logger for per-frame inference metrics.

	Columns:
	- ts_iso, mode, batch_idx, frame_idx, gpu_mem_mb, gpu_util, latency_ms, fps, resolution, precision, pred_label, confidence
	"""

	HEADER = [
		"ts_iso",
		"mode",
		"batch_idx",
		"frame_idx",
		"gpu_mem_mb",
		"gpu_util",
		"latency_ms",
		"fps",
		"resolution",
		"precision",
		"pred_label",
		"confidence",
	]

	def __init__(self, logs_dir: str = "./logs", run_name: Optional[str] = None, mode: str = "afre"):
		self.logs_dir = Path(logs_dir)
		self.logs_dir.mkdir(parents=True, exist_ok=True)
		name = run_name or time.strftime("%Y%m%d-%H%M%S")
		self.csv_path = self.logs_dir / f"{name}_{mode}.csv"
		self.mode = mode
		self._ensure_header()

	def _ensure_header(self):
		if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
			with self.csv_path.open("w", newline="") as f:
				writer = csv.writer(f)
				writer.writerow(self.HEADER)

	def log_frame(self, *, batch_idx: int, frame_idx: int, gpu_mem_mb: float, gpu_util: float, latency_ms: float, fps: float, resolution: int, precision: str, pred_label: int, confidence: float):
		with self.csv_path.open("a", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				time.strftime("%Y-%m-%dT%H:%M:%S"),
				self.mode,
				batch_idx,
				frame_idx,
				float(gpu_mem_mb),
				float(gpu_util),
				float(latency_ms),
				float(fps),
				int(resolution),
				precision,
				int(pred_label),
				float(confidence),
			])

	def path(self) -> Path:
		return self.csv_path


def aggregate_metrics(rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
	"""Compute aggregates over logged rows. Expects rows with MetricsLogger.HEADER keys."""
	n = 0
	sum_fps = 0.0
	sum_lat = 0.0
	sum_util = 0.0
	sum_conf = 0.0
	for r in rows:
		n += 1
		sum_fps += float(r["fps"]) if r.get("fps") else 0.0
		sum_lat += float(r["latency_ms"]) if r.get("latency_ms") else 0.0
		sum_util += float(r["gpu_util"]) if r.get("gpu_util") else 0.0
		sum_conf += float(r["confidence"]) if r.get("confidence") else 0.0
	if n == 0:
		return {"mean_fps": 0.0, "avg_latency_ms": 0.0, "avg_gpu_util": 0.0, "avg_confidence": 0.0}
	return {
		"mean_fps": sum_fps / n,
		"avg_latency_ms": sum_lat / n,
		"avg_gpu_util": sum_util / n,
		"avg_confidence": sum_conf / n,
	}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
	if not path.exists():
		return []
	with path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		return [row for row in reader]


def compare_accuracy_vs_baseline(afre_rows: List[Dict[str, str]], baseline_rows: List[Dict[str, str]]) -> float:
	"""Return AFRE accuracy difference (%) vs baseline using logged per-frame labels if both contain truth.
	This scaffold assumes labels are unknown, so we approximate by confidence trend difference (0..1 -> percent)."""
	if not afre_rows or not baseline_rows:
		return 0.0
	af_conf = sum(float(r.get("confidence", 0.0)) for r in afre_rows) / len(afre_rows)
	bl_conf = sum(float(r.get("confidence", 0.0)) for r in baseline_rows) / len(baseline_rows)
	# Approximate accuracy delta as confidence delta in percent
	return (af_conf - bl_conf) * 100.0
