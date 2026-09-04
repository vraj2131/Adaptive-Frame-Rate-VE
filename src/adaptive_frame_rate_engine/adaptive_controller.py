from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class AdaptiveState:
	target_fps: float = 20.0
	resolution_levels: List[int] = field(default_factory=lambda: [512, 384, 256])
	resolution_index: int = 0  # 0 is highest resolution
	precision: str = "fp32"  # "fp32" or "fp16"
	frame_interval: int = 1  # process every Nth frame


@dataclass
class AdaptiveMetrics:
	measured_fps: float
	latency_ms: float
	gpu_util_percent: float
	gpu_mem_mb: float
	acc_trend: float  # proxy metric; e.g., running mean confidence (0..1)


@dataclass
class AdaptiveDecision:
	changed: bool
	new_precision: Optional[str] = None
	new_resolution_index: Optional[int] = None
	new_frame_interval: Optional[int] = None
	notes: str = ""


class AdaptiveController:
	"""Adaptive controller that tunes precision, input resolution, and frame interval.

	Control logic (heuristic):
	- If GPU util > 80% and device supports it, switch to FP16; if < 50%, revert to FP32.
	- If FPS < target by >10%, reduce resolution one step; if FPS > target+20%, increase resolution one step.
	- If GPU util spikes > 90% or latency > 1.5x median, increase frame_interval (drop frames) up to 4.
	- If GPU util < 50% and FPS >= target, decrease frame_interval toward 1.
	- If accuracy/confidence trend drops strongly (<0.5), prefer increasing resolution (if possible) and avoid FP16 if numerics could hurt.
	"""

	def __init__(self, target_fps: float = 20.0, resolution_levels: Optional[List[int]] = None):
		self.state = AdaptiveState(
			target_fps=target_fps,
			resolution_levels=resolution_levels or [512, 384, 256],
			resolution_index=0,
			precision="fp32",
			frame_interval=1,
		)

	def current_params(self) -> Dict[str, object]:
		return {
			"precision": self.state.precision,
			"resolution": self.state.resolution_levels[self.state.resolution_index],
			"frame_interval": self.state.frame_interval,
		}

	def update(self, metrics: AdaptiveMetrics) -> AdaptiveDecision:
		changed = False
		notes: List[str] = []

		# Precision control
		if metrics.gpu_util_percent > 80.0 and self.state.precision != "fp16":
			self.state.precision = "fp16"
			changed = True
			notes.append("util>80: switch fp16")
		elif metrics.gpu_util_percent < 50.0 and self.state.precision != "fp32":
			self.state.precision = "fp32"
			changed = True
			notes.append("util<50: revert fp32")

		# Resolution control based on FPS gap
		fps = metrics.measured_fps
		if fps > 0:
			if fps < 0.9 * self.state.target_fps and self.state.resolution_index < len(self.state.resolution_levels) - 1:
				self.state.resolution_index += 1
				changed = True
				notes.append("fps low: downscale")
			elif fps > 1.2 * self.state.target_fps and self.state.resolution_index > 0:
				self.state.resolution_index -= 1
				changed = True
				notes.append("fps high: upscale")

		# Frame interval (frame dropping) control
		if metrics.gpu_util_percent > 90.0 or metrics.latency_ms > 75.0:
			if self.state.frame_interval < 4:
				self.state.frame_interval += 1
				changed = True
				notes.append("spike: increase frame interval")
		elif metrics.gpu_util_percent < 50.0 and fps >= self.state.target_fps and self.state.frame_interval > 1:
			self.state.frame_interval -= 1
			changed = True
			notes.append("low util: decrease frame interval")

		# Accuracy/confidence guardrail
		if metrics.acc_trend < 0.5:
			# Prefer better fidelity when confidence low
			if self.state.resolution_index > 0:
				self.state.resolution_index -= 1
				changed = True
				notes.append("low confidence: upscale")
			if self.state.precision == "fp16":
				self.state.precision = "fp32"
				changed = True
				notes.append("low confidence: revert fp32")

		return AdaptiveDecision(
			changed=changed,
			new_precision=self.state.precision,
			new_resolution_index=self.state.resolution_index,
			new_frame_interval=self.state.frame_interval,
			notes=", ".join(notes) if notes else "stable",
		)
