import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
from torchvision import transforms

# Optional NVML for GPU utilization
try:
	import pynvml  # type: ignore
	has_nvml = True
except Exception:
	has_nvml = False

try:
	from medmnist import INFO, Evaluator
	from medmnist import OrganMNISTAxial
	exists_medmnist = True
except Exception:
	exists_medmnist = False
	OrganMNISTAxial = None  # type: ignore
	INFO = {}
	Evaluator = None  # type: ignore


def get_device(prefer_gpu: bool = True) -> torch.device:
	if prefer_gpu and torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def get_gpu_stats() -> Tuple[float, float]:
	"""Return (utilization_percent, memory_used_mb) for GPU 0 if available, else (0,0)."""
	if not torch.cuda.is_available():
		return 0.0, 0.0
	util = 0.0
	mem_mb = 0.0
	try:
		mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
		if has_nvml:
			pynvml.nvmlInit()
			handle = pynvml.nvmlDeviceGetHandleByIndex(0)
			util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
			util = float(util_info.gpu)
			pynvml.nvmlShutdown()
	except Exception:
		pass
	return util, mem_mb


def ensure_data(root: Path, dataset: str = "organmnist_axial", download: bool = True):
	"""Ensure MedMNIST dataset is available; returns train and test datasets.

	Parameters:
	- root: base directory for datasets
	- dataset: medmnist dataset key, default organmnist_axial
	- download: whether to download if missing
	"""
	root.mkdir(parents=True, exist_ok=True)
	if not exists_medmnist:
		raise RuntimeError("medmnist is not installed. Please install medmnist to use sample medical images.")

	if dataset != "organmnist_axial":
		raise ValueError("Currently only 'organmnist_axial' is supported in this scaffold.")

	from medmnist import OrganMNISTAxial

	data_flag = "organmnist_axial"
	info = INFO[data_flag]
	num_classes = len(info["label"])

	common_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	train_dataset = OrganMNISTAxial(split="train", root=str(root), download=download, transform=common_transform)
	test_dataset = OrganMNISTAxial(split="test", root=str(root), download=download, transform=common_transform)
	return train_dataset, test_dataset, num_classes


def default_image_transform() -> transforms.Compose:
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


def set_torch_deterministic(seed: int = 42):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True
