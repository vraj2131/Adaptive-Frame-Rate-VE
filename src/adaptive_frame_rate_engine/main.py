import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .utils import get_device, get_gpu_stats, ensure_data, set_torch_deterministic
from .inference_engine import InferenceEngine
from .metrics import FpsMeter, top1_accuracy
from .adaptive_controller import AdaptiveController
from .metrics import MetricsLogger


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Adaptive Frame Rate Engine (AFRE)")
	p.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenet_v3_small"], help="Backbone model")
	p.add_argument("--dataset", type=str, default="organmnist_axial", help="MedMNIST dataset key")
	p.add_argument("--batch-size", type=int, default=16)
	p.add_argument("--num-batches", type=int, default=50)
	p.add_argument("--data-root", type=str, default=str(Path.home() / ".afre" / "data"))
	p.add_argument("--frames-dir", type=str, default="", help="Directory of CT frames to run directory mode")
	p.add_argument("--mode", type=str, default="adaptive", choices=["adaptive", "baseline", "dataset"], help="Run mode")
	p.add_argument("--target-fps", type=float, default=20.0)
	return p.parse_args()


def main():
	args = parse_args()
	set_torch_deterministic()
	device = get_device(prefer_gpu=True)

	engine = None
	controller = AdaptiveController(target_fps=args.target_fps)

	if args.mode in ("adaptive", "baseline") and args.frames_dir:
		# Directory mode
		engine = InferenceEngine(model_name=args.model, num_classes=11, device=device, pretrained=True)
		frames_dir = args.frames_dir
		if args.mode == "adaptive":
			logger = MetricsLogger(mode='afre')
			_, summary = engine.process_directory_afre(frames_dir, controller, batch_size=args.batch_size, logger=logger)
		else:
			logger = MetricsLogger(mode='baseline')
			_, summary = engine.process_directory_baseline(frames_dir, resolution=512, precision='fp32', batch_size=args.batch_size, logger=logger)
		print(summary)
		return

	# Dataset mode (default for quick demo)
	train_ds, test_ds, num_classes = ensure_data(Path(args.data_root), dataset=args.dataset, download=True)
	engine = InferenceEngine(model_name=args.model, num_classes=num_classes, device=device, pretrained=True)
	loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

	fps_meter = FpsMeter(window=50)
	total_correct = 0
	total_seen = 0

	start_time = time.perf_counter()
	for i, batch in enumerate(loader):
		if i >= args.num_batches:
			break
		images = batch[0]
		labels = batch[1].squeeze() if isinstance(batch[1], torch.Tensor) else torch.tensor(batch[1])

		t0 = time.perf_counter()
		logits = engine.predict(images)
		t1 = time.perf_counter()

		acc = top1_accuracy(logits, labels)
		pred = torch.argmax(logits, dim=1).cpu()
		total_correct += int((pred == labels).sum().item())
		total_seen += labels.shape[0]

		fps_meter.tick()
		current_fps = fps_meter.fps()
		_, gpu_mem = get_gpu_stats()

		print(f"Batch {i+1}/{args.num_batches} | fps={current_fps:.1f} | gpu_mem_mb={gpu_mem:.1f} | batch_acc={acc*100:.1f}%")

	total_time = time.perf_counter() - start_time
	overall_acc = (total_correct / max(1, total_seen)) * 100.0
	avg_fps = (total_seen / total_time) if total_time > 0 else 0.0
	print("--- Summary ---")
	print(f"Device: {device}")
	print(f"Model: {args.model} | Dataset: {args.dataset}")
	print(f"Batches: {min(args.num_batches, len(loader))} | Batch size: {args.batch_size}")
	print(f"Overall Acc: {overall_acc:.2f}% | Avg FPS: {avg_fps:.2f}")


if __name__ == "__main__":
	main()
