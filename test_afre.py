import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from adaptive_frame_rate_engine.inference_engine import InferenceEngine
from adaptive_frame_rate_engine.adaptive_controller import AdaptiveController
from adaptive_frame_rate_engine.metrics import MetricsLogger, read_csv_rows, aggregate_metrics, compare_accuracy_vs_baseline


def ensure_dirs():
	Path('plots').mkdir(parents=True, exist_ok=True)
	Path('logs').mkdir(parents=True, exist_ok=True)


def run_benchmark(frames_dir: str, model: str = 'resnet50', num_classes: int = 11, batch_size: int = 8):
	ensure_dirs()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	engine = InferenceEngine(model_name=model, num_classes=num_classes, device=device, pretrained=True)

	# Baseline
	bl_logger = MetricsLogger(mode='baseline')
	bl_t0 = time.perf_counter()
	_, bl_summary = engine.process_directory_baseline(frames_dir, resolution=512, precision='fp32', batch_size=batch_size, logger=bl_logger)
	bl_time = time.perf_counter() - bl_t0

	# Adaptive
	controller = AdaptiveController(target_fps=20.0, resolution_levels=[512,384,256])
	af_logger = MetricsLogger(mode='afre')
	af_t0 = time.perf_counter()
	_, af_summary = engine.process_directory_afre(frames_dir, controller, batch_size=batch_size, logger=af_logger)
	af_time = time.perf_counter() - af_t0

	# Load logs
	af_rows = read_csv_rows(af_logger.path())
	bl_rows = read_csv_rows(bl_logger.path())
	agg_af = aggregate_metrics(af_rows)
	agg_bl = aggregate_metrics(bl_rows)
	acc_delta = compare_accuracy_vs_baseline(af_rows, bl_rows)

	throughput_gain = (agg_af.get('mean_fps', 0.0) / max(1e-6, agg_bl.get('mean_fps', 1e-6)))
	acc_drop = -acc_delta  # if AFRE lower confidence -> negative delta -> drop positive
	gpu_util_reduction = max(0.0, agg_bl.get('avg_gpu_util', 0.0) - agg_af.get('avg_gpu_util', 0.0))

	print('--- Benchmark Results ---')
	print(f"Baseline:   avg_fps={agg_bl.get('mean_fps',0):.2f} avg_lat={agg_bl.get('avg_latency_ms',0):.1f}ms avg_util={agg_bl.get('avg_gpu_util',0):.1f}% time={bl_time:.1f}s")
	print(f"Adaptive:   avg_fps={agg_af.get('mean_fps',0):.2f} avg_lat={agg_af.get('avg_latency_ms',0):.1f}ms avg_util={agg_af.get('avg_gpu_util',0):.1f}% time={af_time:.1f}s")
	print(f"Throughput gain: {throughput_gain:.2f}x | Accuracy drop: {acc_drop:.2f}% | GPU util reduction: {gpu_util_reduction:.1f}%")

	# Plots
	af_df = pd.DataFrame(af_rows)
	bl_df = pd.DataFrame(bl_rows)
	for df in [af_df, bl_df]:
		for col in ['fps','latency_ms','gpu_util','gpu_mem_mb','resolution','confidence']:
			if col in df.columns:
				df[col] = pd.to_numeric(df[col], errors='coerce')

	plt.figure(figsize=(10,4))
	plt.subplot(1,2,1)
	if not af_df.empty:
		af_df.groupby('resolution')['fps'].mean().plot(kind='bar', label='AFRE', alpha=0.7)
	if not bl_df.empty:
		bl_df.groupby('resolution')['fps'].mean().plot(kind='bar', label='Baseline', alpha=0.7)
	plt.title('FPS vs Resolution')
	plt.ylabel('FPS')
	plt.legend()

	plt.subplot(1,2,2)
	if not af_df.empty:
		plt.scatter(af_df['gpu_util'], af_df['fps'], s=8, label='AFRE')
	if not bl_df.empty:
		plt.scatter(bl_df['gpu_util'], bl_df['fps'], s=8, label='Baseline')
	plt.title('Throughput vs GPU Load')
	plt.xlabel('GPU Util (%)')
	plt.ylabel('FPS')
	plt.legend()
	plt.tight_layout()
	plt.savefig('plots/benchmark_plots.png', dpi=160)
	plt.close()

	summary = {
		'baseline': {**bl_summary, **agg_bl, 'total_time_s': bl_time},
		'adaptive': {**af_summary, **agg_af, 'total_time_s': af_time},
		'throughput_gain_x': throughput_gain,
		'accuracy_drop_percent': acc_drop,
		'gpu_util_reduction_percent': gpu_util_reduction,
	}
	with open('metrics_summary.json', 'w') as f:
		json.dump(summary, f, indent=2)

	return summary


if __name__ == '__main__':
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument('--frames-dir', type=str, required=True)
	ap.add_argument('--model', type=str, default='resnet50')
	ap.add_argument('--batch-size', type=int, default=8)
	args = ap.parse_args()
	summary = run_benchmark(args.frames_dir, model=args.model, batch_size=args.batch_size)
	print(json.dumps(summary, indent=2))
