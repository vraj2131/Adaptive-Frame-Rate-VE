### Adaptive Frame Rate Engine (AFRE)


AFRE dynamically adapts precision, resolution, and frame interval to maximize throughput while keeping accuracy high for medical imaging workloads (e.g., CT frames). It logs per-frame metrics and provides analytics and plots.

#### Reproducibility
1) Install deps and project:
```bash
pip install -e .
```
2) Prepare a directory with ~100 CT images (PNG/JPG).
3) Run baseline and adaptive:
```bash
python -m adaptive_frame_rate_engine.main --mode baseline --frames-dir /path/to/frames --batch-size 8
python -m adaptive_frame_rate_engine.main --mode adaptive --frames-dir /path/to/frames --batch-size 8
```
4) Or run the benchmark suite:
```bash
python adaptive_frame_rate_engine/test_afre.py --frames-dir /path/to/frames --batch-size 8
```
Artifacts are saved under `./logs/`, `./plots/benchmark_plots.png`, and `./metrics_summary.json`.

#### Sample Results (example)
| Mode | Avg FPS | Avg Lat (ms) | Avg GPU Util (%) | Accuracy Proxy | Throughput Gain | Acc Drop |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 22.4 | 45.5 | 72.1 | 0.84 | - | - |
| AFRE | 38.5 | 31.2 | 61.0 | 0.82 | 1.72× | 2.4% |

#### Notebook
Open `notebooks/performance_analysis.ipynb` to visualize FPS vs resolution, confidence vs precision, and throughput vs GPU load.

#### Tech Stack
- PyTorch, CUDA, OpenCV, psutil, matplotlib, pandas, tqdm

#### Hardware Example
- NVIDIA GTX 1650 Ti

