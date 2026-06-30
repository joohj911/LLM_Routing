# lm_routing

A framework for training and evaluating LLM routers that intelligently route queries between a strong model and a weak model based on query difficulty.

This fork focuses on **agent/tool-calling routing** using [Qwen3.5](https://huggingface.co/Qwen) as the model pair and [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) as the evaluation benchmark, with local embeddings via `intfloat/multilingual-e5-small`.

## Overview

Two routing methods are compared across two model pairs:

| Method | Description |
|--------|-------------|
| `random` | Random baseline — uniform routing at each threshold |
| `mf` | Matrix Factorization router trained on pairwise pass/fail labels |
| `uniroute` | UniRoute K-Means cluster-based router ([arXiv:2502.08773](https://arxiv.org/abs/2502.08773)) |
| `permodel` | Per-model regression router — independent P(pass) regressors per model; routes by predicted gain P(strong)−P(weak) (R2-Router skeleton, budget-free) |

**Model pairs evaluated:**
- Pair A: `Qwen/Qwen3.5-0.8B` (weak) vs `Qwen/Qwen3.5-9B` (strong)
- Pair B: `Qwen/Qwen3.5-2B` (weak) vs `Qwen/Qwen3.5-9B` (strong)

**Embeddings:** `intfloat/multilingual-e5-small` (384-dim, fully local, no API key)  
**Benchmark:** BFCL v4 (single-turn: non-live + live categories)

Each router produces a deferral curve: x = strong model call %, y = pass rate.

## Installation

```bash
git clone https://github.com/joohj911/LLM_Routing.git
cd LLM_Routing

# 1. Install the torch build that matches your NVIDIA driver FIRST.
#    Check your driver's max CUDA version with `nvidia-smi` (top-right).
#    Example for CUDA 12.1 drivers:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 2. Install the package (torch is pinned to 2.5.x so this won't upgrade it).
pip install -e ".[eval]"
# Optional: 4-bit quantization for low-VRAM inference
pip install -e ".[eval,quant]"
```

> **Why install torch first?** Newer torch wheels (2.6+) target newer CUDA
> runtimes that older drivers don't support. The package pins
> `torch>=2.5.1,<2.6`, but the *CUDA build* (cu121 vs cu124) is selected by the
> `--index-url` above — so install torch explicitly before `pip install -e .`.

## Quick Start: Full Pipeline

Run the entire experiment pipeline with one command:

```bash
bash run_experiments.sh
```

Options:
```
--bfcl-dir DIR           Base directory for BFCL data (default: ./bfcl_data)
--results-dir DIR         Output directory for results  (default: ./results)
--output-excel FILE       Output Excel file             (default: routing_results.xlsx)
--load-in-4bit            Use 4-bit quantization for model evaluation
--skip-embed              Skip embedding generation (reuse existing bfcl_data/)
--skip-eval-models        Skip model evaluation (reuse existing eval_results.json)
--num-results N           Threshold points per router   (default: 10)
--random-iters N          Random router averaging iters (default: 10)
```

Output:
- `routing_results.xlsx` — Excel workbook with deferral curves and summary
- `routing_curves.png` — side-by-side line graphs for both model pairs

## Step-by-Step Pipeline

### Step 1: Generate BFCL Embeddings

Download BFCL v4 from GitHub and generate multilingual-e5-small embeddings:

```bash
python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py embed \
  --output-dir ./bfcl_data
```

Output:
- `bfcl_data/embeddings.npy` — prompt embeddings (shape: N × 384)
- `bfcl_data/prompts.json` — prompt metadata with BFCL IDs and split names

### Step 2: Evaluate Models on BFCL

Evaluate all three models. Each uses all available GPUs via `device_map="auto"` with auto-detected batch size:

```bash
python lm_routing/evals/eval_bfcl_models.py \
  --prompts-path ./bfcl_data/prompts.json \
  --output-path ./eval_results.json \
  --models Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B
```

Options:
- `--models` — one or more HuggingFace model IDs
- `--batch-size` — batch size for `model.generate()` (default: 0 = auto-detect from GPU memory at 80% utilization)
- `--load-in-4bit` — 4-bit quantization for low-VRAM setups (requires `bitsandbytes`)
- `--max-new-tokens` — max generation length (default: 512)

Output: `eval_results.json` — per-sample pass/fail for each model.

The script prints a summary at the end:

```
============================================================
BFCL Evaluation Summary
============================================================
  Total samples :  1234
      Qwen/Qwen3.5-0.8B :  612/1234  (49.6%)
        Qwen/Qwen3.5-2B :  768/1234  (62.2%)
        Qwen/Qwen3.5-9B : 1003/1234  (81.3%)
============================================================
```

> **Note:** The script prints the full HuggingFace model IDs used in the next step, e.g. `--weak-model Qwen/Qwen3.5-2B --strong-model Qwen/Qwen3.5-9B`.

### Step 3: Convert to Train/Test Splits

Run once per model pair, writing to separate directories:

```bash
# Pair A: 0.8B vs 9B
python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path ./eval_results.json \
  --prompts-path ./bfcl_data/prompts.json \
  --output-dir   ./bfcl_data_0.8B \
  --weak-model   Qwen/Qwen3.5-0.8B \
  --strong-model Qwen/Qwen3.5-9B

# Pair B: 2B vs 9B
python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path ./eval_results.json \
  --prompts-path ./bfcl_data/prompts.json \
  --output-dir   ./bfcl_data_2B \
  --weak-model   Qwen/Qwen3.5-2B \
  --strong-model Qwen/Qwen3.5-9B
```

Output per pair:
- `train_data.json` — training labels for MF and UniRoute routers
- `test_data.json` — held-out test set for evaluation

**Labeling rule:**
| Weak passes | Strong passes | Label | Reason |
|---|---|---|---|
| ✓ | ✓ | Route to weak | Weak is sufficient |
| ✗ | ✓ | Route to strong | Strong is needed |
| ✓ | ✗ | Route to weak | Weak succeeded; strong failed |
| ✗ | ✗ | Route to strong | Neither succeeded → send to frontier |

### Step 4: Train the MF Router

```bash
python lm_routing/routers/matrix_factorization/train_matrix_factorization.py \
  --train-data  ./bfcl_data_2B/train_data.json \
  --npy-path    ./bfcl_data/embeddings.npy \
  --output-path ./bfcl_data_2B/mf_model.pt \
  --num-epochs 100 \
  --dim 128 \
  --text-dim 384 \
  --batch-size 64
```

The checkpoint is saved without the prompt embedding matrix (not needed at inference time).

### Step 5: Train the UniRoute Router

UniRoute (§5.1, [arXiv:2502.08773](https://arxiv.org/abs/2502.08773)) represents each LLM as a per-cluster error rate vector and routes by comparing weak vs. strong error rates in the nearest K-Means cluster.

```bash
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   ./bfcl_data_2B/train_data.json \
  --npy-path     ./bfcl_data/embeddings.npy \
  --output-path  ./bfcl_data_2B/uniroute_model.pt \
  --weak-model   Qwen/Qwen3.5-2B \
  --strong-model Qwen/Qwen3.5-9B
```

The script automatically selects the best K (number of clusters) via an internal validation AUC sweep over candidates `{5, 10, 13, 20, 30, max(5, N_val//50)}`.

### Step 6: Evaluate All Routers

Evaluate random baseline, MF, and UniRoute together:

```bash
python -m lm_routing.evals.evaluate \
  --routers random mf uniroute \
  --test-data             ./bfcl_data_2B/test_data.json \
  --mf-checkpoint         ./bfcl_data_2B/mf_model.pt \
  --uniroute-checkpoint   ./bfcl_data_2B/uniroute_model.pt \
  --strong-model          Qwen/Qwen3.5-9B \
  --weak-model            Qwen/Qwen3.5-2B \
  --output                ./results/pair_2B \
  --output-json           ./results/pair_2B/eval_results.json
```

Example output:

```
=================================================================
BFCL Routing Summary
=================================================================
  Weak model   (      Qwen/Qwen3.5-2B):   62.4%
  Strong model (      Qwen/Qwen3.5-9B):   81.3%

  Router             Threshold  Pass Rate    Weak%   Strong%
  ---------------------------------------------------------
  random              0.1000     64.1%    90.0%    10.0%
  mf                  0.3000     79.8%    72.1%    27.9%
  uniroute            0.5000     76.3%    68.4%    31.6%
  ...
=================================================================
```

### Step 7: Collect Results into Excel

```bash
python collect_results.py \
  --results-jsons \
    ./results/pair_0.8B/eval_results.json \
    ./results/pair_2B/eval_results.json \
  --output routing_results.xlsx
```

Output:
- **Sheet "Deferral Curves"** — raw data: pair, method, threshold, pass rate, weak%, strong%
- **Sheet "Summary"** — weak-only and strong-only accuracy per model pair
- **Sheet "Graphs"** — embedded PNG with side-by-side deferral curve plots

## Local Inference (Two-GPU Setup)

After training, load both models locally with `LocalController`. Each model is pinned to its own GPU — no per-request model loading:

```python
from lm_routing.local_pipeline import LocalController

controller = LocalController(
    routers=["mf"],
    strong_model="Qwen/Qwen3.5-9B",
    weak_model="Qwen/Qwen3.5-2B",
    strong_device="cuda:1",
    weak_device="cuda:0",
    config={"mf": {"checkpoint_path": "./bfcl_data_2B/mf_model.pt", "text_dim": 384}},
)

response = controller.completion(
    router="mf",
    threshold=0.3,   # tune based on desired strong model call %
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
    tools=[...],     # OpenAI tool format
)
print(response["choices"][0]["message"])
```

## Configuration

Router configuration is passed via `--config` YAML or via `--mf-checkpoint` / `--uniroute-checkpoint` shortcuts:

```yaml
# config.yaml
mf:
  checkpoint_path: ./bfcl_data_2B/mf_model.pt
uniroute:
  checkpoint_path: ./bfcl_data_2B/uniroute_model.pt
```

## Extending

Implement the abstract `Router` class in `lm_routing/routers/routers.py` and add it to `ROUTER_CLS`. The only required method is `calculate_strong_win_rate(prompt) -> float`. If the returned value exceeds the user-specified threshold, the request goes to the strong model.

## Acknowledgements

This project is a modified derivative of [RouteLLM](https://github.com/lm-sys/RouteLLM)
(Apache-2.0). The matrix-factorization router builds on RouteLLM; substantial
changes were made for tool-calling routing (BFCL evaluation, Qwen3.5 model pairs,
local multilingual-e5 embeddings, UniRoute and per-model regression routers). See
[`NOTICE`](./NOTICE) for attribution and [`LICENSE`](./LICENSE) for terms.

## Citation

```bibtex
@misc{ong2024routellmlearningroutellms,
      title={RouteLLM: Learning to Route LLMs with Preference Data},
      author={Isaac Ong and Amjad Almahairi and Vincent Wu and Wei-Lin Chiang and Tianhao Wu and Joseph E. Gonzalez and M Waleed Kadous and Ion Stoica},
      year={2024},
      eprint={2406.18665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.18665},
}

@misc{chen2025uniroutescalablellmrouting,
      title={UniRoute: Scalable LLM Routing via Unified Representation Learning},
      year={2025},
      eprint={2502.08773},
      archivePrefix={arXiv},
}
```
