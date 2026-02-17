# QWOP Model Evaluation Guide

This guide explains how to use the new evaluation system to run your trained QRDQN model multiple times and find the best (fastest) completion times.

## Overview

The evaluation system runs your trained model N times with different seeds, tracks all results, and saves recordings of the top K fastest successful runs. You can then replay these recordings to watch the best performances.

## Quick Start

### 1. Run Evaluation

```bash
python qwop-gym.py evaluate -c config/evaluate_qrdqn_proven.yml
```

This will:
- Run your QRDQN model 500 times with seeds 10000-10499
- Track success rate, times, and distances
- Save recordings of the top 10 fastest runs
- Generate summary statistics (JSON and CSV)

### 2. View Results

After evaluation completes, check the output directory:

```bash
ls -lh data/evaluations/QRDQN-PROVEN-k3jlgned/
```

You'll find:
- `summary.json` - Overall statistics
- `all_runs.csv` - Details of all 500 runs
- `run_1_*.rec` - Recording of the best (fastest) run
- `run_2_*.rec` - Second best run
- ... up to `run_10_*.rec`

### 3. Replay Best Run

```bash
python qwop-gym.py replay -c config/replay_best_run.yml
```

This will replay the best run in the browser so you can watch it.

## Configuration

### Evaluation Config (`config/evaluate_qrdqn_proven.yml`)

Key parameters:
- `n_runs: 500` - Number of evaluation runs (increase for more attempts at finding fast times)
- `keep_top_n: 10` - How many best runs to save recordings for
- `seed_start: 10000` - Starting seed (ensures reproducibility)
- `model_file` - Path to your trained model
- `steps_per_step: 4` - Should match your training configuration

### Replay Config (`config/replay_best_run.yml`)

- `recordings` - Path to recording file(s) to replay
- `fps: 30` - Playback speed (30 = normal QWOP speed)
- `steps_per_step: 4` - Must match evaluation settings

## Output Format

### Summary JSON
```json
{
  "model_file": "data/QRDQN-PROVEN-k3jlgned/model.zip",
  "total_runs": 500,
  "successful_runs": 487,
  "success_rate_percent": 97.4,
  "best_time_seconds": 11.23,
  "average_time_seconds": 12.67,
  "top_n_saved": 10
}
```

### CSV Output
Contains all runs with columns: seed, time, distance, success

## Tips

1. **Finding World Records**: Run more evaluations to increase chances of finding fast times
   ```bash
   # Edit config to increase n_runs to 1000 or more
   n_runs: 1000
   ```

2. **Performance**: Evaluation runs headless (no browser rendering) for speed
   - Expect ~100-200 runs per hour depending on episode length

3. **Reproducibility**: All runs use sequential seeds starting from `seed_start`
   - You can re-run specific seeds by adjusting `seed_start` and `n_runs: 1`

4. **Replay Specific Run**: Edit `recordings` in replay config to point to specific run:
   ```yaml
   recordings:
     - "data/evaluations/QRDQN-PROVEN-k3jlgned/run_1_11.23s_seed10042.rec"
   ```

5. **Multiple Evaluations**: Change `out_dir` to run multiple evaluation batches:
   ```yaml
   out_dir: "data/evaluations/QRDQN-PROVEN-batch2"
   ```

## Example Output

```
======================================================================
Starting evaluation: 500 runs
Model: data/QRDQN-PROVEN-k3jlgned/model.zip
Output directory: data/evaluations/QRDQN-PROVEN-k3jlgned
======================================================================

Run 1/500: time=12.34s, distance=100.0m, success=True [NEW TOP-1]
Run 2/500: time=11.89s, distance=100.0m, success=True [NEW TOP-1]
...
Run 50/500: time=13.01s, distance=100.0m, success=True
  → Progress: Best=11.23s, Avg=12.45s, Success=96.0%
...
Run 500/500: time=12.98s, distance=100.0m, success=True

======================================================================
Saving top 10 recordings...
======================================================================

  [1] run_1_11.23s_seed10042.rec - 11.23s
  [2] run_2_11.45s_seed10156.rec - 11.45s
  ...
  [10] run_10_12.01s_seed10389.rec - 12.01s

======================================================================
EVALUATION SUMMARY
======================================================================
Total runs:        500
Successful:        487 (97.4%)
Failed:            13
Best time:         11.23s (seed=10042)
Average time:      12.67s
Recordings saved:  10
Output directory:  data/evaluations/QRDQN-PROVEN-k3jlgned
======================================================================
```

## Troubleshooting

1. **Model not found**: Verify `model_file` path in config
2. **Environment errors**: Ensure browser/driver paths are correct in `config/env.yml`
3. **Low success rate**: Model may need more training or different hyperparameters

## Files Created

1. `qwop_gym/tools/evaluate.py` - Main evaluation script
2. `config/evaluate_qrdqn_proven.yml` - Evaluation configuration
3. `config/replay_best_run.yml` - Replay configuration for best runs
4. Updated `qwop_gym/tools/main.py` with new "evaluate" action

Enjoy finding those world record times!
