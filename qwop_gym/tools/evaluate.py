# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import gymnasium as gym
import importlib
import os
import json
import csv
import heapq
import glob
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class RunResult:
    """Stores results from a single evaluation run"""
    seed: int
    time: float
    distance: float
    success: bool
    actions: List[int]
    
    def __lt__(self, other):
        # For heapq - lower time is better, but heapq is min-heap
        # We want to keep the slowest of the top-N, so we reverse comparison
        return self.time > other.time


def load_model(mod_name, cls_name, file):
    print("Loading %s model from %s" % (cls_name, file))
    mod = importlib.import_module(mod_name)
    
    if cls_name == "BC":
        return mod.reconstruct_policy(file)
    
    return getattr(mod, cls_name).load(file)


def run_episode(env, model, steps_per_step, seed=None, verbose=False):
    """Run a single episode and return results"""
    obs, info = env.reset(seed=seed)
    
    # Log seed and initial observation checksum to verify variation
    if verbose:
        obs_checksum = hash(obs.tobytes())
        print(f"  [DEBUG] Seed={seed}, obs_checksum={obs_checksum}")
    
    actions = []
    terminated = False
    model_calls = 0
    env_steps = 0
    
    while not terminated:
        action, _states = model.predict(obs)
        actions.append(int(action))
        model_calls += 1
        
        for _ in range(steps_per_step):
            obs, reward, terminated, truncated, info = env.step(action)
            env_steps += 1
            if terminated:
                break
    
    if verbose:
        print(f"  [DEBUG] Model calls: {model_calls}, Env steps: {env_steps}, Actions recorded: {len(actions)}")
        print(f"  [DEBUG] Final time: {info['time']:.2f}s, distance: {info['distance']:.1f}m")
        expected_actions = (info['time'] * 30) / env.unwrapped.frames_per_step / steps_per_step
        print(f"  [DEBUG] Expected actions for {info['time']:.2f}s: ~{expected_actions:.0f}")
    
    return RunResult(
        seed=env.unwrapped.seedval,
        time=info["time"],
        distance=info["distance"],
        success=info["is_success"],
        actions=actions
    )


def save_recording(result: RunResult, filepath: str):
    """Save a run as a .rec file compatible with replay tool"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        # Write header with seed
        f.write(f"seed={result.seed}\n")
        
        # Write actions
        for action in result.actions:
            f.write(f"{action}\n")
        
        # Write episode marker (successful episode)
        f.write("*\n")


def save_recording_incrementally(result: RunResult, top_n_heap, keep_top_n, out_dir):
    """
    Save recording immediately if it's in top N.
    Returns True if saved, False otherwise.
    """
    if len(top_n_heap) <= keep_top_n:
        # It's in top N - save it with temp filename
        filename = f"run_temp_{result.time:.2f}s_seed{result.seed}.rec"
        filepath = os.path.join(out_dir, filename)
        save_recording(result, filepath)
        return True
    elif result.time < top_n_heap[0].time:
        # Better than worst in top N - save it, delete old worst
        old_worst = top_n_heap[0]
        
        # Delete old worst file(s)
        old_pattern = f"run_temp_{old_worst.time:.2f}s_seed{old_worst.seed}.rec"
        old_path = os.path.join(out_dir, old_pattern)
        if os.path.exists(old_path):
            os.remove(old_path)
        
        # Save new file
        filename = f"run_temp_{result.time:.2f}s_seed{result.seed}.rec"
        filepath = os.path.join(out_dir, filename)
        save_recording(result, filepath)
        return True
    
    return False


def evaluate(
    n_runs,
    model_file,
    model_mod,
    model_cls,
    steps_per_step,
    keep_top_n,
    out_dir,
    seed_start,
):
    """
    Evaluate a trained model over multiple runs and save the best recordings.
    
    Args:
        n_runs: Number of evaluation runs to perform
        model_file: Path to the trained model file
        model_mod: Module name (e.g., "sb3_contrib")
        model_cls: Algorithm class name (e.g., "QRDQN")
        steps_per_step: Number of env.step() calls per action
        keep_top_n: Number of best runs to save recordings for
        out_dir: Output directory for results and recordings
        seed_start: Starting seed value for runs
    """
    # Load the trained model
    model = load_model(model_mod, model_cls, model_file)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize tracking variables
    all_results = []
    top_n_heap = []  # Min-heap to track top N runs
    successful_times = []
    failed_distances = []
    
    print(f"\n{'='*70}")
    print(f"Starting evaluation: {n_runs} runs")
    print(f"Model: {model_file}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*70}\n")
    
    # Create environment once and reuse it for all runs
    env = gym.make("local/QWOP-v1", seed=seed_start)
    
    try:
        # Initial resets (gymnasium requirement)
        env.reset()
        env.reset()
        
        # Run evaluations
        for i in range(n_runs):
            seed = seed_start + i
            
            # Run episode with new seed (reloads page in same browser)
            # Enable verbose logging for first 3 runs to show seed/obs variation
            verbose = (i < 3)
            result = run_episode(env, model, steps_per_step, seed=seed, verbose=verbose)
            all_results.append(result)
            
            # Track statistics
            if result.success:
                successful_times.append(result.time)
                
                # Update top N heap and save incrementally
                if len(top_n_heap) < keep_top_n:
                    heapq.heappush(top_n_heap, result)
                    status = "[NEW TOP-%d]" % len(top_n_heap)
                    save_recording_incrementally(result, top_n_heap, keep_top_n, out_dir)
                elif result.time < top_n_heap[0].time:
                    # Replace the slowest in top N
                    heapq.heapreplace(top_n_heap, result)
                    status = "[NEW TOP-%d]" % keep_top_n
                    save_recording_incrementally(result, top_n_heap, keep_top_n, out_dir)
                else:
                    status = ""
                
                print(f"Run {i+1}/{n_runs}: time={result.time:.2f}s, distance={result.distance:.1f}m, success=True {status}")
            else:
                failed_distances.append(result.distance)
                print(f"Run {i+1}/{n_runs}: time={result.time:.2f}s, distance={result.distance:.1f}m, success=False")
            
            # Check for identical results after 50 runs
            if (i + 1) == 50 and len(all_results) >= 50:
                unique_times = set(r.time for r in all_results)
                if len(unique_times) == 1:
                    print(f"\n{'!'*70}")
                    print("WARNING: All 50 runs produced identical results!")
                    print(f"Time: {all_results[0].time:.2f}s (every single run)")
                    print("This suggests the model is fully deterministic.")
                    print("Different seeds are not producing variation in outcomes.")
                    print(f"{'!'*70}\n")
            
            # Print progress summary every 50 runs
            if (i + 1) % 50 == 0 and successful_times:
                best_time = min(successful_times)
                avg_time = sum(successful_times) / len(successful_times)
                success_rate = len(successful_times) / (i + 1) * 100
                print(f"  → Progress: Best={best_time:.2f}s, Avg={avg_time:.2f}s, Success={success_rate:.1f}%")
    
    finally:
        env.close()
    
    # Sort top N by time (ascending)
    top_n_results = sorted(top_n_heap, key=lambda x: x.time)
    
    # Rename temp recordings to final rankings
    print(f"\n{'='*70}")
    print(f"Finalizing top {len(top_n_results)} recordings...")
    print(f"{'='*70}\n")
    
    for rank, result in enumerate(top_n_results, 1):
        # Find the temp file for this result
        temp_filename = f"run_temp_{result.time:.2f}s_seed{result.seed}.rec"
        temp_filepath = os.path.join(out_dir, temp_filename)
        
        # Create final filename with ranking
        final_filename = f"run_{rank}_{result.time:.2f}s_seed{result.seed}.rec"
        final_filepath = os.path.join(out_dir, final_filename)
        
        # Rename if temp file exists
        if os.path.exists(temp_filepath):
            os.rename(temp_filepath, final_filepath)
            print(f"  [{rank}] {final_filename} - {result.time:.2f}s")
        else:
            # If temp file doesn't exist (shouldn't happen), create it
            save_recording(result, final_filepath)
            print(f"  [{rank}] {final_filename} - {result.time:.2f}s (created)")
    
    # Calculate statistics
    total_success = len(successful_times)
    success_rate = (total_success / n_runs * 100) if n_runs > 0 else 0
    best_time = float(min(successful_times)) if successful_times else None
    avg_time = float(sum(successful_times) / len(successful_times)) if successful_times else None
    
    # Save summary statistics
    summary = {
        "model_file": model_file,
        "total_runs": n_runs,
        "successful_runs": total_success,
        "failed_runs": n_runs - total_success,
        "success_rate_percent": round(float(success_rate), 2),
        "best_time_seconds": round(best_time, 2) if best_time else None,
        "average_time_seconds": round(avg_time, 2) if avg_time else None,
        "top_n_saved": len(top_n_results),
        "seed_range": {"start": seed_start, "end": seed_start + n_runs - 1}
    }
    
    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results to CSV
    csv_file = os.path.join(out_dir, "all_runs.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "time", "distance", "success"])
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                "seed": result.seed,
                "time": round(float(result.time), 2),
                "distance": round(float(result.distance), 2),
                "success": result.success
            })
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs:        {n_runs}")
    print(f"Successful:        {total_success} ({success_rate:.1f}%)")
    print(f"Failed:            {n_runs - total_success}")
    if best_time:
        print(f"Best time:         {best_time:.2f}s (seed={top_n_results[0].seed})")
    if avg_time:
        print(f"Average time:      {avg_time:.2f}s")
    print(f"Recordings saved:  {len(top_n_results)}")
    print(f"Output directory:  {out_dir}")
    print(f"{'='*70}\n")
    
    if top_n_results:
        print("Top runs:")
        for rank, result in enumerate(top_n_results, 1):
            print(f"  [{rank}] {result.time:.2f}s (seed={result.seed}, distance={result.distance:.1f}m)")
        print()
