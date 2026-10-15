#!/usr/bin/env python3
"""
Analysis and Visualization Script for Sequential Reaching Experiments

Extracts metrics from TensorBoard logs and generates:
1. Learning curves (success rate vs timesteps)
2. Statistical comparison table
3. Bar charts for final performance
4. Stage progression analysis

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --output-dir results/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not found")
    print("Install with: pip install tensorboard")
    sys.exit(1)


def load_tensorboard_data(log_dir):
    """
    Load data from TensorBoard event files.

    Returns:
        dict: {method_name: {seed: {metric: [(step, value), ...]}}}
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        sys.exit(1)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print("Loading TensorBoard logs...")

    # Map directory names to method names
    method_map = {
        'sequential_sparse': 'Sparse Reward',
        'sequential_shaped': 'Shaped Reward',
        'sequential_curriculum': 'Curriculum Learning',
    }

    for method_dir in log_path.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_map.get(method_dir.name)
        if not method_name:
            continue

        print(f"  Processing: {method_name}")

        # Check for seed subdirectories
        seed_dirs = list(method_dir.glob("seed_*"))
        if not seed_dirs:
            seed_dirs = [method_dir]  # No seed subdirs

        for seed_dir in seed_dirs:
            # Extract seed number
            if seed_dir.name.startswith("seed_"):
                seed = int(seed_dir.name.split("_")[1])
            else:
                seed = 0

            # Find TensorBoard event files
            event_files = list(seed_dir.rglob("events.out.tfevents.*"))

            if not event_files:
                print(f"    Warning: No event files in {seed_dir}")
                continue

            # Load events
            for event_file in event_files:
                try:
                    ea = event_accumulator.EventAccumulator(str(event_file.parent))
                    ea.Reload()

                    # Extract metrics
                    for tag in ea.Tags()['scalars']:
                        if 'success_rate' in tag or 'avg_stages' in tag:
                            for event in ea.Scalars(tag):
                                data[method_name][seed][tag].append(
                                    (event.step, event.value)
                                )
                except Exception as e:
                    print(f"    Warning: Failed to load {event_file}: {e}")
                    continue

            print(f"    Loaded seed {seed}: {len(data[method_name][seed])} metrics")

    return data


def compute_statistics(data):
    """
    Compute mean and std across seeds for each method.

    Returns:
        dict: {method: {metric: {'steps': [...], 'mean': [...], 'std': [...]}}}
    """
    stats = {}

    for method, seeds in data.items():
        stats[method] = {}

        # Get all metrics
        all_metrics = set()
        for seed_data in seeds.values():
            all_metrics.update(seed_data.keys())

        for metric in all_metrics:
            # Collect data from all seeds
            seed_curves = []

            for seed, seed_data in seeds.items():
                if metric in seed_data:
                    seed_curves.append(seed_data[metric])

            if not seed_curves:
                continue

            # Align steps (use union of all steps)
            all_steps = sorted(set(
                step for curve in seed_curves for step, _ in curve
            ))

            # Interpolate to common steps
            aligned_values = []
            for curve in seed_curves:
                steps = [s for s, _ in curve]
                values = [v for _, v in curve]

                # Linear interpolation
                interp_values = np.interp(all_steps, steps, values)
                aligned_values.append(interp_values)

            # Compute statistics
            aligned_values = np.array(aligned_values)
            stats[method][metric] = {
                'steps': all_steps,
                'mean': aligned_values.mean(axis=0),
                'std': aligned_values.std(axis=0),
                'n_seeds': len(seed_curves),
            }

    return stats


def plot_learning_curves(stats, output_dir):
    """Generate learning curve plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot success rate
    plt.figure(figsize=(12, 7))

    colors = {
        'Sparse Reward': '#e74c3c',
        'Shaped Reward': '#3498db',
        'Curriculum Learning': '#2ecc71',
    }

    for method, method_stats in stats.items():
        # Find success rate metric
        success_metric = None
        for metric in method_stats.keys():
            if 'success_rate' in metric.lower():
                success_metric = metric
                break

        if not success_metric:
            continue

        data = method_stats[success_metric]
        steps = np.array(data['steps'])
        mean = data['mean']
        std = data['std']

        color = colors.get(method, '#95a5a6')

        plt.plot(steps, mean, label=method, linewidth=2.5, color=color)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Sequential Reaching: Learning Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

    # Save plot
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def generate_comparison_table(stats, output_dir):
    """Generate comparison table with final performance."""
    output_dir = Path(output_dir)

    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} {'Success Rate':<20} {'Avg Stages':<20} {'N Seeds':<10}")
    print("-" * 80)

    results = {}

    for method, method_stats in stats.items():
        # Get final success rate
        success_rate = None
        avg_stages = None
        n_seeds = 0

        for metric, data in method_stats.items():
            if 'success_rate' in metric.lower() and len(data['mean']) > 0:
                # Take last 10% of training as "final"
                final_idx = int(len(data['mean']) * 0.9)
                success_rate = (data['mean'][final_idx:].mean(),
                               data['std'][final_idx:].mean())
                n_seeds = data['n_seeds']

            if 'avg_stages' in metric.lower() and len(data['mean']) > 0:
                final_idx = int(len(data['mean']) * 0.9)
                avg_stages = (data['mean'][final_idx:].mean(),
                             data['std'][final_idx:].mean())

        results[method] = {
            'success_rate': success_rate,
            'avg_stages': avg_stages,
            'n_seeds': n_seeds,
        }

        # Print row
        sr_str = f"{success_rate[0]:.2%} ± {success_rate[1]:.2%}" if success_rate else "N/A"
        as_str = f"{avg_stages[0]:.2f} ± {avg_stages[1]:.2f}" if avg_stages else "N/A"

        print(f"{method:<25} {sr_str:<20} {as_str:<20} {n_seeds:<10}")

    print("=" * 80)

    # Save to JSON
    output_path = output_dir / 'comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {output_path}")

    return results


def plot_bar_comparison(results, output_dir):
    """Generate bar chart for final performance comparison."""
    output_dir = Path(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = list(results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71'][:len(methods)]

    # Success rate bars
    success_means = [results[m]['success_rate'][0] if results[m]['success_rate'] else 0
                     for m in methods]
    success_stds = [results[m]['success_rate'][1] if results[m]['success_rate'] else 0
                    for m in methods]

    x = np.arange(len(methods))
    ax1.bar(x, success_means, yerr=success_stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Final Success Rate', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    # Average stages bars
    stages_means = [results[m]['avg_stages'][0] if results[m]['avg_stages'] else 0
                    for m in methods]
    stages_stds = [results[m]['avg_stages'][1] if results[m]['avg_stages'] else 0
                   for m in methods]

    ax2.bar(x, stages_means, yerr=stages_stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_ylabel('Average Stages Completed', fontsize=12)
    ax2.set_title('Final Stages Completed', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylim(0, 3.5)
    ax2.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='All stages')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'final_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Sequential Reaching experiments")
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Directory for output plots and tables')
    args = parser.parse_args()

    print("=" * 80)
    print("Sequential Reaching: Experiment Analysis")
    print("=" * 80)

    # Load data
    data = load_tensorboard_data(args.log_dir)

    if not data:
        print("\nERROR: No data found in logs!")
        print("Make sure experiments have been run and logs are in:", args.log_dir)
        sys.exit(1)

    print(f"\nLoaded {len(data)} methods:")
    for method, seeds in data.items():
        print(f"  • {method}: {len(seeds)} seeds")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(data)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_learning_curves(stats, output_dir)

    # Generate comparison table
    results = generate_comparison_table(stats, output_dir)

    # Generate bar charts
    plot_bar_comparison(results, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nResults saved in: {output_dir}/")
    print(f"  • learning_curves.png")
    print(f"  • final_comparison_bars.png")
    print(f"  • comparison_results.json")
    print()


if __name__ == "__main__":
    main()
