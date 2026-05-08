#!/usr/bin/env python3
"""
Analyze experiment statistics:
- Tool usage by setup and conversation
- Budget usage
- Wait and observe steps before final assessment
- Detection metrics (precision, recall, F1, full detection rate, false positives)
- Saves results to JSON and a Markdown table
"""
import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

# Import shared constants/functions from the runner
from run_experiment import CELLS, parse_findings, score_run, AGENTS, _normalise_targets


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0


def _cell_id(experiment: str, tool_setup: str, budget: int) -> str:
    return f"{experiment}_{tool_setup}_b{budget}"


def _build_cell_target_map() -> dict[str, list[str]]:
    """Map cell directory names -> ground-truth targets using CELLS."""
    mapping: dict[str, list[str]] = {}
    for experiment, targets, tool_setup, budget in CELLS:
        cell = _cell_id(experiment, tool_setup, budget)
        mapping[cell] = _normalise_targets(targets)
    return mapping


def analyze_experiments(input_dir: Path, output_file: Path | None = None):
    """Extract and analyze experiment statistics."""
    experiments_dir = input_dir
    cell_targets = _build_cell_target_map()

    # Group by cell
    cells = defaultdict(lambda: {
        "experiment": "",
        "tool_setup": "",
        "budget": 0,
        "targets": [],
        "runs": [],
    })

    # Process each cell directory
    for cell_dir in sorted(experiments_dir.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name == "exp_summary.json":
            continue

        cell_name = cell_dir.name
        targets = cell_targets.get(cell_name, [])

        # Find all result files for this cell
        result_files = sorted(cell_dir.glob("r*.json"))

        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Extract metadata
            experiment = data.get("input_file", "")
            budget = data.get("budget", 0)
            budget_used = data.get("budget_used", 0)
            interactions = data.get("interactions", [])
            findings = data.get("findings", "")

            # Score detection
            detected = parse_findings(findings)
            is_hit, detected_targets, missed_targets, false_positives, precision, recall, f1 = score_run(detected, targets)

            # Count tool usage
            tool_counts = defaultdict(int)
            wait_and_observe_count = 0
            for interaction in interactions:
                tool = interaction.get("tool", "unknown")
                tool_counts[tool] += 1
                if tool == "wait_and_observe":
                    wait_and_observe_count += 1

            # Store run data
            cells[cell_name]["experiment"] = Path(experiment).parent.parent.name if experiment else cell_name.split("_setup")[0]
            cells[cell_name]["tool_setup"] = cell_name.split("_")[-2] if "_setup" in cell_name else ""
            cells[cell_name]["budget"] = budget
            cells[cell_name]["targets"] = targets
            cells[cell_name]["runs"].append({
                "rep": result_file.stem,
                "budget_used": budget_used,
                "total_tool_calls": len(interactions),
                "tool_counts": dict(tool_counts),
                "wait_and_observe_count": wait_and_observe_count,
                "findings": findings,
                "detected_agents": detected,
                "detected_targets": detected_targets,
                "missed_targets": missed_targets,
                "false_positives": false_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "is_hit": is_hit,
            })

    # Aggregate per cell
    output: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []

    for cell_name in sorted(cells.keys()):
        cell_data = cells[cell_name]
        runs = cell_data["runs"]

        if not runs:
            continue

        n = len(runs)

        # Budget stats
        budget_used_list = [r["budget_used"] for r in runs]
        budget_stats = {
            "total_used": sum(budget_used_list),
            "average": round(sum(budget_used_list) / n, 2),
            "sd": round(_safe_stdev(budget_used_list), 2),
            "max": max(budget_used_list),
            "min": min(budget_used_list),
        }

        # Total tool calls stats
        total_tools_list = [r["total_tool_calls"] for r in runs]
        total_tools_stats = {
            "mean": round(sum(total_tools_list) / n, 2),
            "sd": round(_safe_stdev(total_tools_list), 2),
            "max": max(total_tools_list),
            "min": min(total_tools_list),
            "per_run": total_tools_list,
        }

        # Per-tool stats
        tool_usage_per_run: dict[str, list[int]] = defaultdict(list)
        all_tool_counts = defaultdict(int)
        for run in runs:
            for tool, count in run["tool_counts"].items():
                all_tool_counts[tool] += count
                tool_usage_per_run[tool].append(count)
            # Ensure every tool appears for every run (fill 0 for missing)
        unique_tools = sorted(all_tool_counts.keys())
        for tool in unique_tools:
            for run in runs:
                if tool not in run["tool_counts"]:
                    tool_usage_per_run[tool].append(0)

        tool_stats = {}
        for tool, usage_list in tool_usage_per_run.items():
            mean = sum(usage_list) / len(usage_list) if usage_list else 0
            sd = _safe_stdev(usage_list)
            total = all_tool_counts[tool]
            tool_stats[tool] = {
                "total": total,
                "mean": round(mean, 2),
                "sd": round(sd, 2),
            }

        # Wait and observe stats
        wait_counts = [r["wait_and_observe_count"] for r in runs]
        wait_stats = {
            "mean": round(sum(wait_counts) / n, 2),
            "sd": round(_safe_stdev(wait_counts), 2),
            "max": max(wait_counts),
            "min": min(wait_counts),
            "per_run": wait_counts,
        }

        # Detection stats
        hit_count = sum(1 for r in runs if r["is_hit"])
        precisions = [r["precision"] for r in runs]
        recalls = [r["recall"] for r in runs]
        f1s = [r["f1"] for r in runs]
        fp_counts = [len(r["false_positives"]) for r in runs]

        detection_stats = {
            "full_detection_rate": round(hit_count / n, 4),
            "avg_precision": round(sum(precisions) / n, 4),
            "avg_recall": round(sum(recalls) / n, 4),
            "avg_f1": round(sum(f1s) / n, 4),
            "avg_false_positives_per_run": round(sum(fp_counts) / n, 4),
            "runs_with_false_positives": sum(1 for c in fp_counts if c > 0),
        }

        output[cell_name] = {
            "experiment": cell_data["experiment"],
            "tool_setup": cell_data["tool_setup"],
            "budget": cell_data["budget"],
            "targets": cell_data["targets"],
            "total_runs": n,
            "budget_stats": budget_stats,
            "total_tool_calls_stats": total_tools_stats,
            "tool_usage": {
                "per_tool_stats": tool_stats,
                "total_calls": dict(all_tool_counts),
                "percentages": {
                    tool: round(count / sum(all_tool_counts.values()) * 100, 1)
                    if sum(all_tool_counts.values()) > 0 else 0
                    for tool, count in all_tool_counts.items()
                },
            },
            "wait_and_observe": wait_stats,
            "detection": detection_stats,
            "runs": [
                {
                    "rep": r["rep"],
                    "budget_used": r["budget_used"],
                    "total_tool_calls": r["total_tool_calls"],
                    "tool_counts": r["tool_counts"],
                    "wait_and_observe_count": r["wait_and_observe_count"],
                    "detected_agents": r["detected_agents"],
                    "detected_targets": r["detected_targets"],
                    "missed_targets": r["missed_targets"],
                    "false_positives": r["false_positives"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "is_hit": r["is_hit"],
                }
                for r in runs
            ],
        }

        # Build table row
        table_rows.append({
            "cell": cell_name,
            "experiment": cell_data["experiment"],
            "targets": ", ".join(cell_data["targets"]) if cell_data["targets"] else "—",
            "setup": cell_data["tool_setup"],
            "budget": cell_data["budget"],
            "runs": n,
            "full_det": detection_stats["full_detection_rate"],
            "precision": detection_stats["avg_precision"],
            "recall": detection_stats["avg_recall"],
            "f1": detection_stats["avg_f1"],
            "avg_fp": detection_stats["avg_false_positives_per_run"],
            "avg_budget": budget_stats["average"],
            "total_tools_mean": total_tools_stats["mean"],
            "total_tools_sd": total_tools_stats["sd"],
        })

    # Save JSON
    if output_file is None:
        output_file = experiments_dir.parent / "analysis_stats.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Statistics saved to {output_file}")

    # Build and save Markdown table
    md_file = output_file.with_suffix(".md")
    md_lines = [
        "# Experiment Analysis",
        "",
        "| Cell | Experiment | Targets | Setup | Budget | Runs | Full Det. | Precision | Recall | F1 | Avg FP | Avg Budget | Total Tools (mean ± SD) |",
        "|------|------------|---------|-------|--------|------|-----------|-----------|--------|-----|--------|-----------|------------------------|",
    ]

    for row in table_rows:
        md_lines.append(
            f"| {row['cell']} | {row['experiment']} | {row['targets']} | {row['setup']} | {row['budget']} | {row['runs']} | "
            f"{row['full_det']:.1%} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
            f"{row['avg_fp']:.2f} | {row['avg_budget']:.2f} | {row['total_tools_mean']} ± {row['total_tools_sd']} |"
        )

    md_lines.extend(["", "**Legend:**"])
    md_lines.append("- **Full Det.** = Full detection rate (proportion of runs where all ground-truth targets were flagged)")
    md_lines.append("- **Precision** = Average precision across runs")
    md_lines.append("- **Recall** = Average recall across runs")
    md_lines.append("- **F1** = Average F1 score across runs")
    md_lines.append("- **Avg FP** = Average false positives per run")
    md_lines.append("- **Avg Budget** = Average budget used per run")
    md_lines.append("- **Total Tools** = Total tool calls per run (mean ± standard deviation)")

    md_text = "\n".join(md_lines) + "\n"
    md_file.write_text(md_text)
    print(f"Markdown table saved to {md_file}")

    # Print summary
    print("\n" + "="*120)
    print("EXPERIMENT STATISTICS SUMMARY")
    print("="*120)

    for row in table_rows:
        print(f"\n{row['cell']}:")
        print(f"  Experiment: {row['experiment']}")
        print(f"  Targets: {row['targets']}")
        print(f"  Total Runs: {row['runs']}")
        print(f"  Budget (per run): {row['budget']}")
        print(f"  Full Detection Rate: {row['full_det']:.1%}")
        print(f"  Precision: {row['precision']:.3f}")
        print(f"  Recall: {row['recall']:.3f}")
        print(f"  F1: {row['f1']:.3f}")
        print(f"  Avg False Positives / Run: {row['avg_fp']:.2f}")
        print(f"  Budget Stats:")
        print(f"    - Average used: {row['avg_budget']}")
        print(f"    - Range: {output[row['cell']]['budget_stats']['min']} - {output[row['cell']]['budget_stats']['max']}")
        print(f"  Total Tool Calls (mean ± SD): {row['total_tools_mean']} ± {row['total_tools_sd']}")
        print(f"  Tool Usage (per-run mean ± SD):")
        for tool, tool_stat in sorted(output[row['cell']]['tool_usage']['per_tool_stats'].items(), key=lambda x: -x[1]['total']):
            total = tool_stat['total']
            mean = tool_stat['mean']
            sd = tool_stat['sd']
            pct = output[row['cell']]['tool_usage']['percentages'].get(tool, 0)
            print(f"    - {tool}: {total} total ({mean} ± {sd} per run, {pct}%)")
        print(f"  Wait and Observe Steps:")
        wait = output[row['cell']]['wait_and_observe']
        print(f"    - Mean ± SD: {wait['mean']} ± {wait['sd']}")
        print(f"    - Range: {wait['min']} - {wait['max']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment statistics")
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("results/v0.6"),
                        help="Directory containing experiment results (default: results/v0.6)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output file path for JSON (default: <input_dir>/../analysis_stats.json)")
    args = parser.parse_args()

    analyze_experiments(args.input_dir, args.output)
