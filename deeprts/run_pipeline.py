#!/usr/bin/env python3
"""
Master Pipeline: DeepRTS GameNGen Experiment

This script runs the complete pipeline:
1. Collect frame-action dataset from DeepRTS
2. Train diffusion model
3. Generate sample rollouts
4. Analyze model for strategic understanding
5. Produce final report

Usage:
    python run_pipeline.py --output ./experiment_results
    
For quick test run:
    python run_pipeline.py --quick-test
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline"""
    # Output
    output_dir: str = "./experiment_results"
    
    # Dataset collection
    num_episodes: int = 100
    map_name: str = "15x15-2-FFA"
    frame_size: int = 64
    frame_skip: int = 4
    agent_type: str = "heuristic"
    
    # Training
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 5e-5
    buffer_size: int = 8
    
    # Analysis
    num_analysis_samples: int = 50
    num_rollout_steps: int = 30
    
    # Inference
    num_generated_frames: int = 50


def run_command(cmd: list, description: str, cwd: str = None) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'numpy', 'PIL', 'tqdm']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'PIL':
                __import__('PIL')
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def run_pipeline(config: PipelineConfig):
    """Run the complete pipeline"""
    
    start_time = datetime.now()
    results = {
        "config": asdict(config),
        "start_time": start_time.isoformat(),
        "steps": {},
        "success": False
    }
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_path = output_path / "dataset"
    model_path = output_path / "model"
    analysis_path = output_path / "analysis"
    generated_path = output_path / "generated"
    
    script_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("DEEPRTS GAMENGEN PIPELINE")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Map: {config.map_name}")
    
    # Step 1: Collect Dataset
    print("\n\n" + "#"*60)
    print("# PHASE 1: DATA COLLECTION")
    print("#"*60)
    
    success = run_command([
        sys.executable, str(script_dir / "collect_dataset.py"),
        "--output", str(dataset_path),
        "--episodes", str(config.num_episodes),
        "--map", config.map_name,
        "--frame-size", str(config.frame_size),
        "--frame-skip", str(config.frame_skip),
        "--agent", config.agent_type,
        "--max-steps", "3000"
    ], "Collecting frame-action dataset")
    
    results["steps"]["data_collection"] = {
        "success": success,
        "output_path": str(dataset_path)
    }
    
    if not success:
        print("❌ Data collection failed!")
        # Continue anyway with mock data for testing
        print("Continuing with available data...")
    
    # Step 2: Train Model
    print("\n\n" + "#"*60)
    print("# PHASE 2: MODEL TRAINING")
    print("#"*60)
    
    success = run_command([
        sys.executable, str(script_dir / "train_diffusion.py"),
        "--dataset", str(dataset_path),
        "--output", str(model_path),
        "--epochs", str(config.num_epochs),
        "--batch-size", str(config.batch_size),
        "--lr", str(config.learning_rate),
        "--buffer-size", str(config.buffer_size),
        "--image-size", str(config.frame_size)
    ], "Training diffusion model")
    
    results["steps"]["training"] = {
        "success": success,
        "output_path": str(model_path)
    }
    
    if not success:
        print("❌ Training failed!")
        save_results(results, output_path)
        return results
    
    # Find checkpoint
    checkpoint_path = model_path / "checkpoint_final.pt"
    if not checkpoint_path.exists():
        checkpoints = list(model_path.glob("checkpoint_*.pt"))
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]
        else:
            print("❌ No checkpoint found!")
            save_results(results, output_path)
            return results
    
    # Step 3: Generate Samples
    print("\n\n" + "#"*60)
    print("# PHASE 3: SAMPLE GENERATION")
    print("#"*60)
    
    success = run_command([
        sys.executable, str(script_dir / "inference.py"),
        "--model", str(checkpoint_path),
        "--dataset", str(dataset_path),
        "--output", str(generated_path),
        "--num-frames", str(config.num_generated_frames),
        "--steps", "20"
    ], "Generating sample rollouts")
    
    results["steps"]["generation"] = {
        "success": success,
        "output_path": str(generated_path)
    }
    
    # Step 4: Analyze Model
    print("\n\n" + "#"*60)
    print("# PHASE 4: STRATEGIC ANALYSIS")
    print("#"*60)
    
    success = run_command([
        sys.executable, str(script_dir / "analyze_model.py"),
        "--model", str(checkpoint_path),
        "--dataset", str(dataset_path),
        "--output", str(analysis_path),
        "--num-samples", str(config.num_analysis_samples),
        "--rollout-steps", str(config.num_rollout_steps)
    ], "Analyzing model for strategic understanding")
    
    results["steps"]["analysis"] = {
        "success": success,
        "output_path": str(analysis_path)
    }
    
    # Load analysis results if available
    analysis_file = analysis_path / "analysis_results.json"
    if analysis_file.exists():
        with open(analysis_file) as f:
            results["analysis_results"] = json.load(f)
    
    # Step 5: Generate Final Report
    print("\n\n" + "#"*60)
    print("# PHASE 5: FINAL REPORT")
    print("#"*60)
    
    end_time = datetime.now()
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = (end_time - start_time).total_seconds()
    results["success"] = all(
        step.get("success", False) 
        for step in results["steps"].values()
    )
    
    # Generate report
    generate_report(results, output_path)
    save_results(results, output_path)
    
    return results


def save_results(results: dict, output_path: Path):
    """Save results to JSON"""
    with open(output_path / "pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)


def generate_report(results: dict, output_path: Path):
    """Generate human-readable report"""
    
    report_lines = [
        "="*60,
        "DEEPRTS GAMENGEN EXPERIMENT REPORT",
        "="*60,
        "",
        f"Date: {results.get('start_time', 'N/A')}",
        f"Duration: {results.get('duration_seconds', 0):.1f} seconds",
        f"Overall Success: {'✓' if results.get('success') else '❌'}",
        "",
        "-"*60,
        "CONFIGURATION",
        "-"*60,
    ]
    
    config = results.get("config", {})
    report_lines.extend([
        f"  Episodes: {config.get('num_episodes', 'N/A')}",
        f"  Map: {config.get('map_name', 'N/A')}",
        f"  Frame Size: {config.get('frame_size', 'N/A')}x{config.get('frame_size', 'N/A')}",
        f"  Training Epochs: {config.get('num_epochs', 'N/A')}",
        "",
        "-"*60,
        "PIPELINE STEPS",
        "-"*60,
    ])
    
    for step_name, step_data in results.get("steps", {}).items():
        status = "✓" if step_data.get("success") else "❌"
        report_lines.append(f"  {status} {step_name}")
    
    # Analysis results
    if "analysis_results" in results:
        report_lines.extend([
            "",
            "-"*60,
            "ANALYSIS RESULTS",
            "-"*60,
        ])
        
        analysis = results["analysis_results"]
        
        if "action_sensitivity" in analysis:
            sens = analysis["action_sensitivity"]
            report_lines.append(f"  Action Sensitivity: {sens.get('mean_difference', 0):.6f}")
        
        if "counterfactual" in analysis:
            cf = analysis["counterfactual"]
            report_lines.append(f"  Counterfactual (shuffled): {cf.get('shuffled_diff', 0):.6f}")
            report_lines.append(f"  Counterfactual (noise): {cf.get('noise_diff', 0):.6f}")
        
        if "strategic_consistency" in analysis:
            sc = analysis["strategic_consistency"]
            report_lines.append(f"  Strategic Consistency: {sc.get('mean_consistency', 0):.6f}")
    
    # Verdict
    report_lines.extend([
        "",
        "-"*60,
        "VERDICT",
        "-"*60,
    ])
    
    if "analysis_results" in results:
        analysis = results["analysis_results"]
        issues = 0
        
        if analysis.get("action_sensitivity", {}).get("mean_difference", 1) < 0.01:
            issues += 1
        if analysis.get("counterfactual", {}).get("shuffled_diff", 1) < 0.01:
            issues += 1
        if analysis.get("strategic_consistency", {}).get("mean_consistency", 1) <= 0:
            issues += 1
        
        if issues == 0:
            verdict = "Model shows signs of UNDERSTANDING game logic"
        elif issues <= 1:
            verdict = "Model shows MIXED results - partial understanding"
        else:
            verdict = "Model appears to be PATTERN-MATCHING, not understanding strategy"
        
        report_lines.append(f"  {verdict}")
    else:
        report_lines.append("  Analysis not completed - unable to determine")
    
    report_lines.extend([
        "",
        "-"*60,
        "OUTPUT FILES",
        "-"*60,
        f"  Dataset: {output_path}/dataset/",
        f"  Model: {output_path}/model/",
        f"  Generated: {output_path}/generated/rollout.gif",
        f"  Analysis: {output_path}/analysis/analysis_results.json",
        f"  Full Results: {output_path}/pipeline_results.json",
        "",
        "="*60,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Print report
    print(report_text)
    
    # Save report
    with open(output_path / "REPORT.txt", "w") as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {output_path}/REPORT.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete DeepRTS GameNGen pipeline"
    )
    parser.add_argument("--output", type=str, default="./experiment_results",
                       help="Output directory for all results")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")
    parser.add_argument("--map", type=str, default="15x15-2-FFA",
                       help="DeepRTS map")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test run with minimal data")
    
    args = parser.parse_args()
    
    if args.quick_test:
        config = PipelineConfig(
            output_dir=args.output,
            num_episodes=5,
            num_epochs=2,
            num_analysis_samples=10,
            num_rollout_steps=10,
            num_generated_frames=10
        )
        print("Running QUICK TEST mode (minimal data)")
    else:
        config = PipelineConfig(
            output_dir=args.output,
            num_episodes=args.episodes,
            num_epochs=args.epochs,
            map_name=args.map
        )
    
    if not check_dependencies():
        sys.exit(1)
    
    results = run_pipeline(config)
    
    if results["success"]:
        print("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
