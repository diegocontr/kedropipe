#!/usr/bin/env python3
"""Model Validation Pipeline Runner

This script demonstrates how to run the model validation pipeline with different configurations.
"""

import subprocess
import sys
from pathlib import Path


def run_kedro_command(cmd_args, description=""):
    """Run a kedro command and handle output."""
    if description:
        print(f"\n{description}")
        print("=" * len(description))

    cmd = ["kedro"] + cmd_args
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Success!")
        if result.stdout.strip():
            print(
                "Output:",
                result.stdout.strip()[:500] + "..."
                if len(result.stdout) > 500
                else result.stdout.strip(),
            )
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print(
            "Error output:",
            e.stderr.strip()[:500] + "..." if len(e.stderr) > 500 else e.stderr.strip(),
        )
        return False


def main():
    """Main function to demonstrate pipeline usage."""
    print("🔍 Model Validation Pipeline Runner")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("kedro_config.yml").exists():
        print("❌ Please run this script from the kedropipe root directory")
        sys.exit(1)

    print("\n📋 Available options:")
    print("1. Run full model validation pipeline")
    print("2. Run specific pipeline nodes")
    print("3. Run with different target column")
    print("4. Check pipeline structure")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        # Run full pipeline
        success = run_kedro_command(
            ["run", "--pipeline", "model_validation"],
            "Running full model validation pipeline",
        )

        if success:
            print("\n📊 Check the following locations for outputs:")
            print("- data/reporting/ - Visualization reports")
            print("- data/processed/ - Processed datasets and metrics")

    elif choice == "2":
        # Run specific nodes
        print("\n📋 Available nodes:")
        print("- generate_predictions_node")
        print("- prepare_validation_data_node")
        print("- calculate_metrics_node")
        print("- model_comparison_node")
        print("- generate_reports_node")

        node_name = input("\nEnter node name: ").strip()
        run_kedro_command(["run", "--nodes", node_name], f"Running node: {node_name}")

    elif choice == "3":
        # Run with different target
        print("\n📋 Available targets (based on synthetic data):")
        print("- target_A")
        print("- target_B (default)")
        print("- target_C")

        target = input("\nEnter target column: ").strip()
        run_kedro_command(
            [
                "run",
                "--pipeline",
                "model_validation",
                "--params",
                f"model_validation.target_column:{target}",
            ],
            f"Running pipeline with target column: {target}",
        )

    elif choice == "4":
        # Check pipeline structure
        run_kedro_command(
            ["viz", "--pipeline", "model_validation"],
            "Generating pipeline visualization (if kedro-viz is installed)",
        )

    else:
        print("❌ Invalid choice")
        sys.exit(1)


if __name__ == "__main__":
    main()
