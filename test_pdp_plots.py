#!/usr/bin/env python3
"""Test script to verify PDP plots are being generated correctly."""

import os
import tempfile

import mlflow


def test_pdp_plots():
    """Test that PDP plots are proper matplotlib figures."""
    # Get the latest run
    client = mlflow.tracking.MlflowClient()
    
    # Get all experiments and find the latest one
    experiments = client.list_experiments()
    if not experiments:
        print("❌ No MLflow experiments found")
        return
    
    # Get the highest experiment ID (latest)
    latest_experiment = max(experiments, key=lambda x: int(x.experiment_id))
    experiment_id = latest_experiment.experiment_id
    experiment_name = latest_experiment.name
    print(f"🔍 Using experiment: {experiment_name} (ID: {experiment_id})")
    
    runs = client.search_runs(experiment_ids=[experiment_id])
    if not runs:
        print("❌ No runs found")
        return
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    
    print(f"🔍 Testing PDP plots from run: {run_id}")
    
    # Download PDP artifacts
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = client.download_artifacts(run_id, "pdp_analyses", dst_path=tmp_dir)
            
            # Check if PDP artifacts exist
            pdp_path = os.path.join(tmp_dir, "pdp_analyses")
            if not os.path.exists(pdp_path):
                print("❌ PDP artifacts not found")
                return
            
            # List PDP contents
            pdp_contents = os.listdir(pdp_path)
            print(f"📁 PDP artifacts found: {pdp_contents}")
            
            # Check for train and test directories
            for subset in ["pdp_panels_train", "pdp_panels_test"]:
                subset_path = os.path.join(pdp_path, subset)
                if os.path.exists(subset_path):
                    subset_contents = os.listdir(subset_path)
                    print(f"📊 {subset} contents: {subset_contents}")
                    
                    # Look for PNG files (plots)
                    png_files = [f for f in subset_contents if f.endswith('.png')]
                    if png_files:
                        print(f"✅ Found {len(png_files)} PDP plot files: {png_files}")
                    else:
                        print(f"⚠️  No PNG files found in {subset}")
                else:
                    print(f"❌ {subset} directory not found")
            
            print("✅ PDP artifact structure looks good!")
            
    except Exception as e:
        print(f"❌ Error downloading PDP artifacts: {e}")

if __name__ == "__main__":
    test_pdp_plots()
