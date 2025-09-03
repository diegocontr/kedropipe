#!/usr/bin/env python3
"""
Script to demonstrate accessing MLflow artifacts from the model validation pipeline.

This script shows how to:
1. Load the model from MLflow
2. Access validation plot artifacts
3. View experiment metrics and parameters
"""

import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient
import json

def main():
    """Demonstrate MLflow integration for model validation."""
    
    # Read the latest model metrics to get run ID
    try:
        with open('data/models/model_metrics.txt', 'r') as f:
            model_metrics = json.loads(f.read())
        
        run_id = model_metrics.get('mlflow_run_id')
        if not run_id:
            print("No MLflow run ID found in model metrics")
            return
            
        print(f"📊 MLflow Run ID: {run_id}")
        print("="*50)
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Get run information
        run = client.get_run(run_id)
        print(f"🏃 Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        print(f"📅 Start Time: {run.info.start_time}")
        print(f"⏱️  Status: {run.info.status}")
        
        # Display key metrics
        print("\n📈 Model Metrics:")
        key_metrics = ['train_mae', 'test_mae', 'train_rmse', 'test_rmse', 'best_iteration']
        for metric in key_metrics:
            if metric in run.data.metrics:
                print(f"   {metric}: {run.data.metrics[metric]:.4f}")
        
        # Display parameters
        print("\n⚙️  Model Parameters:")
        key_params = ['loss_function', 'iterations', 'learning_rate', 'depth']
        for param in key_params:
            if param in run.data.params:
                print(f"   {param}: {run.data.params[param]}")
        
        # List artifacts
        print("\n📁 Artifacts:")
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            print(f"   📄 {artifact.path}")
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_artifact in sub_artifacts:
                    print(f"      📊 {sub_artifact.path}")
        
        # Show how to load the model
        print(f"\n🤖 Model Loading:")
        model_uri = f"runs:/{run_id}/catboost_model"
        print(f"   Model URI: {model_uri}")
        print("   To load: mlflow.catboost.load_model(model_uri)")
        
        # Show how to download artifacts
        print(f"\n📥 Artifact Access:")
        validation_plots_path = f"validation_plots"
        print(f"   To download validation plots:")
        print(f"   client.download_artifacts('{run_id}', '{validation_plots_path}')")
        
        print(f"\n🌐 MLflow UI:")
        print("   Start the MLflow UI with: mlflow ui")
        print("   Then visit: http://localhost:5000")
        
        # Verify model can be loaded
        try:
            model = mlflow.catboost.load_model(model_uri)
            print(f"\n✅ Model successfully loaded from MLflow!")
            print(f"   Model type: {type(model)}")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model training pipeline has been run with MLflow integration.")

if __name__ == "__main__":
    main()
