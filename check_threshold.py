import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

metrics = client.get_run(run_id).data.metrics
accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)

if accuracy < 0.85:
    print("❌ Accuracy below threshold!")
    sys.exit(1)
else:
    print("✅ Accuracy above threshold. Continue...")