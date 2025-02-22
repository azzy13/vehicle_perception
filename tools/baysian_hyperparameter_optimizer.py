import optuna
import optuna.visualization as vis
import json
import os
import torch
from yolox.exp import get_exp
from yolox.evaluators import KittiEvaluator
from types import SimpleNamespace

# Base arguments for evaluator setup
exp_file = "exps/example/mot/merge_dataset.py"
ckpt_file = "pretrained/latest_ckpt.pth.tar"
result_folder = "YOLOX_outputs/merge_dataset/track_results"
ground_truth_folder = "YOLOX_outputs/merge_dataset/ground_truth"
metric_file = os.path.join(result_folder, "metrics.json")

# Function to parse MOTA from `metric.json`
def parse_mota_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        return metrics["mota"]["OVERALL"]  # Extract MOTA (OVERALL) directly
    except Exception as e:
        print(f"Error reading MOTA from {file_path}: {e}")
        return float("-inf")  # Return a very low value if MOTA isn't found


def objective(trial):
    # Suggest parameters for optimization
    params = {
        "--conf": trial.suggest_float("conf", 0.1, 0.9),
        "--nms": trial.suggest_float("nms", 0.1, 0.9),
        "--track_thresh": trial.suggest_float("track_thresh", 0.1, 0.9),
        "--match_thresh": trial.suggest_float("match_thresh", 0.1, 0.9),
        "--track_buffer": trial.suggest_categorical("track_buffer", [30, 60, 120, 300, 600]),
        "--min-box-area": trial.suggest_categorical("min-box-area", [10, 100, 150]),
    }

    # Load experiment and model
    exp = get_exp(exp_file, exp_name="merge_dataset")
    model = exp.get_model()
    model.eval()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    model.to(device)

    # Load model weights
    model.load_state_dict(torch.load(ckpt_file, map_location=device)["model"])

    # Convert args dictionary to SimpleNamespace for attribute-style access
    args = SimpleNamespace(
        track_thresh=params["--track_thresh"],
        track_buffer=params["--track_buffer"],
        match_thresh=params["--match_thresh"],
        min_box_area=params["--min-box-area"],
        ctra=False,  # Toggle for CTRA if needed
        mot20=False,
    )

    # Set up dataloader and evaluator
    dataloader = exp.get_eval_loader(batch_size=1, is_distributed=False)
    evaluator = KittiEvaluator(
        args=args,
        dataloader=dataloader,
        img_size=(800, 1440),
        confthre=params["--conf"],
        nmsthre=params["--nms"],
        num_classes=exp.num_classes,
    )

    # Run evaluation
    print(f"Running evaluation with parameters: {params}")
    evaluator.evaluate(
        model=model,
        result_folder=result_folder,
        ground_truth_folder=ground_truth_folder,
    )

    # Parse MOTA from `metric.json`
    mota = parse_mota_from_file(metric_file)
    print(f"Parsed MOTA: {mota}")

    return mota


# Create a study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)  # Adjust n_trials for more thorough optimization

# Save study results
study.trials_dataframe().to_csv("optuna_study_results.csv", index=False)

# Save the best parameters
best_params_file = "optuna_best_params.json"
with open(best_params_file, "w") as f:
    json.dump(study.best_params, f, indent=4)
print(f"Best parameters saved to {best_params_file}")

# Visualize optimization history
vis.plot_optimization_history(study).show()
