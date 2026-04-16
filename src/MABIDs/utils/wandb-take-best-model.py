import os

import wandb

DEFAULT_WANDB_ENTITY_NAME = "prograf-uff"
DEFAULT_WANDB_PROJECT_NAME = "chemical-analysis-phosphate"
DEFAULT_WANDB_SWEEP_ID = "46skucmg"

MODEL_DIR = "artifacts"
MODEL_WANDB_PATH = "checkpoint/PhosphateNetworkSqueezeNetStyle.ckpt"
MODEL_WANDB_TYPE = "experiments"
MODEL_NAME = f"__Exp_H2O"

METRIC_BEST_RUN = "MAE/Test/Epoch"

if __name__ == "__main__":
    api = wandb.Api()
    sweep = api.sweep(f"{DEFAULT_WANDB_ENTITY_NAME}/{DEFAULT_WANDB_PROJECT_NAME}/{DEFAULT_WANDB_SWEEP_ID}")
    finished_runs = list(filter(lambda run: run.state == "finished", sweep.runs))
    best_run = sorted(finished_runs, key=lambda run: run.summary[METRIC_BEST_RUN])[0]
    os.makedirs(os.path.join(MODEL_DIR, MODEL_WANDB_PATH.split("/")[0]), exist_ok=True)
    model_saved = best_run.file(MODEL_WANDB_PATH).download(root=MODEL_DIR, replace=True)

    wandb.init(entity=DEFAULT_WANDB_ENTITY_NAME, project=DEFAULT_WANDB_PROJECT_NAME)
    model_metadata = {"metric": best_run.summary[METRIC_BEST_RUN]}
    try:
        model_best_metric = wandb.use_artifact(f"{MODEL_NAME}:latest").metadata["metric"]
        if best_run.summary[METRIC_BEST_RUN] <= model_best_metric:
            model_artifact = wandb.Artifact(MODEL_NAME, type=MODEL_WANDB_TYPE, metadata=model_metadata)
            model_artifact.add_file(model_saved.name)
            wandb.log_artifact(model_artifact, aliases=["latest"])
    except wandb.errors.CommError:
        model_artifact = wandb.Artifact(MODEL_NAME, type=MODEL_WANDB_TYPE, metadata=model_metadata)
        model_artifact.add_file(model_saved.name)
        wandb.log_artifact(model_artifact, aliases=["latest"])
