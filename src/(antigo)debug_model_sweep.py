import torch
import wandb
import yaml
import os
from MABIDs import chemical_analysis as ca
import numpy as np

from typing import Dict, List
from wandb.wandb_run import Run
from pytorch_lightning import Trainer
from engine.lightning import Dataset, BaseModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from engine.models import *

os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".

# reduces mat mul precision (for performance)
torch.set_float32_matmul_precision('high')

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(f"{BASE_DIR}/settings.yaml"), "r") as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)
    # global variables
    ANALYTE = settings["analyte"]
    CACHE_DIR = os.path.join("..", "cache_dir", ANALYTE)
    SAMPLES_DIR = settings["samples_dir"]
    SWEEP_ID = settings["sweep_id"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    RETURN_NODE = settings["return_node"]
    CLASSIFIER_MODEL = settings["classifier_model"]
    # training hyperparams
    MAX_EPOCHS = settings["model"]["max_epochs"]
    #TODO ativar reduce on plateau no base model (optmizer)
    LR_PATIENCE = settings["model"]["learning_rate_patience"]  # reduce on plateau technique
    EARLY_STOP_PATIENCE = 2*LR_PATIENCE + 1                    # early stop technique
    LOSS_FUNCTION = settings["model"]["loss_function"]

CLASSIFIERS = {
    "mlp1": {"model": MLP1, "requires_flatten": True},
    "DynamicMLP": {"model": DynamicMLP, "requires_flatten": True},
    "squeezenet": {"model": SqueezeNetClassifier, "requires_flatten": False},
              }

CLASSIFIER_CONFIG = {
                    "model_class": CLASSIFIERS[CLASSIFIER_MODEL]["model"],
                     "requires_flatten": CLASSIFIERS[CLASSIFIER_MODEL]["requires_flatten"]
                    }

LOSSES_FUNCTIONS = {'cross_entropy': nn.CrossEntropyLoss()}
CHOSEN_LOSS = LOSSES_FUNCTIONS.get(LOSS_FUNCTION)
# # presaved devices
# with open(os.path.join("src/devices_mapper.yaml"), "r") as f:
#     devices = yaml.load(f, Loader=yaml.FullLoader)

# reads sweep configs yaml
with open('src/sweep_config.yaml') as f:
        SWEEP_CONFIGS = yaml.load(f, Loader=yaml.FullLoader)


def main():

        # checkpoint callback setting
        #checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_ROOT, filename= run.name, save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=False, save_weights_only=True)#every_n_epochs=CHECKPOINT_SAVE_INTERVAL)

        # load and prepare dataset
        #samples, processed_samples, mapper = prepare_samples_dataset(ANALYTE, SAMPLES_DIR, CACHE_DIR, devices)
    configs = {"batch_size": 16}
    data_module = Dataset(ANALYTE, configs)
    data_module.prepare_data()
    num_classes = data_module.num_classes

    with open(os.path.join(BASE_DIR, "..", f"processed_dataset/{ANALYTE}_metadata.yaml"), "r") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    input_dim = metadata["num_channels"]

    # load model
    model = BaseModel(classifier_config=CLASSIFIER_CONFIG, input_dim=input_dim, loss_function=CHOSEN_LOSS, learning_rate=0.0001, learning_rate_patience=LR_PATIENCE, num_classes=num_classes)
    # define trainer settings
    trainer = Trainer(#callbacks=[EarlyStopping(monitor="test_loss", mode="min")], logger=logger)
                    accelerator = "cpu",
                    max_epochs = MAX_EPOCHS,
                    callbacks = [#checkpoint_callback,
                                LearningRateMonitor(logging_interval='epoch'),
                                EarlyStopping(
                                            monitor="Loss/Val",
                                            mode="min",
                                            patience= EARLY_STOP_PATIENCE
                                        ),],
                    gradient_clip_val = 0.5,
                    gradient_clip_algorithm = "value",  # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping
                    log_every_n_steps = 1,
                    num_sanity_val_steps = 0,
                    enable_progress_bar = True,
                    detect_anomaly = True,
                    )
    # fit a model
    trainer.fit(model=model, datamodule=data_module)#, train_dataloaders=dataset

if __name__ == "__main__":
    main()