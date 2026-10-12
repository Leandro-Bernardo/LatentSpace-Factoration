import torch
import wandb
import yaml
import os

from wandb.wandb_run import Run
from pytorch_lightning import Trainer
from engine.lightning import Dataset, BaseLightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from engine.models import *
from engine._configs import *

os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".

# reduces mat mul precision (for performance)
torch.set_float32_matmul_precision('high')

CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

def main():
    experiment_config = ExperimentConfig.from_yaml("settings.yaml")
    # starts wandb
    with wandb.init() as run:
        assert isinstance(run, Run)
        # initialize logger
        logger = WandbLogger(project=f"{experiment_config.analyte}_latent_space_factoring", experiment=run)
        # gets sweep configs
        experiment_config = experiment_config.update_from_wandb(run.config.as_dict())
        # checkpoint callback setting
        checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_SAVE_PATH, filename= run.name, save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=False, save_weights_only=True)
        # prepare the data
        data_module = Dataset(experiment_config)
        data_module.prepare_data()

        metadata = DatasetMetadata.from_yaml(f"{experiment_config.analyte}_metadata.yaml")

        # Assures the processed dataset is suited for the training configuration.
        assert (metadata.datatype == "preprocessed_pmf_feature_maps" and experiment_config.fine_tune_or_train_cnn is False) \
                    or (metadata.datatype == "preprocessed_pmf_raw_data" and experiment_config.fine_tune_or_train_cnn is True) ,\
                    f"The preexisting dataset ({metadata.datatype}) is incompatible with the current training configuration (fine-tune CNN: {experiment_config.fine_tune_or_train_cnn})."
        # load model
        model = BaseLightningModule(
                            experiment_configs=experiment_config,
                            data_module=data_module,
                            dataset_metadata=metadata
                            )
        # define trainer settings
        trainer = Trainer(#callbacks=[EarlyStopping(monitor="test_loss", mode="min")], logger=logger)
                        logger = logger,
                        accelerator = "gpu",
                        max_epochs = experiment_config.model.max_epochs,
                        callbacks = [checkpoint_callback,
                                    LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(
                                                monitor="Loss/Val",
                                                mode="min",
                                                patience= experiment_config.model.early_stopping_patience
                                            ),],
                        gradient_clip_val = experiment_config.gradient_clip,
                        gradient_clip_algorithm = "value",  # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping
                        log_every_n_steps = 1,
                        num_sanity_val_steps = 0,
                        enable_progress_bar = True,
                        detect_anomaly = True,
                        )
        # fit a model
        trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()