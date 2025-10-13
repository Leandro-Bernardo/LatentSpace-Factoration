import torch
import wandb
import yaml
import os
import chemical_analysis as ca
import numpy as np
import shutil

from typing import Dict, List
from wandb.wandb_run import Run
from pytorch_lightning import Trainer
from engine.classifier import vanilla
from engine.lightning import Dataset, BaseModel
from engine.feature_extractor import squeezenet1_1
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

os.environ["WANDB_CONSOLE"] = "off"  # Needed to avoid "ValueError: signal only works in main thread of the main interpreter".

# reduces mat mul precision (for performance)
torch.set_float32_matmul_precision('high')


with open(os.path.join(".", "settings.yaml"), "r") as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)
    # global variables
    ANALYTE = settings["analyte"]
    CACHE_DIR = os.path.join("..", "cache_dir", ANALYTE)
    SAMPLES_DIR = settings["samples_dir"]
    FEATURE_EXTRACTOR = settings["feature_extractor"]
    SWEEP_ID = settings["sweep_id"]
    MODEL = settings["mlp_model"]
    # training hyperparams
    MAX_EPOCHS = settings["model"]["max_epochs"]
    LR = settings["model"]["learning_rate"]
    LR_PATIENCE = settings["model"]["learning_rate_patience"]
    EARLY_STOP_PATIENCE = 2*LR_PATIENCE + 1
    LOSS_FUNCTION = settings["model"]["loss_function"]
    GRADIENT_CLIPPING = settings["model"]["gradient_clipping"]

# presaved devices
with open(os.path.join("devices_mapper.yaml"), "r") as f:
    devices = yaml.load(f, Loader=yaml.FullLoader)

# reads sweep configs yaml
with open('./sweep_config.yaml') as f:
        SWEEP_CONFIGS = yaml.load(f, Loader=yaml.FullLoader)

# empy cache dir
try:
    shutil.rmtree(CACHE_DIR)
    #print(f"Directory '{CACHE_DIR}' and its contents deleted successfully.")
    os.makedirs(CACHE_DIR, exist_ok = False)
except OSError as e:
    raise f"Error: {e}, manually delete {CACHE_DIR}"

# network options
networks_choices = {"vanilla": vanilla}
MODEL_NETWORK = networks_choices[MODEL]

def prepare_samples_dataset(analyte:str, dir:str, cache:str, devices: Dict[str, Dict[str, int]]):

    preprocessing = {
                "alkalinity":{"dataset": ca.alkalinity.AlkalinitySampleDataset, "processed_dataset": ca.alkalinity.ProcessedAlkalinitySampleDataset},
                "chloride": {"dataset": ca.chloride.ChlorideSampleDataset, "processed_dataset": ca.chloride.ProcessedChlorideSampleDataset},
                "sulfate": {"dataset": ca.sulfate.SulfateSampleDataset, "processed_dataset": ca.sulfate.ProcessedSulfateSampleDataset},
                "phosphate": {"dataset": ca.phosphate.PhosphateSampleDataset, "processed_dataset": ca.phosphate.ProcessedPhosphateSampleDataset},
                "bisulfite": {"dataset": ca.bisulfite2d.Bisulfite2DSampleDataset, "processed_dataset": ca.bisulfite2d.ProcessedBisulfite2DSampleDataset},
                "iron2": {"dataset": ca.iron2.Iron2SampleDataset, "processed_dataset": ca.iron2.ProcessedIron2SampleDataset},
                "iron3": {"dataset": ca.iron3.Iron3SampleDataset, "processed_dataset": ca.iron3.ProcessedIron3SampleDataset},
                "iron_oxid": {"dataset": ca.iron_oxid.IronOxidSampleDataset, "processed_dataset": ca.iron_oxid.ProcessedIronOxidSampleDataset},
                "ph": {"dataset": ca.ph.PhSampleDataset, "processed_dataset": ca.ph.ProcessedPhSampleDataset},
                }

    #TODO resolver cenário em que não tem pca previamente calculado
    pca_stats = {
            "bisulfite"  : {"lab_mean": np.load(ca.bisulfite2d.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.bisulfite2d.PCA_STATS)['lab_sorted_eigenvectors']},
            "chloride"  : {"lab_mean": np.load(ca.chloride.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.chloride.PCA_STATS)['lab_sorted_eigenvectors']},
            "iron2"  : {"lab_mean": np.load(ca.iron2.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron2.PCA_STATS)['lab_sorted_eigenvectors']},
            "iron3"  : {"lab_mean": np.load(ca.iron3.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron3.PCA_STATS)['lab_sorted_eigenvectors']},
            #"iron_oxid"  : {"lab_mean": np.load(ca.iron_oxid.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron_oxid.PCA_STATS)['lab_sorted_eigenvectors']},
            "phosphate"  : {"lab_mean": np.load(ca.phosphate.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.phosphate.PCA_STATS)['lab_sorted_eigenvectors']},
            }

    sample_dataset = preprocessing[analyte]["dataset"]
    processed_dataset = preprocessing[analyte]["processed_dataset"]

    # samples preprocessing
    samples = sample_dataset(
        base_dirs = dir,
        progress_bar = True,
        skip_blank_samples = False,
        skip_incomplete_samples = True,
        skip_inference_sample= True,
        skip_training_sample = False,
        verbose = True
    )
    if analyte in pca_stats.keys(): # does have PCA
        processed_samples = processed_dataset(
                dataset = samples,
                cache_dir = cache,
                num_augmented_samples = 0,
                progress_bar = True,
                transform = None,
                lab_mean= pca_stats[f"{analyte}"]['lab_mean'],
                lab_sorted_eigenvectors = pca_stats[f"{analyte}"]['lab_sorted_eigenvectors'])
    else: # doenst have PCA
        processed_samples = processed_dataset(
            dataset = samples,
            cache_dir = cache,
            num_augmented_samples = 0,
            progress_bar = True,
            transform = None, )

    assert len(samples) == len(processed_samples), "samples and processed samples missmatch size"

    current_samples_devices = set([i.sample.get("device")["model"] for i in processed_samples])
    if analyte not in devices.keys():
        devices[f"{analyte}"] = {model:i for model, i in enumerate(current_samples_devices, start=0)}
        with open("devices.yaml", "w", encoding="utf-8") as f:
            yaml.dump(devices, f, sort_keys=False, allow_unicode=True)

    for model in current_samples_devices:
        if model not in devices.get(analyte).keys():
            idx = max(devices.get(analyte).values()) + 1
            devices.get(analyte)[model] = idx
        with open("devices.yaml", "w", encoding="utf-8") as f:
            yaml.dump(devices, f, sort_keys=False, allow_unicode=True)

    return samples, processed_samples, devices


def main():

    # starts wandb
    with wandb.init(config=SWEEP_CONFIGS) as run:
        assert isinstance(run, Run)
        # initialize logger
        logger = WandbLogger(project=f"{ANALYTE}_latent_space_factoring", experiment=run)
        # gets sweep configs
        configs = run.config.as_dict()

        # checkpoint callback setting
        #checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_ROOT, filename= run.name, save_top_k=1, monitor='Loss/Val', mode='min', enable_version_counter=False, save_last=False, save_weights_only=True)#every_n_epochs=CHECKPOINT_SAVE_INTERVAL)

        # load and prepare dataset  #TODO usar untyped storage para evitar processar tudo novamente
        samples, processed_samples, mapper = prepare_samples_dataset(ANALYTE, SAMPLES_DIR, CACHE_DIR, devices)
        data_module = Dataset(samples, processed_samples, mapper)

        # load model

        model = BaseModel(dataset=data_module, model=MODEL_NETWORK, loss_function=LOSS_FUNCTION, batch_size=configs["batch_size"], learning_rate=configs["lr"], learning_rate_patience=LR_PATIENCE, sweep_config = configs)
        # define trainer settings
        trainer = Trainer(#callbacks=[EarlyStopping(monitor="test_loss", mode="min")], logger=logger)
                        logger = logger,
                        accelerator = "gpu",
                        max_epochs = MAX_EPOCHS,
                        callbacks = [#checkpoint_callback,
                                    LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(
                                                monitor="Loss/Val",
                                                mode="min",
                                                patience= EARLY_STOP_PATIENCE
                                            ),],
                        gradient_clip_val = configs["gradient_clip"],
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