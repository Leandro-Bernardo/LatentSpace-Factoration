import torch, os, yaml
from torch.optim import SGD
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch import Generator, tensor, from_numpy
from typing import Any, Dict, List, Tuple
from .models import *
import chemical_analysis as ca
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "..", "settings.yaml"), "r") as f:
    data_settings = yaml.load(f, Loader=yaml.FullLoader)
try:
    with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "r") as f:
        devices = yaml.load(f, Loader=yaml.FullLoader)
except:
    devices = {} # Lazy Initialization


class Preprocessing():
    def __init__(self, analyte: str, sample_dir: str, cache_dir: str, devices: Dict[str, Dict[str, int]], backbone: str, requires_flatten: bool, return_node: Optional[str] = None, frozen_weights: Optional[bool] = True):
        self.analyte = analyte
        self.sample_dir = sample_dir
        self.cache_dir = cache_dir
        self.devices = devices  # Lazy Initialization
        self.save_path = os.path.join(os.path.dirname(__file__), "..")
        self.feature_extractor = FeatureExtractor(backbone=backbone, return_node=return_node, freeze=frozen_weights, requires_flatten=requires_flatten)


    def prepare_samples_dataset(self):

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

        sample_dataset = preprocessing[self.analyte]["dataset"]
        processed_dataset = preprocessing[self.analyte]["processed_dataset"]

        # samples preprocessing
        samples = sample_dataset(
            base_dirs = self.sample_dir,
            progress_bar = True,
            skip_blank_samples = True,
            skip_incomplete_samples = True,
            skip_inference_sample= True,
            skip_training_sample = False,
            verbose = True
        )
        if self.analyte in pca_stats.keys(): # does have PCA
            processed_samples = processed_dataset(
                    dataset = samples,
                    cache_dir = self.cache,
                    num_augmented_samples = 0,
                    progress_bar = True,
                    transform = None,
                    lab_mean= pca_stats[f"{self.analyte}"]['lab_mean'],
                    lab_sorted_eigenvectors = pca_stats[f"{self.analyte}"]['lab_sorted_eigenvectors'])
        else: # doenst have PCA
            processed_samples = processed_dataset(
                dataset = samples,
                cache_dir = self.cache_dir,
                num_augmented_samples = 0,
                progress_bar = True,
                transform = None, )

        assert len(samples) == len(processed_samples), "samples and processed samples missmatch size"

        # TODO otimizar para ter menos loops
        current_samples_devices = set([i.sample.get("device")["model"].lower() for i in processed_samples])

        if self.analyte not in self.devices.keys():
            self.devices[f"{self.analyte}"] = {model:i for model, i in enumerate(current_samples_devices, start=0)}
            with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(self.devices, f, sort_keys=False, allow_unicode=True)

        #TODO verificar caso que o celular nao existe mais no dataset (excluir e resetar a contagem / resetar o analito sempre que for usado (no preprocessamento))
        for model in current_samples_devices:
            if model not in self.devices.get(self.analyte).values():
                idx = max(self.devices.get(self.analyte).keys()) + 1
                self.devices.get(self.analyte)[idx] = model
            with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(self.devices, f, sort_keys=False, allow_unicode=True)

        # extract features with selected backbone
        features = []
        labels = []
        num_classes = len(current_samples_devices)
        one_hot_encodding = torch.nn.functional.one_hot(torch.arange(0, num_classes), num_classes=num_classes)
        name_to_idx = {name.lower(): idx for idx, name in self.devices.get(self.analyte).items()}
        # TODO otimizar para processar com GPU e batches
        for processed_sample in processed_samples:
            # process input X (pmf)
            pmf_tensor = torch.tensor(processed_sample.calibrated_pmf).unsqueeze(0)
            features.append(self.feature_extractor(pmf_tensor))
            # process output y (cellphone model)
            sample_device = processed_sample.sample.get("device")["model"].lower()
            sample_device_idx = name_to_idx.get(sample_device)
            labels.append(one_hot_encodding[sample_device_idx])
        # salvar os dados como memmap


class Dataset(LightningDataModule): #Trocar para DataModule?
    def __init__(self, saved_samples_path: str, **kwargs ): #samples, processed_samples, mapper: Dict, args, **kwags):
        super().__init__()
        # TODO apagar desnecessarios
        # self.samples = samples
        # self.processed_samples = processed_samples
        # self.current_analyte = data_settings['analyte']
        # self.mapper = devices[current_analyte]
        # self.one_hot = torch.nn.functional.one_hot(torch.arange(0, num_class), num_classes=num_class)
        self.saved_samples_path = saved_samples_path

    def prepare_data(self):
        try:
            torch.load(self.saved_samples_path) # TODO carregar untyped storage data aqui
        except:
            import shutil

            analyte = data_settings["analyte"]
            sample_dir = data_settings["samples_dir"]
            cache_dir = os.path.join("..", "cache_dir", analyte)
            feature_extractor = data_settings['feature_extractor']
            requires_flatten = True if data_settings['classifier_model'] != "squeezenet" else False
            return_node = data_settings.get('return_node') if isinstance(data_settings.get('return_node'), str) else None
            frozen_weights = data_settings['frozen_weights']
            # empty cache dir from previous sweep
            try:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)

                os.makedirs(cache_dir, exist_ok=False)

            except OSError as e:
                raise RuntimeError(f"Could not prepare cache directory {cache_dir}") from e

            preprocessing = Preprocessing(analyte=analyte,
                                          sample_dir=sample_dir,
                                          cache_dir=cache_dir,
                                          devices=devices,
                                          backbone=feature_extractor,
                                          requires_flatten=requires_flatten,
                                          return_node=return_node,
                                          frozen_weights=frozen_weights,)
            preprocessing.prepare_samples_dataset()

            # carregar memmap data aqui

        # TODO ajustar a partir daqui (ja que a extracao foi feita na classe Preprocessing())
        for processed_sample in self.processed_samples:
            self.true_class_value.append(self.one_hot[self.mapper.get(processed_sample.sample.get("device")["model"])]) # Classe (Modelo celular)
            self.samples_pmf.append(processed_sample.calibrated_pmf) # Entrada (PMF)
        self.true_class_value = tensor(self.true_class_value)
        self.samples_pmf = from_numpy(self.samples_pmf)

    def setup(self, stage:str):
        len_dataset = len(self.processed_samples)
        n_train = int(0.6*len_dataset)
        n_val = int(0.2*len_dataset)
        n_test = len_dataset - n_train - n_val

        train_samples, val_samples, test_samples = random_split(self.samples, [n_train, n_val, n_test], generator = Generator().manual_seed(42))
        train_models, val_models, test_models = random_split(self.models, [n_train, n_val, n_test], generator = Generator().manual_seed(42))

        if stage == "fit":
            self.dataset_train = TensorDataset(train_samples, train_models)

        elif stage == "validate":
            self.dataset_val = TensorDataset(val_samples, val_models)

        elif stage == "test":
            self.dataset_test = TensorDataset(test_samples, test_models)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = 32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size = 32, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=32, shuffle=False)

class BaseModel(LightningModule):
    def __init__(self, *, classifier_config: Dict[str, Any], batch_size: int, loss_function: torch.nn.Module, learning_rate: float, learning_rate_patience: int = None, frozen_weights: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.classifier_config = classifier_config
        self.classifier = None  # Lazy Initialization
        self.criterion = loss_function
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        # self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
        #                                                         "MAE": MeanAbsoluteError(),
        #                                                         "MAPE": MeanAbsolutePercentageError(),
        #                                                         "MSE": MeanSquaredError(),
        #                                                         #"WMAPE": WeightedMeanAbsolutePercentageError(),
        #                                                         #"SMAPE": SymmetricMeanAbsolutePercentageError(),
        #                                                        }) for mode_name in ["Train", "Val", "Test"]})
        #self.early_stopping_patience = early_stopping_patience

    def setup(self, stage: str):

        devices_path = os.path.join(os.path.dirname(__file__), "..", "devices.yaml")
        with open(devices_path, "r") as f:
            devices = yaml.load(f, Loader=yaml.FullLoader)

        analyte = data_settings['analyte']
        num_class = len(devices[analyte])

        self.classifier = self.classifier_config["model_class"](num_class=num_class)

    def configure_optimizers(self):
        self.optimizer = SGD(self.parameters(), lr = self.learning_rate)
        # self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.learning_rate_patience)

        return {f"optimizer: self.optimizer"}
        # return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.reduce_lr_on_plateau, "monitor": "Loss/Val"}}
        #return [self.optmizer], [self.reduce_lr_on_plateau]

    # def configure_callbacks(self) -> List[Callback]:
    # # Apply early stopping.
    #  return [EarlyStopping(monitor="Loss/Val", mode="min", patience=self.early_stopping_patience)]

    def forward(self, x: Any):
        x = self.classifier(x)

        return x

    # TODO verificar o dado de entrada
    #defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, torch.tensor], stage: str):
        X, y = batch[0].squeeze(), batch[1].squeeze()
        predicted_value = self(X)    # o proprio objeto de BaseModel é o modelo (https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
        predicted_value = predicted_value.squeeze()
        # Compute and log the loss value.
        loss = self.criterion(predicted_value, y)
        self.log(f"Loss/{stage}", loss, prog_bar=True)
        # Compute and log step metrics.
        # metrics: MetricCollection = self.metrics[stage]  # type: ignore
        # self.log_dict({f'{metric_name}/{stage}/Step': value for metric_name, value in metrics(predicted_value, y).items()})
        return loss

    def training_step(self, batch: List[torch.tensor]):#, batch_idx: int):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: List[torch.tensor]):#, batch_idx: int):
        return self._any_step(batch, "Val")

    def test_step(self, batch: List[torch.tensor]):#, batch_idx: int):
        return self._any_step(batch, "Test")

    # def _any_epoch_end(self, stage: str):
    #     metrics: MetricCollection = self.metrics[stage]  # type: ignore
    #     self.log_dict({f'{metric_name}/{stage}/Epoch': value for metric_name, value in metrics.compute().items()}, on_step=False, on_epoch=True) # logs metrics on epoch end
    #     metrics.reset()
        # Print loss at the end of each epoch
        #loss = self.trainer.callback_metrics[f"Loss/{stage}"]
        #print(f"Epoch {self.current_epoch} - Loss/{stage}: {loss.item()}")

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")