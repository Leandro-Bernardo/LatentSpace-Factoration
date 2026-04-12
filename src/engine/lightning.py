import torch, os, yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch import Generator
from torch.nn import ModuleDict
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from typing import Any, Dict, List, Tuple
from .models import *
from math import ceil
import chemical_analysis as ca
import numpy as np
from tqdm import tqdm

with open(os.path.join(os.path.dirname(__file__), "..", "settings.yaml"), "r") as f:
    data_settings = yaml.load(f, Loader=yaml.FullLoader)
try:
    with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "r") as f:
        devices = yaml.load(f, Loader=yaml.FullLoader)
except:
    devices = {} # Lazy Initialization


class Preprocessing():
    def __init__(self, analyte: str, sample_dir: str, cache_dir: str, devices: Dict[str, Dict[str, int]], backbone: str, return_node: Optional[str] = None, frozen_weights: Optional[bool] = True):
        self.analyte = analyte
        self.sample_dir = sample_dir
        self.cache_dir = cache_dir
        self.devices = devices
        self.save_path = os.path.join(os.path.dirname(__file__), "..")
        self.feature_extractor = FeatureExtractor(analyte=self.analyte, backbone=backbone, return_node=return_node)

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
        #TODO conferir se está tudo correto com os PCAs
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
            self.devices[f"{self.analyte}"] = {model:i for i, model in enumerate(current_samples_devices, start=0)}
            with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(self.devices, f, sort_keys=False, allow_unicode=True)

        #TODO verificar caso que o celular nao existe mais no dataset (excluir e resetar a contagem / resetar o analito sempre que for usado (no preprocessamento))
        for model in current_samples_devices:
            if model not in self.devices.get(self.analyte).keys():
                idx = max(self.devices.get(self.analyte).values()) + 1
                self.devices.get(self.analyte)[model] = idx
            with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(self.devices, f, sort_keys=False, allow_unicode=True)

        # extract features with selected backbone
        features = []
        labels = []
        shape = None
        num_classes = len(current_samples_devices)
        pretrained_model = self.feature_extractor._load_from_checkpoint()  # loads pretrained model
        pretrained_model.eval()
        # TODO otimizar para processar com GPU e batches
        for processed_sample in tqdm(processed_samples, desc = 'extracting features'):
            # process the input X (pmf)
            pmf_tensor = torch.tensor(processed_sample.calibrated_pmf).unsqueeze(0)
            current_pmf_extracted_features = pretrained_model(pmf_tensor).get('feature')
            current_pmf_extracted_features_shape = current_pmf_extracted_features.shape # shape : (batch, channel, height, width)
            features.append(current_pmf_extracted_features.detach().cpu().numpy())
            # process the output y (cellphone model)
            sample_device = processed_sample.sample.get("device")["model"].lower()
            sample_device_idx = self.devices[f"{self.analyte}"].get(sample_device)
            labels.append(sample_device_idx)
            # assures samples have same shape
            if shape != None:
                assert shape == current_pmf_extracted_features_shape
            else:
                shape = current_pmf_extracted_features_shape

        # saves the processed data as memmaps (https://github.com/numpy/numpy/blob/main/numpy/_core/memmap.py#L23-L362)
        save_path = os.path.join(os.path.dirname(__file__), "..", "..", "processed_dataset")
        os.makedirs(save_path, exist_ok=True)
        x_save_path = os.path.join(save_path, f"{self.analyte}_processed_samples.dat")
        y_save_path = os.path.join(save_path, f"{self.analyte}_labels.dat")
        N, C, H, W = len(processed_samples), current_pmf_extracted_features_shape[-3], current_pmf_extracted_features_shape[-2], current_pmf_extracted_features_shape[-1]
        x_memmap = np.memmap(x_save_path, dtype = np.float32, mode = 'w+', shape = (N, C, H, W))
        y_memmap = np.memmap(y_save_path, dtype = np.int64, mode = 'w+', shape = (N))
        # write data on memmap obj
        for i in tqdm(range(N), desc= 'saving data'):
            x_memmap[i] = features[i]
            y_memmap[i] = labels[i]

        x_memmap.flush()
        y_memmap.flush()

        # write data metadata (num classes, num samples, num feature maps (channels), height, width)
        with open(os.path.join(save_path, f"{self.analyte}_metadata.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(tuple([num_classes, N, C, H, W]), f, sort_keys=False, allow_unicode=True)

class Dataset(LightningDataModule): #Trocar para DataModule?
    def __init__(self, analyte: str, sweep_configs = None, **kwargs ): #samples, processed_samples, mapper: Dict, args, **kwags):
        super().__init__()
        self.analyte = analyte
        self.sweep_configs = sweep_configs

    def prepare_data(self):
        try:
            load_path =  os.path.join(os.path.dirname(__file__), "..", "..", "processed_dataset") #torch.load(self.saved_samples_path) # TODO carregar untyped storage data aqui
            with open(os.path.join(load_path, f"{self.analyte}_metadata.yaml"), "r") as f:
                self.num_classes, N, C, H, W = yaml.load(f, Loader=yaml.FullLoader)
            X = np.memmap(os.path.join(load_path, f"{self.analyte}_processed_samples.dat"), dtype=np.float32, mode='r', shape=(N, C, H, W))
            #y = np.memmap(os.path.join(load_path, f"{self.analyte}_labels.dat"), dtype=np.float32, mode='r', shape=(N, self.num_classes))
            y = np.memmap(os.path.join(load_path, f"{self.analyte}_labels.dat"), dtype=np.int64, mode='r', shape=(N))
        except:
            import shutil
            analyte = data_settings["analyte"]
            sample_dir = data_settings["samples_dir"]
            cache_dir = os.path.join("..", "cache_dir", analyte)
            feature_extractor = data_settings['feature_extractor']
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
                                          return_node=return_node,
                                          frozen_weights=frozen_weights,)
            preprocessing.prepare_samples_dataset()

            load_path =  os.path.join(os.path.dirname(__file__), "..", "..", "processed_dataset") #torch.load(self.saved_samples_path) # TODO carregar untyped storage data aqui
            with open(os.path.join(load_path, f"{self.analyte}_metadata.yaml"), "r") as f:
                self.num_classes, N, C, H, W = yaml.load(f, Loader=yaml.FullLoader)
            X = np.memmap(os.path.join(load_path, f"{self.analyte}_processed_samples.dat"), dtype=np.float32, mode='r', shape=(N, C, H, W))
            #y = np.memmap(os.path.join(load_path, f"{self.analyte}_labels.dat"), dtype=np.float32, mode='r', shape=(N, self.num_classes))
            y = np.memmap(os.path.join(load_path, f"{self.analyte}_labels.dat"), dtype=np.int64, mode='r', shape=(N))


        sample_extracted_features = torch.from_numpy(X)
        #true_class_value = torch.tensor(y)
        true_class_value = torch.tensor(y, dtype=torch.long)
        self.dataset = TensorDataset(sample_extracted_features, true_class_value)


    def setup(self, stage:str):
        len_dataset = len(self.dataset)
        # ~60% ~20% ~20%
        n_train = ceil(0.6*len_dataset)
        n_val = ceil(0.2*len_dataset)
        n_test = len_dataset - n_train - n_val

        train_set, val_set, test_set = random_split(self.dataset, [n_train, n_val, n_test], generator = Generator().manual_seed(42))

        if stage == "fit":
            self.dataset_train = train_set
            self.dataset_val = val_set

        elif stage == "validate":
            self.dataset_val = val_set

        elif stage == "test":
            self.dataset_test = test_set

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = self.sweep_configs["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size = self.sweep_configs["batch_size"], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False)

class BaseModel(LightningModule):
    def __init__(self, *, classifier_config: Dict[str, Any], loss_function: torch.nn.Module, learning_rate: float, learning_rate_patience: int = None, early_stopping_patience: int = 10, num_classes, frozen_weights: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.classifier_config = classifier_config
        self.classifier = None  # Lazy Initialization
        self.criterion = loss_function
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.early_stopping_patience = early_stopping_patience
        # TODO tornar dinamico a instanciacao dos parametros do modelo (input_dim = 512)
        self.classifier = self.classifier_config["model_class"](input_dim = 512, num_classes=num_classes)
        self.requires_flatten = self.classifier_config["requires_flatten"]
        self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
                                                    "acc": Accuracy(task="multiclass", num_classes=num_classes),
                                                    "precision": Precision(task="multiclass", num_classes=num_classes),
                                                    "recall": Recall(task="multiclass", num_classes=num_classes),
                                                    "F1-score": F1Score(task="multiclass", num_classes=num_classes)
                                                    }) for mode_name in ["Train", "Val", "Test"]})

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(), lr=1e-3) #SGD(self.parameters(), lr = self.learning_rate)
        self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.learning_rate_patience)


        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.reduce_lr_on_plateau, "monitor": "Loss/Val"}}

    # def configure_callbacks(self) -> List[Callback]:
    # # Apply early stopping.
    #  return [EarlyStopping(monitor="Loss/Val", mode="min", patience=self.early_stopping_patience)]

    def forward(self, x: Any):
        x = self.classifier(x)

        return x

    #defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, torch.tensor], stage: str):
        X, y = batch[0], batch[1]
        # if self.requires_flatten:
        #     X = torch.flatten(X, start_dim=1)
        predicted_value = self(X)    # o proprio objeto de BaseModel é o modelo (https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
        predicted_value = predicted_value.squeeze()
        # Compute and log the loss value.
        loss = self.criterion(predicted_value, y)
        self.log(f"Loss/{stage}", loss, prog_bar=True)
        # Compute and log step metrics.
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Step': value for metric_name, value in metrics(predicted_value, y).items()})

        return loss

    def training_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Val")

    def test_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Test")

    def _any_epoch_end(self, stage: str):
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Epoch': value for metric_name, value in metrics.compute().items()}, on_step=False, on_epoch=True) # logs metrics on epoch end
        metrics.reset()
        #Print loss at the end of each epoch
        #loss = self.trainer.callback_metrics[f"Loss/{stage}"]
        #print(f"Epoch {self.current_epoch} - Loss/{stage}: {loss.item()}")

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")