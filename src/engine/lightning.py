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
from MABIDs import chemical_analysis as ca
import numpy as np
import matplotlib
matplotlib.use("Agg")  # renders plots only in memory
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import wandb
#import multiprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm



with open(os.path.join(os.path.dirname(__file__), "..", "settings.yaml"), "r") as f:
    data_settings = yaml.load(f, Loader=yaml.FullLoader)
try:
    with open(os.path.join(os.path.dirname(__file__), "..", "devices.yaml"), "r") as f:
        devices = yaml.load(f, Loader=yaml.FullLoader)
except:
    devices = {} # Lazy Initialization


class Preprocessing():
    def __init__(self, analyte: str, sample_dir: str, cache_dir: str, devices: Dict[str, Dict[str, int]], backbone: str, return_node: Optional[str] = None, frozen_weights: Optional[bool] = True, save_pmfs_as_img: Optional[bool] = False ):
        self.analyte = analyte
        self.sample_dir = sample_dir
        self.cache_dir = cache_dir
        self.devices = devices
        self.save_path = os.path.join(os.path.dirname(__file__), "..")
        self.feature_extractor = FeatureExtractor(analyte=self.analyte, backbone=backbone, return_node=return_node)
        self.save_pmfs_as_img = save_pmfs_as_img

    def prepare_samples_dataset(self):
        processed_samples, reduction_level = self.process_samples()

        self.feature_extraction(processed_samples, reduction_level, self.save_pmfs_as_img)


    def process_samples(self):
        preprocessing = {
                    "alkalinity":{"dataset": ca.alkalinity.AlkalinitySampleDataset, "processed_dataset": ca.alkalinity.ProcessedAlkalinitySampleDataset, "reduction_level": 0.05},
                    "chloride": {"dataset": ca.chloride.ChlorideSampleDataset, "processed_dataset": ca.chloride.ProcessedChlorideSampleDataset, "reduction_level": 0.10},
                    "sulfate": {"dataset": ca.sulfate.SulfateSampleDataset, "processed_dataset": ca.sulfate.ProcessedSulfateSampleDataset, "reduction_level": 0.10},
                    "phosphate": {"dataset": ca.phosphate.PhosphateSampleDataset, "processed_dataset": ca.phosphate.ProcessedPhosphateSampleDataset, "reduction_level": 0.10},
                    "bisulfite": {"dataset": ca.bisulfite2d.Bisulfite2DSampleDataset, "processed_dataset": ca.bisulfite2d.ProcessedBisulfite2DSampleDataset, "reduction_level": 0.10},
                    "iron2": {"dataset": ca.iron2.Iron2SampleDataset, "processed_dataset": ca.iron2.ProcessedIron2SampleDataset, "reduction_level": 0.10},
                    "iron3": {"dataset": ca.iron3.Iron3SampleDataset, "processed_dataset": ca.iron3.ProcessedIron3SampleDataset, "reduction_level": 0.10},
                    "iron_oxid": {"dataset": ca.iron_oxid.IronOxidSampleDataset, "processed_dataset": ca.iron_oxid.ProcessedIronOxidSampleDataset, "reduction_level": 0.10},
                    "ph": {"dataset": ca.ph.PhSampleDataset, "processed_dataset": ca.ph.ProcessedPhSampleDataset, "reduction_level": 0.10},
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
        reduction_level = preprocessing[self.analyte]["reduction_level"]

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

        return processed_samples, reduction_level

    def feature_extraction(self, processed_samples, reduction_level, save_pmfs_as_img):
        if save_pmfs_as_img:
            pmfs_as_img = {"original_pmf": [],
                                "roi_pmf": [],
                        "resized_roi_pmf": []}
        processed_sample = processed_samples
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

        # calculates the ROI based on the reduction level of each analyte
        print("computing the calibrated PMF ROI")
        input_roi, input_range = processed_samples.compute_calibrated_pmf_roi(reduction_level)
        in_x, out_x, in_y, out_y = input_roi[0][0], input_roi[0][1], input_roi[1][0], input_roi[1][1]
        # extract features with selected backbone
        features = []
        labels = []
        shape = None
        num_classes = len(current_samples_devices)
        pretrained_model = self.feature_extractor.load_from_checkpoint()  # loads pretrained model
        pretrained_model.eval()
        # TODO otimizar para processar com GPU e batches
        for processed_sample in tqdm(processed_samples, desc = 'extracting features'):
            # process the input X (pmf)
            original_pmf = processed_sample.calibrated_pmf
            roi_pmf = original_pmf[in_x:out_x, in_y:out_y]
            pmf_tensor = torch.tensor(roi_pmf)
            #TODO adaptar o resize do chemical analysis
            pmf_tensor_resized = torch.nn.functional.interpolate(pmf_tensor.unsqueeze(0).unsqueeze(0), size=(511, 511), mode='bilinear', align_corners=False)
            current_pmf_extracted_features = pretrained_model(pmf_tensor_resized.squeeze(0)).get('feature')
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
            if save_pmfs_as_img:
                pmfs_as_img["original_pmf"].append(original_pmf)
                pmfs_as_img["roi_pmf"].append(roi_pmf)
                pmfs_as_img["resized_roi_pmf"].append(pmf_tensor_resized.squeeze())

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
            data = {
                    "num_classes": num_classes,
                    "num_samples": N,
                    "num_channels": C,
                    "height": H,
                    "width": W
                }

            yaml.dump(data, f, sort_keys=False, allow_unicode=True)

        # saves pmfs as image for debuging
        if save_pmfs_as_img:
            assert (len(pmfs_as_img["original_pmf"]) == len(pmfs_as_img["roi_pmf"])) & (len(pmfs_as_img["original_pmf"]) == len(pmfs_as_img["resized_roi_pmf"])) & (len(pmfs_as_img["roi_pmf"]) == len(pmfs_as_img["resized_roi_pmf"]))

            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug", f"{self.analyte}_pmfs")
            os.makedirs(output_dir, exist_ok=True)
            for i in range(len(pmfs_as_img["original_pmf"])):
                original = pmfs_as_img["original_pmf"][i]
                roi = pmfs_as_img["roi_pmf"][i]
                resized = pmfs_as_img["resized_roi_pmf"][i]

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(original, cmap="viridis")
                axes[0].set_title(f"Original ({original.shape[0]} x {original.shape[1]})")
                axes[0].axis("off")

                axes[1].imshow(roi, cmap="viridis")
                axes[1].set_title(f"ROI ({roi.shape[0]} x {roi.shape[1]})")
                axes[1].axis("off")

                axes[2].imshow(resized, cmap="viridis")
                axes[2].set_title(f"Resized ROI ({resized.shape[0]} x {resized.shape[1]})")
                axes[2].axis("off")

                file_path = os.path.join(output_dir, f"pmf_{i}.png")
                plt.savefig(file_path, bbox_inches="tight")

                plt.close(fig)

class Dataset(LightningDataModule):
    def __init__(self, analyte: str, sweep_configs = None, **kwargs ):
        super().__init__()
        self.analyte = analyte
        self.sweep_configs = sweep_configs

    def prepare_data(self):
        try:
            load_path =  os.path.join(os.path.dirname(__file__), "..", "..", "processed_dataset") #torch.load(self.saved_samples_path) # TODO carregar untyped storage data aqui
            with open(os.path.join(load_path, f"{self.analyte}_metadata.yaml"), "r") as f:
                metadata = yaml.load(f, Loader=yaml.FullLoader)
            self.num_classes, N, C, H, W = metadata["num_classes"], metadata["num_samples"], metadata["num_channels"], metadata["height"], metadata["width"]
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
            save_pmf_as_img = data_settings['save_pmf_as_img']
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
                                          frozen_weights=frozen_weights,
                                          save_pmfs_as_img=save_pmf_as_img)
            preprocessing.prepare_samples_dataset()

            load_path =  os.path.join(os.path.dirname(__file__), "..", "..", "processed_dataset") #torch.load(self.saved_samples_path) # TODO carregar untyped storage data aqui
            with open(os.path.join(load_path, f"{self.analyte}_metadata.yaml"), "r") as f:
                metadata = yaml.load(f, Loader=yaml.FullLoader)
            self.num_classes, N, C, H, W = metadata["num_classes"], metadata["num_samples"], metadata["num_channels"], metadata["height"], metadata["width"]
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

        self.dataset_train = train_set
        self.dataset_val = val_set
        self.dataset_test = test_set

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size = self.sweep_configs["batch_size"])#, shuffle=True, num_workers= 2, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size = self.sweep_configs["batch_size"])#, shuffle=False, num_workers= 2, pin_memory=True, drop_last=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1,  shuffle=False)#, num_workers= 2, pin_memory=True, drop_last=False, persistent_workers=True)

class BaseModel(LightningModule):
    def __init__(self, *, classifier_config: Dict[str, Any], input_dim: int, loss_function: torch.nn.Module, learning_rate: float, learning_rate_patience: int = None, early_stopping_patience: int = 10, num_classes, frozen_weights: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.classifier_config = classifier_config
        self.classifier = None  # Lazy Initialization
        self.criterion = loss_function
        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.early_stopping_patience = early_stopping_patience
        self.classifier = self.classifier_config["model_class"](input_dim = input_dim, num_classes=num_classes)
        self.requires_flatten = self.classifier_config["requires_flatten"]
        self.metrics = ModuleDict({mode_name: MetricCollection({  # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs
                                                    "acc": Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
                                                    "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                                                    "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
                                                    "F1-score": F1Score(task="multiclass", num_classes=num_classes, average="macro")
                                                    }) for mode_name in ["Train", "Val", "Test"]})
        self._inference_time = {"predictions": [], "targets": []}

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

    # Defines basics operations for train, validadion and test
    def _any_step(self, batch: Tuple[torch.tensor, torch.tensor], stage: str):
        X, y = batch[0], batch[1]
        logits  = self(X)    # BaseModel obj is the network itself (https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
        # Compute and log the loss value.
        loss = self.criterion(logits , y)
        self.log(f"Loss/{stage}", loss, prog_bar=True)
        # Compute and log step metrics.
        predicted_value = torch.argmax(logits, dim=1)
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Step': value for metric_name, value in metrics(logits, y).items()})
        return loss

    def training_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Train")

    def validation_step(self, batch: List[torch.tensor]):
        return self._any_step(batch, "Val")

    def test_step(self, batch: List[torch.tensor]):
        self.eval()
        X, y = batch[0], batch[1]
        logits = self(X)
        preds = torch.argmax(logits, dim=1)

        metrics: MetricCollection = self.metrics["Test"]
        metrics(preds, y)

        self._inference_time["predictions"].append(preds.detach().cpu().item())
        self._inference_time["targets"].append(y.detach().cpu().item())

        metrics: MetricCollection = self.metrics["Test"]
        metrics(logits, y)

        with torch.no_grad():
            self._inference_time["predictions"].append(preds.detach().cpu().item())
            self._inference_time["targets"].append(y.detach().cpu().item())

    def _any_epoch_end(self, stage: str):
        # calculates metrics
        metrics: MetricCollection = self.metrics[stage]  # type: ignore
        self.log_dict({f'{metric_name}/{stage}/Epoch': value for metric_name, value in metrics.compute().items()}, on_step=False, on_epoch=True) # logs metrics on epoch end
        metrics.reset()

    def on_train_epoch_end(self):
        self._any_epoch_end("Train")

    def on_validation_epoch_end(self):
        self._any_epoch_end("Val")

    def on_test_epoch_end(self):
        self._any_epoch_end("Test")

    def on_train_end(self):
        self.eval()
        self.trainer.test(model=self, datamodule=self.trainer.datamodule, ckpt_path="best")

        preds   = np.array(self._inference_time["predictions"])
        targets = np.array(self._inference_time["targets"])

        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title("Confusion Matrix — Best Model (Test Set)")
        plt.tight_layout()

        self.logger.experiment.log({"confusion_matrix/Test/BestModel": wandb.Image(fig)})
        plt.savefig("confusion_matrix_best_model.png", dpi=150)
        plt.close(fig)

        self._inference_time = {"predictions": [], "targets": []} # resets