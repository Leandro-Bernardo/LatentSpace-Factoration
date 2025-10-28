import torch
import yaml

import numpy as np
from torch.optim import SGD
from engine.data_manager import LightningDataModule, LightningModule
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch import Generator, tensor, from_numpy
import chemical_analysis as ca
from typing import Any, Dict, List, Tuple



class Dataset(LightningDataModule):

    def init(self, samples, processed_samples, mapper: Dict, args, **kwags):
        self.samples = samples
        self.processed_samples = processed_samples
        self.mapper = mapper
        

    def prepare_data(self):
        with open("devices_mapper.yaml", "r") as f:
            data = yaml.safe_load(f)
        self.mapper = data["alkalinity"]
        num_classes = len(self.mapper)
        
        for processed_sample in self.processed_samples:
            # one hot
            id = self.mapper.get(self.mapper.get(processed_sample.sample.get("device")["model"]))
            one_hot = np.zeros(num_classes)
            one_hot[id] = 1
            self.models.append(one_hot)
            
            self.samples_pmf.append(processed_sample.calibrated_pmf)

        self.models = tensor(self.models)
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
    def __init__(self, *, dataset: DataLoader, model: torch.nn.Module, batch_size: int, loss_function: torch.nn.Module, learning_rate: float, learning_rate_patience: int = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.model = model()
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

    def configure_optimizers(self):
        self.optimizer = SGD(self.parameters(), lr = self.learning_rate)
        # self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.learning_rate_patience)

        return {f"optimizer: {self.optimizer}"}
        # return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.reduce_lr_on_plateau, "monitor": "Loss/Val"}}
        #return [self.optmizer], [self.reduce_lr_on_plateau]

    # def configure_callbacks(self) -> List[Callback]:
    # # Apply early stopping.
    #  return [EarlyStopping(monitor="Loss/Val", mode="min", patience=self.early_stopping_patience)]

    def forward(self, x: Any):
        return self.model(x)

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