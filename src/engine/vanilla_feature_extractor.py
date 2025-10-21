import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import models, transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from engine.classifier import sequeeze_classif

wandb_logger = WandbLogger(log_model="all")

class SNModel(LightningModule):
  def __init__(self, input_size, num_classes):
    super().__init__()
    self.backbone = models.squeezenet1_1(weights="DEFAULT")
    for param in self.backbone.parameters(): # TODO: Tentar freezar da forma como o Lightning implementa
      param.requires_grad = False
  
    self.last_layer = create_feature_extractor(self.backbone, ['features.12.cat'])

  def forward(self, x):
    x = self.last_layer(x)
    x = x.get('features.12.cat')
    x = sequeeze_classif(x)
    return x