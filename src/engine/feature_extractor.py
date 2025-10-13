import torch
import torchvision

import lightning as pl
from torch import nn

from lightning import BaseModel
from torchvision import models




class squeezenet1_1(BaseModel):

    def __init__(self, dataset, batch_size: int = 32, learning_rate: float = 0.00001, **kwargs):
        super().__init__(
            dataset=dataset,
            model=models.squeezenet1_1(weights="DEFAULT"),
            batch_size=batch_size,
            loss_function=nn.CrossEntropyLoss(),
            learning_rate=learning_rate,
            **kwargs
        )


        # Extrai o feature extractor
        self.feature_extractor = self.model.features
        # Frozen Weights
        self.feature_extractor.eval()


    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            flattened = torch.flatten(features, 1)
        return flattened
