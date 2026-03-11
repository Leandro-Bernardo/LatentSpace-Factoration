import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List, Dict, Optional
import yaml, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "..", "settings.yaml"), "r") as f:
    data_settings = yaml.load(f, Loader=yaml.FullLoader)

#input_size = 756

# pegar modelo pre treinado do projeto
class FeatureExtractor(nn.Module): # feature extractor backbone

    def __init__(self, backbone: str, return_node: Optional[str] = None, freeze: bool = True, requires_flatten: bool = False ):
        super().__init__()

        self.requires_flatten = requires_flatten

        if backbone == "squeezenet":
            backbone = models.squeezenet1_1(weights="DEFAULT")
            if return_node is None:
                return_node = {'features.12.cat': 'feature'} # key are node(s) and value(s) is the user alias {node: alias}
            else:
                return_node = {return_node : 'feature'}

        elif backbone == "vgg11":
            backbone = models.vgg11(weights="DEFAULT")
            if return_node is None:
                return_node = {'features.20': 'feature'}     # key are node(s) and value(s) is the user alias {node: alias}
            else:
                return_node = {return_node : 'feature'}

        else:
            raise ValueError("Unsupported Backbone")

        if freeze:
            for p in backbone.parameters():
                p.requires_grad = False

        self.extractor = create_feature_extractor(backbone, return_node)


    def forward(self, x):

        out = self.extractor(x)
        x = out['feature']
        if self.requires_flatten:
            x = torch.flatten(x, 1)
        return x

class MLP1(torch.nn.Module):
    """_Classificador feito utilizando uma MLP_
    """
    def __init__(self, num_class: int, device: str = "cuda",  **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=784, out_features=512, bias=True),
                                    torch.nn.ReLU(),)
        self.l5 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=512, out_features=256, bias=True),
                                    torch.nn.ReLU(),)
        self.l4 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=256, out_features=128, bias=True),
                                    torch.nn.ReLU(),)
        self.l3 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=128, out_features=64, bias=True),
                                    torch.nn.ReLU(),)
        self.l2 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=64, out_features=32, bias=True),
                                    torch.nn.ReLU(),)
        self.l1 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=32, out_features=16, bias=True),
                                    torch.nn.ReLU(),)
        self.output_layer = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=16, out_features=10, bias=True),
                                    torch.nn.ReLU(),)

    def forward(self, x: torch.Tensor):

        x = self.input_layer(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x

class SqueezeNetClassifier(torch.nn.Module):
    """_Classificador original da arquitetura SqueezeNet_
        Utilizando o classificador com a mesma arquitetura da Squeeze Net original, uma camada de convolução + avg pooling
    """
    def __init__(self, num_class: int, device: str = "cuda",  **kwargs):
        super().__init__()
        self.num_class = num_class
        # TODO: [David] Alterar o tamanho da entrada (baseado no pmf.shape)
        final_conv = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.layer_classfier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5), final_conv, torch.nn.ReLU(inplace=True), torch.nn.AvgPool2d(13)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer_classfier(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x