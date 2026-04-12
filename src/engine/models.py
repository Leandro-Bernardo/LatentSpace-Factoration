import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List, Dict, Optional
import chemical_analysis as ca
import yaml, os
from collections import OrderedDict
from chemical_analysis import alkalinity, bisulfite2d, chloride, iron2, iron32d, ph, phosphate, redox, sulfate # TODO alterar versao de sulfato para sulfato 2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "..", "settings.yaml"), "r") as f:
    data_settings = yaml.load(f, Loader=yaml.FullLoader)

#input_size = 756

# pegar modelo pre treinado do projeto
class FeatureExtractor(nn.Module): # feature extractor backbone
    # class attrs
    squeezenets = {
                "alkalinity": alkalinity.AlkalinityNetworkSqueezeNetStyle,
                "bisulfite2d": bisulfite2d.Bisulfite2DNetworkSqueezeNetStyle,
                "chloride": chloride.ChlorideNetworkSqueezeNetStyle,
                "iron2": iron2.Iron2NetworkSqueezeNetStyle,
                "iron3": iron32d.Iron3NetworkSqueezeNetStyle,
                "ph": ph.PhNetworkSqueezeNetStyle,
                "phosphate": phosphate.PhosphateNetworkSqueezeNetStyle,
                "redox": redox.RedoxNetworkSqueezeNetStyle,
                "sulfate": sulfate.SulfateNetworkSqueezeNetStyle,
                }
    vgg11s = {
            "alkalinity": alkalinity.AlkalinityNetworkVgg11Style,
            "bisulfite2d": bisulfite2d.Bisulfite2DNetworkVgg11Style,
            "chloride": chloride.ChlorideNetworkVgg11Style,
            "iron2": iron2.Iron2NetworkVgg11Style,
            "iron3": iron32d.Iron3NetworkVgg11Style,
            "ph": ph.PhNetworkVgg11Style,
            "phosphate": phosphate.PhosphateNetworkVgg11Style,
            "redox": redox.RedoxNetworkVgg11Style,
            "sulfate": sulfate.SulfateNetworkVgg11Style,
            }
    checkpoints = {
                        "alkalinity": {"squeezenet" : alkalinity.NETWORK_CHECKPOINT, "vgg11" : alkalinity.UPNETWORK_CHECKPOINT},
                        "bisulfite2d": {"squeezenet" : bisulfite2d.NETWORK_CHECKPOINT, "vgg11" : bisulfite2d.UPNETWORK_CHECKPOINT},
                        "chloride": {"squeezenet" : chloride.NETWORK_CHECKPOINT, "vgg11" : chloride.UPNETWORK_CHECKPOINT},
                        "iron2": {"squeezenet" : iron2.NETWORK_CHECKPOINT, "vgg11" : iron2.UPNETWORK_CHECKPOINT},
                        "iron3": {"squeezenet" : iron32d.NETWORK_CHECKPOINT, "vgg11" : iron32d.UPNETWORK_CHECKPOINT},
                        "ph": {"squeezenet" : ph.NETWORK_CHECKPOINT, "vgg11" : ph.UPNETWORK_CHECKPOINT},
                        "phosphate": {"squeezenet" : phosphate.NETWORK_CHECKPOINT, "vgg11" : phosphate.UPNETWORK_CHECKPOINT},
                        "redox": {"squeezenet" : redox.NETWORK_CHECKPOINT, "vgg11" : redox.UPNETWORK_CHECKPOINT},
                        "sulfate": {"squeezenet" : sulfate.NETWORK_CHECKPOINT, "vgg11" : sulfate.UPNETWORK_CHECKPOINT},
                        }

    def __init__(self, analyte: str, backbone: Optional[str] = "squeezenet", return_node: Optional[str] = None, *args, **kwargs):
        super().__init__()
        self.analyte = analyte
        self.backbone = backbone
        self.return_node = return_node
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_from_checkpoint(self, *args, **kwargs):
        if self.backbone == "squeezenet":
            self.backbone = self.squeezenets[self.analyte]
            self.checkpoint = self.checkpoints[self.analyte]["squeezenet"]
            if self.return_node is None:
                return_node = {'model.backbone.features.12.cat': 'feature'} # key are node(s) and value(s) is the user alias {node: alias}
            else:
                return_node = {return_node : 'feature'}

        elif self.backbone == "vgg11":
            self.backbone = self.vgg11s[self.analyte]
            self.checkpoint = self.checkpoints[self.analyte]["vgg11"]
            if self.return_node is None:
                # TODO pegar o nome correto da rede do projeto
                return_node = {'features.20': 'feature'}     # key are node(s) and value(s) is the user alias {node: alias}
            else:
                return_node = {return_node : 'feature'}

        else:
            raise ValueError("Unsupported Backbone")

        # TODO verificar se está tudo correto (codigo veio do _model.py .load_from_checkpoint())
        state_dict = torch.load(self.checkpoint, map_location = self._device, weights_only=False)
        hyper_parameters = state_dict["hyper_parameters"]
        net = hyper_parameters["network_class"](**hyper_parameters)
        net.load_state_dict(OrderedDict([(key.lstrip("net."), value) for (key, value) in state_dict["state_dict"].items() if key.startswith("net.")]))
        net.eval()
        extractor = create_feature_extractor(net, return_node)

        return extractor

    def forward(self, x):

        out = self.extractor(x)
        x = out['feature']
        return x

class MLP1(torch.nn.Module):
    """_Classificador feito utilizando uma MLP_
    """
    def __init__(self, num_classes: int, device: str = "cuda",  **kwargs):
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

# TODO validar
class DynamicMLP(nn.Module):
    """_Classificador dinamico utilizando uma MLP_
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # apply avgPool (global pool if output shape is (1,1))

        def nearest_power_of_two(n):
            return 2 ** (n.bit_length() - 1)

        layers = []
        dims = []

        first_dim = nearest_power_of_two(input_dim)
        if first_dim == input_dim:
            first_dim = first_dim // 2
        dims = [input_dim, first_dim]

        while dims[-1] // 2 >= self.num_classes:
            dims.append(dims[-1] // 2)

        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(0.3))
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], self.num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)           # (N, C, 1, 1)
        x = torch.flatten(x, 1)   # (N, C)
        return self.model(x)

    # def __init__(self, num_classes):
    #     super().__init__()
    #     self.pool = nn.AdaptiveAvgPool2d((1,1))
    #     self.fc = nn.Linear(512, num_classes)

    # def forward(self, x):
    #     x = self.pool(x)
    #     x = torch.flatten(x, 1)
    #     return self.fc(x)