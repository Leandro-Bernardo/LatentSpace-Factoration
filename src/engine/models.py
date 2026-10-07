import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Tuple, List, Dict, Any, Optional, Callable
from MABIDs import chemical_analysis as ca
import yaml, os
from collections import OrderedDict
from MABIDs.chemical_analysis import alkalinity, bisulfite2d, chloride, iron2, iron32d, ph, phosphate, redox, sulfate
import sys

sys.modules['chemical_analysis'] = ca

class _ModelRegistry:
    """Static factory for torchvision models and custom architectures."""

    _TORCHVISION_DEFAULT_FEATURE_NODES = {
        "vgg11": "features.20",
        "vgg11_bn": "features.28",
        "vgg16": "features.30",
        "vgg16_bn": "features.43",
        "vgg19": "features.36",
        "squeezenet1_0": "features.12",
        "squeezenet1_1": "features.12",
        "resnet18": "layer4",
        "resnet50": "layer4",
    }
    _MABIDS_CHECKPOINTS = {
            "alkalinity": {"squeezenet": alkalinity.NETWORK_CHECKPOINT, "vgg11": alkalinity.UPNETWORK_CHECKPOINT},
            "bisulfite": {"squeezenet": bisulfite2d.NETWORK_CHECKPOINT, "vgg11": bisulfite2d.UPNETWORK_CHECKPOINT},
            "chloride": {"squeezenet": chloride.NETWORK_CHECKPOINT, "vgg11": chloride.UPNETWORK_CHECKPOINT},
            "iron2": {"squeezenet": iron2.NETWORK_CHECKPOINT, "vgg11": iron2.UPNETWORK_CHECKPOINT},
            "iron3": {"squeezenet": iron32d.NETWORK_CHECKPOINT, "vgg11": iron32d.UPNETWORK_CHECKPOINT},
            "ph": {"squeezenet": ph.NETWORK_CHECKPOINT, "vgg11": ph.UPNETWORK_CHECKPOINT},
            "phosphate": {"squeezenet": phosphate.NETWORK_CHECKPOINT, "vgg11": phosphate.UPNETWORK_CHECKPOINT},
            "redox": {"squeezenet": redox.NETWORK_CHECKPOINT, "vgg11": redox.UPNETWORK_CHECKPOINT},
            "sulfate": {"squeezenet": sulfate.NETWORK_CHECKPOINT, "vgg11": sulfate.UPNETWORK_CHECKPOINT},
        }
    _MABIDS_CHECKPOINT_DEFAULT_FEATURE_NODES = {
        "squeezenet": "model.backbone.features.12.cat",
        "vgg11": "features.20",
    }

    @classmethod
    def build_from_torchvision(cls, backbone_name: str, pretrained: bool = False, return_node: Optional[str] = None, **kwargs: Any) -> Tuple[nn.Module, Dict[str, str]]:
        """Loads a model from torchvision."""
        backbone_name = backbone_name.lower()
        weights = "DEFAULT" if pretrained else None

        # Loads the model (Pre-trained or not)
        model = get_model(backbone_name, weights=weights, **kwargs)

        # Extract and returns the feature extractor module (CNNs)
        if return_node is None:
            return_node = cls._get_default_feature_node(model, backbone_name)

        return model, {return_node: 'feature'}

    @classmethod
    def build_from_mabid_checkpoint(cls, analyte: str, backbone_name: str, return_node: Optional[str] = None) -> Tuple[nn.Module, Dict[str, str]]:
        """"Loads a pretrained MABID model."""
        analyte = analyte.lower()
        backbone_name = backbone_name.lower()

        checkpoint_path = cls._MABIDS_CHECKPOINTS[analyte][backbone_name]
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hyper_parameters = state_dict["hyper_parameters"]
        net_cls = hyper_parameters["network_class"]
        init_kwargs = {k: v for k, v in hyper_parameters.items() if k != "network_class"}
        model = net_cls(**init_kwargs)
        cleaned_weights = OrderedDict([
            (k.removeprefix("net."), v)
            for k, v in state_dict["state_dict"].items()
            if k.startswith("net.")
        ])
        model.load_state_dict(cleaned_weights)

        if return_node is None:
            return_node = cls._MABIDS_CHECKPOINT_DEFAULT_FEATURE_NODES.get(backbone_name)

        return model, {return_node: 'feature'}

    @classmethod
    def _get_default_feature_node(cls, model: nn.Module, backbone_name: str) -> str:
        # Case when the backbone is mapped.
        if backbone_name in cls._TORCHVISION_DEFAULT_FEATURE_NODES:
            return cls._TORCHVISION_DEFAULT_FEATURE_NODES[backbone_name]

        # Dynamic solution for unlisted VGGs / SqueezeNets.
        if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
            last_idx = len(model.features) - 1
            return f"features.{last_idx}"

        #  Dynamic solution for unlisted ResNet / ConvNeXt.
        if hasattr(model, "layer4"):
            return "layer4"

        raise ValueError(f"Could not infer the default feature node for {backbone_name}. \nAdd it to DEFAULT_FEATURE_NODES or pass return_node explicitly.")

class FeatureExtractor(nn.Module):

    def __init__(self, analyte: str, use_torchvision_model: bool, torchvision_model_pretrained: bool, feature_extractor_backbone: str, freeze_cnn_weights: bool,  return_node: Optional[str] = None, device: Optional[str] = "cpu", *args, **kwargs):
        super().__init__()
        self.analyte = analyte
        self.use_torchvision_model = use_torchvision_model
        self.torchvision_model_pretrained = torchvision_model_pretrained
        self.feature_extractor_backbone = feature_extractor_backbone
        self.freeze_cnn_weights = freeze_cnn_weights
        self.return_node = return_node
        self._device = device

        # Internal Extractor
        if self.use_torchvision_model:
            model_name = "squeezenet1_1" if self.feature_extractor_backbone == "squeezenet" else feature_extractor_backbone
            base_net, return_node_dict = _ModelRegistry.build_from_torchvision(backbone_name=model_name, pretrained=self.torchvision_model_pretrained, return_node=self.return_node)
        else:
            base_net, return_node_dict = _ModelRegistry.build_from_mabid_checkpoint(analyte=self.analyte, backbone_name=self.feature_extractor_backbone, return_node=self.return_node)

        self.extractor = create_feature_extractor(base_net, return_node_dict)

        # Detach (or not) from the FX graph (Froze or not the weights).
        for param in self.extractor.parameters():
            if param.is_floating_point():
                param.requires_grad = not self.freeze_cnn_weights

        # Moves the model to device (CPU or GPU).
        self.extractor.to(self._device)

    def train(self, mode: bool = True):
        """Prevents the extractor from reactivating BatchNorm and Dropout if the weights are frozen."""
        super().train(mode)
        if self.freeze_cnn_weights:
            self.extractor.eval()
        return self

    def forward(self, x: torch.Tensor):
        out = self.extractor(x)
        return out['feature']

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
        #x = torch.nn.functional.softmax(x, dim=1)

        return x

class SqueezeNetClassifier(torch.nn.Module):
    """_Classificador original da arquitetura SqueezeNet_
        Utilizando o classificador com a mesma arquitetura da Squeeze Net original, uma camada de convolução + avg pooling
    """
    def __init__(self, num_classes: int, in_channels: int = 512, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.layer_classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(in_channels, self.num_classes, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )

    def forward(self, x: torch.Tensor):
        x = self.layer_classifier(x)
        #x = torch.nn.functional.softmax(x, dim=1)
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
        x = self.pool(x)          # (N, C, 1, 1)
        x = torch.flatten(x, 1)   # (N, C)
        return self.model(x)


