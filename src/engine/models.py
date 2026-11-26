import torch
import torch.nn as nn
from torch.func import functional_call
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

class _FeatureExtractorModelRegistry:
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
    def build_from_torchvision(cls, backbone_name: str, pretrained: bool = False, in_channels: int = 1, return_node: Optional[str] = None, **kwargs: Any) -> Tuple[nn.Module, Dict[str, str]]:
        """Loads a model from torchvision."""
        backbone_name = backbone_name.lower()
        weights = "DEFAULT" if pretrained else None

        # Loads the model (Pre-trained or not)
        model = get_model(backbone_name, weights=weights, **kwargs)

        # Adapts the input channels before mapping the FX graph
        if in_channels != 3:
            model = cls.adapt_first_conv(model, in_channels=in_channels, pretrained=pretrained)

        # Extract and returns the feature extractor module (CNNs)
        if return_node is None:
            return_node = cls._get_default_feature_node(model, backbone_name)

        return model, {return_node: 'feature'}

    @classmethod
    def build_from_mabid_checkpoint(cls, analyte: str, backbone_name: str, return_node: Optional[str] = None) -> Tuple[nn.Module, Dict[str, str]]:
        """"Loads a pretrained MABID model."""
        analyte = analyte.lower()
        backbone_name = backbone_name.lower()

        try:
            checkpoint_path = cls._MABIDS_CHECKPOINTS[analyte][backbone_name]
        except:
            raise Exception(f"Unsuported MABID backbone. \nAvailable: squeezenet or vgg11. \nChosen: {backbone_name}")
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

    @classmethod
    def adapt_first_conv(cls, model: nn.Module, in_channels: int = 1, pretrained: bool = False) -> nn.Module:
        """Replaces the first convolution of torchvision models to accept custom in_channels."""
        # 1. Locate the first convolutional layer according to the architecture.
        parent_module, attr_name, old_conv = cls._get_first_conv(model)

        if old_conv.in_channels == in_channels:
            return model

        # 2. Create the new convolution keeping kernel_size, stride, padding, etc.
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=1 if old_conv.groups == 1 else in_channels,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )

        # 3. Adapt pre-trained weights (averaged across the RGB channel axis).
        if pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    # Original shape: [out_channels, 3, K_h, K_w] -> New shape: [out_channels, 1, K_h, K_w]
                    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    # Gerenic case.
                    new_conv.weight.data = old_conv.weight.data.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels, :, :]

                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data.clone()

        # 4. Replaces the layer.
        if isinstance(parent_module, (nn.Sequential, nn.ModuleList)):
            parent_module[int(attr_name)] = new_conv
        else:
            setattr(parent_module, attr_name, new_conv)

        return model

    @classmethod
    def _get_first_conv(cls, model: nn.Module) -> Tuple[nn.Module, str, nn.Conv2d]:
        """Identify the parent module, the attribute/index name, and the initial Conv2d layer."""
        #  ResNet family.
        if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
            return model, "conv1", model.conv1

        #  VGG, SqueezeNet and AlexNet family (features[0] is Conv2d).
        if hasattr(model, "features") and len(model.features) > 0:
            if isinstance(model.features[0], nn.Conv2d):
                return model.features, "0", model.features[0]

            #  ConvNeXt, EfficientNet, MobileNetV2/V3 (features[0] is a block, ex: Conv2dNormActivation).
            if hasattr(model.features[0], "0") and isinstance(model.features[0][0], nn.Conv2d):
                return model.features[0], "0", model.features[0][0]

        # DenseNet family.
        if hasattr(model, "features") and hasattr(model.features, "conv0"):
            return model.features, "conv0", model.features.conv0

        raise ValueError(f"Could not automatically identify the first convolution for {type(model).__name__}.")

class FeatureExtractor(nn.Module):

    def __init__(self, analyte: str, use_torchvision_model: bool, torchvision_model_pretrained: bool, feature_extractor_backbone: str, freeze_cnn_weights: bool,  in_channels: int = 1, processing_device: str = "cpu", return_node: Optional[str] = None, *args, **kwargs):
        super().__init__()
        self.analyte = analyte
        self.use_torchvision_model = use_torchvision_model
        self.torchvision_model_pretrained = torchvision_model_pretrained
        self.feature_extractor_backbone = feature_extractor_backbone
        self.freeze_cnn_weights = freeze_cnn_weights
        self.in_channels = in_channels
        self.processing_device = processing_device
        self.return_node = return_node

        # Internal Extractor
        if self.use_torchvision_model:
            model_name = "squeezenet1_1" if self.feature_extractor_backbone == "squeezenet" else feature_extractor_backbone
            base_net, return_node_dict = _FeatureExtractorModelRegistry.build_from_torchvision(backbone_name=model_name, pretrained=self.torchvision_model_pretrained, in_channels=self.in_channels, return_node=self.return_node)
        else:
            base_net, return_node_dict = _FeatureExtractorModelRegistry.build_from_mabid_checkpoint(analyte=self.analyte, backbone_name=self.feature_extractor_backbone, return_node=self.return_node)

        self.extractor = create_feature_extractor(base_net, return_node_dict)

        # Detach (or not) from the FX graph (Froze or not the weights).
        for param in self.extractor.parameters():
            if param.is_floating_point():
                param.requires_grad = not self.freeze_cnn_weights

        # Moves the model to processing device (CPU or GPU).
        self.extractor.to(self.processing_device)

    def train(self, mode: bool = True):
        """Prevents the extractor from reactivating BatchNorm and Dropout if the weights are frozen."""
        super().train(mode)
        if self.freeze_cnn_weights:
            self.extractor.eval()
        return self

    def forward(self, x: torch.Tensor):
        out = self.extractor(x)
        return out['feature']

class DynamicClassifier(nn.Module):
    """Classificador dinamico utilizando uma MLP reduzindo por fator de 2.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # apply avgPool (global pool if output shape is (1,1))

        assert input_dim > 1, f"Input dim is too low ({input_dim}). Verify the data."

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

class DynamicClassifierv2(nn.Module):
    """Classificador dinâmico utilizando uma MLP reduzindo por fator de 4."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        assert (
            input_dim > 1
        ), f"Input dim is too low ({input_dim}). Verify the data."

        def nearest_power_of_four(n):
            shift = (n.bit_length() - 1) // 2
            return 4**shift

        layers = []

        first_dim = nearest_power_of_four(input_dim)
        if first_dim == input_dim:
            first_dim = first_dim // 4

        if first_dim < self.num_classes:
            dims = [input_dim]
        else:
            dims = [input_dim, first_dim]
            while dims[-1] // 4 >= self.num_classes:
                dims.append(dims[-1] // 4)

        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(0.3))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], self.num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)  # (N, C, 1, 1)
        x = torch.flatten(x, 1)  # (N, C)
        return self.model(x)

class DynamicClassifierv3(nn.Module):
    """Classificador dinâmico utilizando uma MLP reduzindo por fator de 8."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        assert (
            input_dim > 1
        ), f"Input dim is too low ({input_dim}). Verify the data."

        def nearest_power_of_eight(n):
            shift = (n.bit_length() - 1) // 3
            return 8**shift

        layers = []

        first_dim = nearest_power_of_eight(input_dim)
        if first_dim == input_dim:
            first_dim = first_dim // 8

        if first_dim < self.num_classes:
            dims = [input_dim]
        else:
            dims = [input_dim, first_dim]
            while dims[-1] // 8 >= self.num_classes:
                dims.append(dims[-1] // 8)

        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(0.3))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], self.num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)  # (N, C, 1, 1)
        x = torch.flatten(x, 1)  # (N, C)
        return self.model(x)

class SqueezeNetClassifier(nn.Module):
    """Classificador original da arquitetura SqueezeNet (Conv 1x1 + Global Avg Pool).
    """
    def __init__(self, num_classes: int, input_dim: int = 512, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.layer_classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(input_dim, self.num_classes, kernel_size=1),
            #torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )

    def forward(self, x: torch.Tensor):
        x = self.layer_classifier(x)
        return x

class VGGClassifier(nn.Module):
    """Classificador original da arquitetura da VGG 11.
    """

    def __init__(self, num_classes=1000, input_dim=512 * 7 * 7, dropout_rate=0.5, *args, **kwargs):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # First fully conected layer
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # Second fully conected layer
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # Third fully conected layer (Logits` Output`)
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.classifier(x)
        return x

class DynamicRegressor(nn.Module):
    """Regressor dinâmico utilizando uma MLP reduzindo por fator de 2."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        min_hidden_dim: int = 8,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        assert input_dim > 1, f"Input dim is too low ({input_dim}). Verify the data."

        def nearest_power_of_two(n: int) -> int:
            return 2 ** (n.bit_length() - 1)

        stop_dim = max(self.output_dim, min_hidden_dim)
        first_dim = nearest_power_of_two(input_dim)
        if first_dim == input_dim:
            first_dim = first_dim // 2

        if first_dim < stop_dim:
            dims = [input_dim]
        else:
            dims = [input_dim, first_dim]
            while dims[-1] // 2 >= stop_dim:
                dims.append(dims[-1] // 2)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)         # (N, C, 1, 1)
        x = torch.flatten(x, 1)  # (N, C)
        return self.model(x)     # (N, output_dim)

class DynamicRegressorv2(nn.Module):
    """Regressor dinâmico utilizando uma MLP reduzindo por fator de 4."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        min_hidden_dim: int = 8,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        assert input_dim > 1, f"Input dim is too low ({input_dim}). Verify the data."

        def nearest_power_of_four(n: int) -> int:
            shift = (n.bit_length() - 1) // 2
            return 4**shift

        stop_dim = max(self.output_dim, min_hidden_dim)
        first_dim = nearest_power_of_four(input_dim)
        if first_dim == input_dim:
            first_dim = first_dim // 4

        if first_dim < stop_dim:
            dims = [input_dim]
        else:
            dims = [input_dim, first_dim]
            while dims[-1] // 4 >= stop_dim:
                dims.append(dims[-1] // 4)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)         # (N, C, 1, 1)
        x = torch.flatten(x, 1)  # (N, C)
        return self.model(x)     # (N, output_dim)

class DynamicRegressorv3(nn.Module):
    """Regressor dinâmico utilizando uma MLP reduzindo por fator de 8."""

    def __init__(self, input_dim: int, output_dim: int = 1, min_hidden_dim: int = 8, dropout_rate: float = 0.3, ):
        super().__init__()
        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        assert input_dim > 1, f"Input dim is too low ({input_dim}). Verify the data."

        def nearest_power_of_eight(n: int) -> int:
            shift = (n.bit_length() - 1) // 3
            return 8**shift

        stop_dim = max(self.output_dim, min_hidden_dim)
        first_dim = nearest_power_of_eight(input_dim)
        if first_dim == input_dim:
            first_dim = first_dim // 8

        if first_dim < stop_dim:
            dims = [input_dim]
        else:
            dims = [input_dim, first_dim]
            while dims[-1] // 8 >= stop_dim:
                dims.append(dims[-1] // 8)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)         # (N, C, 1, 1)
        x = torch.flatten(x, 1)  # (N, C)
        return self.model(x)     # (N, output_dim)

class SqueezeNetRegressor(nn.Module):
    """Regressor original da arquitetura SqueezeNet (Conv 1x1 + Global Avg Pool).
    """

    def __init__(self, output_dim: int = 1, input_dim: int = 512, dropout_rate: float = 0.5, **kwargs, ):
        super().__init__()
        self.output_dim = output_dim
        self.layer_regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(input_dim, self.output_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_regressor(x)  # (N, output_dim)

class VGGRegressor(nn.Module):
    """Regressor original da arquitetura da VGG 11.
    """

    def __init__(self, output_dim: int = 1, input_dim: int = 512 * 7 * 7, dropout_rate: float = 0.5, *args, **kwargs, ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            # First fully connected layer
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            # Second fully connected layer
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            # Third fully connected layer (Regression Output)
            nn.Linear(4096, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.regressor(x)  # (N, output_dim)

class LSFModel(nn.Module):
    """Gerencia o backbone convolucional e os cabeçotes de classificação e regressão,
    particionando o espaço latente em z_clf e z_reg e congelando os pesos dos
    cabeçotes nas avaliações cruzadas (F_bar e G_bar).
    """

    def __init__(self, feature_extractor: nn.Module, classifier_model: nn.Module, regressor_model: nn.Module, ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier_model = classifier_model
        self.regressor_model = regressor_model

    @staticmethod
    def _forward_frozen_weights(module: nn.Module, z: torch.Tensor) -> torch.Tensor:
        """Executa o head (classificador ou regressor) com pesos congelados e buffers protegidos,
        mantendo o gradiente ativo para atualizar o backbone convolucional.
        """
        frozen_params = {k: v.detach() for k, v in module.named_parameters()}
        frozen_buffers = {k: v.detach().clone() for k, v in module.named_buffers()}
        return functional_call(module, (frozen_params, frozen_buffers), (z,))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # 1. Extrai o espaço latente compartilhado a partir da CNN
        z = self.feature_extractor(x)

        if z.size(1) % 2 != 0:
            raise ValueError(
                f"A dimensão latente (dim=1) deve ser par para o particionamento, recebido: {z.size(1)}"
            )

        # 2. Particiona em dois subespaços disjuntos (funciona para (B, D) ou (B, C, H, W))
        z_reg, z_clf = torch.chunk(z, chunks=2, dim=1)

        # 3. Passes principais (cabeçote DESCONGELADO + backbone treina o subespaço correto)
        logits_reg_for_F = self.regressor_model(z_reg)
        logits_clf_for_G = self.classifier_model(z_clf)

        # 4. Passes cruzados (cabeçote CONGELADO + backbone expulsa informação do subespaço oposto)
        logits_reg_for_F_bar = self._forward_frozen_weights(self.regressor_model, z_clf)
        logits_clf_for_G_bar = self._forward_frozen_weights(self.classifier_model, z_reg)

        return {
            "logits_reg_for_F": logits_reg_for_F,
            "logits_reg_for_F_bar": logits_reg_for_F_bar,
            "logits_clf_for_G": logits_clf_for_G,
            "logits_clf_for_G_bar": logits_clf_for_G_bar,
        }