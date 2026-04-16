from ._model import IronOxideEstimationFunction, IronOxideNetwork, IronOxideNetworkMobileNetV3Style, IronOxideNetworkSqueezeNetStyle, IronOxideNetworkShuffleNetV2X10Style, IronOxideNetworkShuffleNetV2X15Style, IronOxideNetworkShuffleNetV2X20Style, IronOxideNetworkVgg11Style, IronOxideUpNetwork, IronOxideNetworkUpVgg11Style, IronOxideIntervalNetwork, IronOxideIntervalNetworkSqueezeNetStyle, IronOxideBinaryNetwork, IronOxideBinaryNetworkSqueezeNetStyle
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import IronOxideSampleDataset, ProcessedIronOxideSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "IronOxideNetwork.ckpt"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "IronOxideUpNetwork.ckpt"))