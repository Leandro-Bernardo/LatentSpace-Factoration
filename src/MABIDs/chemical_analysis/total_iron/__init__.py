from ._model import TotalIronEstimationFunction, TotalIronIntervalNetwork, TotalIronIntervalNetworkSqueezeNetStyle, TotalIronNetwork, TotalIronNetworkSqueezeNetStyle, TotalIronNetworkVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import TotalIronSampleDataset, ProcessedTotalIronSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "TotalIronNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "TotalIronPcaStats.npz"))
