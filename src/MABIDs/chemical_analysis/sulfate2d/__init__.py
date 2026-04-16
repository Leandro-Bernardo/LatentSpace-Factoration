from ._model import Sulfate2DEstimationFunction, Sulfate2DIntervalNetwork, Sulfate2DIntervalNetworkSqueezeNetStyle, Sulfate2DNetwork, Sulfate2DNetworkSqueezeNetStyle, Sulfate2DNetworkVgg11Style, Sulfate2DUpNetwork, Sulfate2DNetworkUpVgg11Style
from ._utils import compute_masks, compute_pmf
from typing import Final
import os

if not any(map(lambda name: name.startswith("ANDROID_"), os.environ)):
    from ._dataset import Sulfate2DSampleDataset, ProcessedSulfate2DSampleDataset


NETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "Sulfate2dNetwork.ckpt"))
PCA_STATS: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "Sulfate2dPcaStats.npz"))
UPNETWORK_CHECKPOINT: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_resources", "Sulfate2dUpNetwork.ckpt"))