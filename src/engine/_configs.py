import os
from pathlib import Path
from typing import Literal, Optional, Dict, Tuple, Any
import yaml
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT_DIR = Path(__file__).resolve().parent.parent


class PreprocessingConfig(BaseModel):
    samples_dir: str = Field(description="Dataset dir.")
    cache_dir: Optional[str] = Field(description="Cache dir")
    debug_save_pmfs_as_img: bool = Field(default=False,
                                        description= "Saves the processed pmf`s as images for a fast validation of the data.")
    fine_tune_cnn: bool = Field(default=False,
                                        description= """Set to True when fine-tuning the CNN backbone end-to-end (raw inputs required).
                                                        Set to False when training only the classification model on frozen feature maps.
                                                        If True, saves the PMF (raw inputs) to disk instead of extracted feature maps.""")


    @field_validator("samples_dir")
    @classmethod
    def validate_samples_dir(cls, value: str) -> str:
        path = ROOT_DIR / value if not os.path.isabs(value) else Path(value)
        if not path.exists():
            raise ValueError(f"Diretório de amostras não encontrado: {path}")
        return str(path)

class TrainingSetup(BaseModel):
    max_epochs: int = Field(default=500, gt=0,
                            description= "Maximum number of Epochs")
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0,
                                description= "The learning rate of the model. Value is defined by the WandB sweep. Default is only used for debug purposes.")
    learning_rate_patience: int = Field(default=10, ge=0,
                                        description= "Number of Epochs before reducing the learning rate after reaching a plateau.")
    early_stopping_patience: Optional[int] = Field(default=None,
                                                    description= "Number of Epochs before early stopping the training after reaching a plateau")
    loss_function: Literal["cross_entropy"] = Field(default="cross_entropy",
                                                    description= "The cost function used on training")

    @model_validator(mode="after")
    def compute_early_stopping(self):
        if self.early_stopping_patience is None:
            self.early_stopping_patience = 2 * self.learning_rate_patience + 1
        return self

    def get_loss_module(self) -> nn.Module:
        available_losses = {
                            "cross_entropy": nn.CrossEntropyLoss
                            }
        return available_losses[self.loss_function]()

class ExperimentConfig(BaseModel):
    analyte: Literal[
                    "alkalinity", "bisulfite", "bisulfite2d", "chloride",
                    "sulfate", "phosphate", "iron2", "iron3", "ph", "redox"
                ] = Field(default=None,
                            description="The analyte.")
    classifier_model: Literal["mlp1", "dynamicMLP", "squeezenet"] = Field(default=None,
                                                                            description="The classifier architecture used for predicting the device.")
    feature_extractor: Literal["squeezenet", "vgg11"] = Field(default="squeezenet",
                                                                description="The architecture used for extracting features from input images. The CNN module.")
    fine_tune_cnn: bool = Field(default=False,
                                    description= "Indicates if the CNN module used for extract features maps will be fine-tuned or not")
    return_node: Optional[str] = Field(default=None,
                                        description="The node from the CNN model used for colecting the feature maps. Set to none to use the last CNN block.")

    # Sub-configs
    preprocessing: PreprocessingConfig
    model: TrainingSetup

    # Sweep parameters (WandB)
    batch_size: int = Field(default=32, gt=0,
                            description= "Batch size. Value is defined by the WandB sweep. Default is only used for debug purposes.")
    gradient_clip: float = Field(default=0.5, ge=0.0,
                                description= "Gradient Clip. Value is defined by the WandB sweep. Default is only used for debug purposes.")

    @classmethod
    def from_yaml(cls, settings_filename: str = "settings.yaml") -> "ExperimentConfig":
        settings_path = ROOT_DIR / settings_filename
        if not settings_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {settings_path}")

        with open(settings_path, "r", encoding="utf-8") as f:
            raw_settings = yaml.load(f, Loader=yaml.FullLoader)

        # Maping and final structure
        analyte = raw_settings["analyte"]
        cache_dir = ROOT_DIR / "cache_dir" / analyte

        merged = {
                "analyte": analyte,
                "feature_extractor": raw_settings.get("feature_extractor", "squeezenet"),
                "return_node": raw_settings.get("return_node"),
                "classifier_model": raw_settings.get("classifier_model"),
                "fine_tune_cnn": raw_settings.get("fine_tune_cnn", False),
                "preprocessing": {
                    "samples_dir": raw_settings.get("samples_dir"),
                    "cache_dir": str(cache_dir),
                    "debug_save_pmfs_as_img": raw_settings.get("save_pmf_as_img", False),
                    "fine_tune_cnn": raw_settings.get("fine_tune_cnn", False),
                                },
                "model": raw_settings.get("model", {}),
                }

        return cls.model_validate(merged)

    def update_from_wandb(self, wandb_configs: Dict[str, Any]) -> "ExperimentConfig":
        """Updates WandB`s hyperparemeters in runtine ."""
        current_data = self.model_dump()

        for key, val in wandb_configs.items():
            if key in current_data:
                current_data[key] = val
            elif key in current_data["model"]:
                current_data["model"][key] = val

        return ExperimentConfig.model_validate(current_data)

    def get_classifier_architecture(self) -> Tuple[str, bool]:
        # Lazy initialization
        from .models import MLP1, DynamicMLP, SqueezeNetClassifier

        mapping = {
            "mlp1": {"model": MLP1, "requires_flatten": True},
            "dynamicMLP": {"model": DynamicMLP, "requires_flatten": True},
            "squeezenet": {"model": SqueezeNetClassifier, "requires_flatten": False},
        }
        chosen = mapping[self.classifier_model]
        return (chosen["model"], chosen["requires_flatten"])

class DatasetMetadata(BaseModel):
    num_classes: int
    num_samples: int
    num_channels: int
    height: int
    width: int

    @classmethod
    def from_yaml(cls, filename: str) -> "DatasetMetadata":
        path = ROOT_DIR / ".." / "processed_dataset" / filename
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls.model_validate(data)