import timm
import timm.data
import torch
import torch.nn as nn
from PIL import Image


class PerceptionEncoder(nn.Module):
    def __init__(
        self,
        model_name="vit_pe_core_large_patch14_336",
        img_size: None | int = None,
    ):
        super().__init__()

        dynamic_img_size = img_size is not None

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=dynamic_img_size,
        )

        # Get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.backbone)

        if img_size is not None:
            data_config["input_size"] = (3, img_size, img_size)

        self._preprocess = timm.data.create_transform(**data_config, is_training=False)

        # Remove head
        self.backbone.head = nn.Identity()

        self.features_dim: int = int(getattr(self.backbone, "num_features"))

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def get_features_dim(self) -> int:
        return self.features_dim


if __name__ == "__main__":
    import autorootcwd  # noqa: F401

    from src.config import Backbone
    from src.encoders._common import inference

    model = PerceptionEncoder(Backbone.PerceptionEncoder_B_p16_224.value, img_size=224)
    inference(model)
