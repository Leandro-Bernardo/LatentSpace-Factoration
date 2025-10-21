import torch
import torch.nn as nn

class vanilla(torch.nn.Module):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__()
        
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1) # Alterar o tamanho da entrada (baseado no pmf.shape)
        self.layer_classfier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool2d(13)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv10(x)
        x = self.output_layer(x)

        return x