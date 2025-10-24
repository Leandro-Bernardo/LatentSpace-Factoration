import torch
import yaml
import vanilla_feature_extractor
import mlp_feature_extractor

data = yaml.safe_load('devices_mapper.yaml')
num_class =  ((data['alkalinity'])[-1] + 1)
input_size = 756 

vanilla_squeeze = vanilla_feature_extractor.Vanilla_feature_extractor(input_size, num_class)


# Classificador pro script do David
class Fluttershy(torch.nn.Module):
    def __init__(self, device: str = "cuda", **kwargs):
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

class VanillaClassifier(torch.nn.Module):
    """_Classificador original da arquitetura SqueezeNet_
        Utilizando o classificador com a mesma arquitetura da Squeeze Net original, uma camada de convolução + avg pooling
    """
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__()
        
        final_conv = torch.nn.Conv2d(512, self.num_classes, kernel_size=1) # Alterar o tamanho da entrada (baseado no pmf.shape)
        self.layer_classfier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5), final_conv, torch.nn.ReLU(inplace=True), torch.nn.AvgPool2d(13)
        )

    def forward(self, x: torch.Tensor):
        x = self.final_conv(x)
        x = torch.softmax(x)
        return x
    
