import torch


    
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
    
    
    
#Classificador telefones

class Applejack(torch.nn.Module):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=65536, out_features=32768, bias=True),
                                    torch.nn.ReLU(),)

        self.l12 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=32768, out_features=16384, bias=True),
                                    torch.nn.ReLU(),)
        self.l11 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=16384, out_features=8192, bias=True),
                                    torch.nn.ReLU(),)
        self.l10 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=8192, out_features=4096, bias=True),
                                    torch.nn.ReLU(),)
        self.l9 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
                                    torch.nn.ReLU(),)
        self.l8 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
                                    torch.nn.ReLU(),)
        self.l7 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=1024, out_features=512, bias=True),
                                    torch.nn.ReLU(),)
        self.l6 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=512, out_features=256, bias=True),
                                    torch.nn.ReLU(),)
        self.l5 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=256, out_features=128, bias=True),
                                    torch.nn.ReLU(),)
        self.l4 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=128, out_features=64, bias=True),
                                    torch.nn.ReLU(),)
        self.l3 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=64, out_features=32, bias=True),
                                    torch.nn.ReLU(),)
        self.l2 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=32, out_features=16, bias=True),
                                    torch.nn.ReLU(),)
        self.l1 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=16, out_features=8, bias=True),
                                    torch.nn.ReLU(),)

        
        self.output_layer =  torch.nn.Sequential(
                                    torch.nn.Linear(in_features=8, out_features=5, bias=True))

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.l12(x)
        x = self.l11(x)
        x = self.l10(x)
        x = self.l9(x)
        x = self.l8(x)
        x = self.l7(x)
        x = self.l6(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x