import torch
import torch.nn as nn



# TODO relacionar as features com o tamanho das imagens


class vanilla(torch.nn.Module):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=832, out_features=512, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.4007686950902262, inplace=False))

        self.l7 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=512, out_features=512, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l6 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=512, out_features=256, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.4324747397573356, inplace=False))
        self.l5 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=256, out_features=256, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.22073755838360256, inplace=False))
        self.l4 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=256, out_features=128, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.22073755838360128, inplace=False))
        self.l3 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=128, out_features=128, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.1166385125068351, inplace=False))
        self.l2 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=128, out_features=16, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0.1166385125068351, inplace=False))
        self.l1 = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=16, out_features=16, bias=True),
                                    torch.nn.ReLU(),)
                                    #torch.nn.Dropout(p=0, inplace=False))
        self.output_layer =  torch.nn.Sequential(
                                    torch.nn.Linear(in_features=16, out_features=1, bias=True))

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.l7(x)
        x = self.l6(x)
        x = self.l5(x)
        x = self.l4(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l1(x)
        x = self.output_layer(x)

        return x