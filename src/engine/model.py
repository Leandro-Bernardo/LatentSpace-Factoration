import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import models, transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model="all")

# Hyperparameters 
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 100

class SNModel(LightningModule):
  def __init__(self, input_size, num_classes):
    super().__init__()
    self.backbone = models.squeezenet1_1(weights="DEFAULT")
    for param in self.backbone.parameters(): # TODO: Tentar freezar da forma como o Lightning implementa
      param.requires_grad = False
    #for param in self.backbone.classifier.parameters(): 
    #  param.requires_grad = True
    #self.backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)) 
    self.last_layer = create_feature_extractor(self.backbone, ['features.12.cat'])
    
    self.loss_fn = nn.CrossEntropyLoss()
    
  
  def forward(self, x):  
    x = self.last_layer(x)
    x = x.get('features.12.cat')
    x = self.backbone.classifier(x)
    return x

  def training_step(self, batch, batch_idx):
    loss, scores, y = self._commons_step(batch, batch_idx)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    loss, scores, y = self._commons_step(batch, batch_idx)
    self.log('val_loss', loss)
    return loss

  def test_step(self, batch, batch_idx):
    loss, scores, y = self._commons_step(batch, batch_idx)
    self.log('test_loss', loss)
    return loss

  def _commons_step(self, batch, batch_idx):
    x, y = batch
    score = self.forward(x)
    score = score.view(score.size(0), -1)  
    y = y.long()                      
    loss = self.loss_fn(score, y)
    return loss, score, y

  def predict_step(self, batch, batch_idx):
    x, y = batch
    score = self.forward(x)
    preds = torch.argmax(score, dim=1)
    return preds

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dados
entire_dataset = datasets.MNIST(root="dataset/", train=True, transform=vgg_transform, download=True)
test_ds = datasets.MNIST(root="dataset/", train=False, transform=vgg_transform, download=True)
train_ds, val_ds = random_split(entire_dataset, [50000, 10000])

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

model = SNModel(input_size=input_size, num_classes=num_classes).to(device)
model.freeze()

trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], accelerator="gpu", min_epochs=1, max_epochs=num_epochs, devices=[0], precision=32, logger=wandb_logger)
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader)
trainer.test(model, test_loader)