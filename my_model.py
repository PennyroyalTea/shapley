import torchvision.models as models

class myResnet18(models.resnet18):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 10)