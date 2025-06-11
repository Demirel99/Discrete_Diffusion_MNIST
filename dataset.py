# file: dataset.py
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class Binarize:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        return (tensor > self.threshold).float()

def get_dataloader(batch_size=128, data_root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        Binarize(threshold=0.5)
    ])

    train_dataset = MNIST(root=data_root, train=True, download=True, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    return dataloader