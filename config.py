import torch
from torch import nn
from torchvision import transforms

# paths
dataset_meta = './dataset/classification/train_labelled.csv'
models_dir = './output/models/'
# Transforms
# TODO: check the best transform to use
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((100, 300)),  # Hint: this might not be the best way to resize images
     transforms.Normalize((0.2, 0.2, 0.2), (0.5, 0.5, 0.5))  # Hint: this might not be the best normalization
     ])

# Model params
lr = 0.0001
img_c = 3
img_h = 100
img_w = 300
n_maps = 64
n_epochs = 10
batch_size = 40

device = 'cuda' if torch.cuda.is_available() else 'cpu'
default_optimizer = torch.optim.Adam
loss_function = nn.CrossEntropyLoss
weights = {0: 1, 1: 100}
print_step = 1
