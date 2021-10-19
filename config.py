import torch
from torch import nn
from torchvision import transforms

# paths
dataset_meta_1 = './dataset/classification/train_labelled.csv'
dataset_meta_2 = './dataset/classification/train_unlabelled.csv'
val_meta = './dataset/classification/val.csv'
rus_nan_meta = './dataset/classification/invalid_and_russian.csv'
prediction_path = './dataset/classification/predictions.csv'
models_dir = './output/models/'
# Transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((100, 300)),
     transforms.Normalize((0.0594, 0.0584, 0.0593), (0.1871, 0.1851, 0.1879))
     ])
# Model params
lr = 0.0001
img_c = 3
img_h = 100
img_w = 300
n_maps = 32
n_epochs = 20
batch_size = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
default_optimizer = torch.optim.Adam
loss_function = nn.CrossEntropyLoss
weights = {0: 1, 1: 1}
print_step = 1

