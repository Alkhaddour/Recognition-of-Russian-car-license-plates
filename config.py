import torch
from torch import nn
from torchvision import transforms

# paths
dataset_meta_1 = './dataset/classification/train_labelled.csv'
dataset_meta_2 = './dataset/classification/train_unlabelled.csv'

models_dir = './output/models/'
# Transforms
# TODO: check the best transform to use
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((100, 300)),  # Hint: this might not be the best way to resize images
     # transforms.Normalize((0.0592, 0.0586, 0.0594), (0.1868, 0.1859, 0.1883)) # using 1000 imgaes only
     transforms.Normalize((0.0594, 0.0584, 0.0593), (0.1871, 0.1851, 0.1879))  # using 8500 imgaes
     ])
MN_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
