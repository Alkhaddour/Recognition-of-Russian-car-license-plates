import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import transform, dataset_meta, img_c, img_h, img_w, n_maps, default_optimizer, lr, loss_function, n_epochs, \
    device, batch_size, models_dir
from models import CNN_1
from trainer import train_model, test_model
from utils.basic_utils import make_valid_path
from utils.data_util import ImagesDataset
from utils.display_util import plot_torch_image, plot_losses
from utils.models_util import ModelManager


def test_data_loader():
    meta = pd.read_csv(dataset_meta)
    image_paths = meta['img_path'].tolist()
    labels = meta['label'].tolist()
    dataset = ImagesDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for idx, imgs, labels in dataloader:
        for id, img, lbl in zip(idx, imgs, labels):
            print(id)
            print(img.shape)
            plot_torch_image(img)
        break


def test_model_CNN1():
    meta = pd.read_csv(dataset_meta)
    image_paths = meta['img_path'].tolist()
    labels = meta['label'].tolist()
    dataset = ImagesDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN_1(img_c, img_h, img_w, n_maps)
    for idx, imgs, labels in dataloader:
        outputs = model(imgs)
        break
    print(outputs)


if __name__ == "__main__":
    # test_data_loader()
    meta = pd.read_csv(dataset_meta)
    image_paths = meta['img_path'].tolist()
    labels = meta['label'].tolist()
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.33, random_state=42)
    print(sum(y_train))
    print(sum(y_val))

    train_set = ImagesDataset(X_train, y_train, transform=transform)
    val_set = ImagesDataset(X_val, y_val, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model = CNN_1(img_c, img_h, img_w, n_maps)
    optimizer = default_optimizer(model.parameters(), lr=lr)
    model_manager = ModelManager("First model", make_valid_path(models_dir, is_dir=True))
    model, model_manager = train_model(model, model_manager, train_loader, val_loader, loss_function, optimizer,
                                       n_epochs, device)
    # print(model_manager.train_losses)
    plot_losses("Train", model_manager.train_losses)
    plot_losses("Validation", model_manager.val_losses)
    model_manager = ModelManager("First model", make_valid_path(models_dir, is_dir=True))
    model, loss = model_manager.load_checkpoint("First model.pkl", model)
    test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    labels, predictions = test_model(model, test_loader, device)
    print("F1-score: ", f1_score(labels, predictions))
