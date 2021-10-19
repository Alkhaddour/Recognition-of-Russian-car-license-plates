import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import transform, img_c, img_h, img_w, n_maps, default_optimizer, lr, loss_function, n_epochs, \
    device, batch_size, models_dir, dataset_meta_2, dataset_meta_1, rus_nan_meta
from models import CNN_1
from train_test import train_model, test_model
from utils.basic_utils import make_valid_path
from utils.data_util import ImagesDataset
from utils.display_util import plot_torch_image, plot_losses
from utils.models_util import ModelManager
import random

# data = pd.read_csv("./dataset/classification/aggregated_results_by_ds__pool_28674905__2021_10_16.tsv", sep='\t')
#
# cols = {'INPUT:img_path': 'img_path',
#         'OUTPUT:label': 'label'}
# data = data[cols.keys()]
# data.columns = list(cols.values())
#
#
# def get_name(path):
#     return os.path.join('dataset/classification/train_unlabelled/',
#                         os.path.split(path)[1])
#
#
# def change_label(lbl):
#     if lbl == -1:
#         lbl = 0
#     return lbl
#
#
# data = data.loc[data['label'] != 0]
# data['img_path'] = data['img_path'].apply(get_name)
# data['label'] = data['label'].apply(change_label)
# print(data['label'].unique())
# print(data.groupby(['label']).count())
# print(data.groupby(['label']).count())
# print(data['label'].unique())
#
# data.to_csv('./dataset/classification/invalid_and_russian.csv', index=False)

if __name__ == "__main__":
    # test_data_loader()
    meta_train_val = pd.read_csv(rus_nan_meta)
    imgs_train_val = meta_train_val['img_path'].tolist()
    lbls_train_val = meta_train_val['label'].tolist()

    all_train_val_set = ImagesDataset(imgs_train_val, lbls_train_val, transform=transform)
    all_train_val_loader = DataLoader(all_train_val_set, batch_size=batch_size, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(imgs_train_val, lbls_train_val, test_size=0.25, random_state=42)
    print(sum(y_train))
    print(sum(y_val))

    train_set = ImagesDataset(X_train, y_train, transform=transform)
    val_set = ImagesDataset(X_val, y_val, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model = CNN_1(img_c, img_h, img_w, n_maps)
    print("params count = ", sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = default_optimizer(model.parameters(), lr=lr)
    model_manager = ModelManager(f"model_rus_nan", make_valid_path(models_dir, is_dir=True))
    model, model_manager = train_model(model, model_manager, train_loader, val_loader, loss_function, optimizer,
                                       n_epochs, device)
    plot_losses(f"Train", model_manager.train_losses)
    plot_losses(f"Validation", model_manager.val_losses)
    model_manager = ModelManager(f"model_rus_nan", make_valid_path(models_dir, is_dir=True))
    model, loss = model_manager.load_checkpoint(f"model_rus_nan.pkl", model)
    lbls_train, preds_train = test_model(model, train_loader, device)
    lbls_val, preds_val = test_model(model, val_loader, device)

        # results
    f1_socre_train = f1_score(lbls_train, preds_train)
    f1_socre_val = f1_score(lbls_val, preds_val)
    print(f"F1-score train: ", f1_socre_train)
    print(f"F1-score val: ", f1_socre_val)
  