import os

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import transform, img_c, img_h, img_w, n_maps, default_optimizer, lr, loss_function, n_epochs, \
    device, batch_size, models_dir, dataset_meta_2, dataset_meta_1
from models import CNN_1
from trainer import train_model, test_model
from utils.basic_utils import make_valid_path
from utils.data_util import ImagesDataset
from utils.display_util import plot_torch_image, plot_losses
from utils.models_util import ModelManager
import  random

# torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # test_data_loader()
    meta_test = pd.read_csv(dataset_meta_1)
    meta_train_val = pd.read_csv(dataset_meta_2)
    imgs_train_val = meta_train_val['img_path'].tolist()
    lbls_train_val = meta_train_val['label'].tolist()
    imgs_test = meta_test['img_path'].tolist()
    lbls_test = meta_test['label'].tolist()

    test_set = ImagesDataset(imgs_test, lbls_test, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    all_train_val_set = ImagesDataset(imgs_train_val, lbls_train_val, transform=transform)
    all_train_val_loader = DataLoader(all_train_val_set, batch_size=batch_size, shuffle=False)

    ru_indexes = [i for i in range(len(lbls_train_val)) if lbls_train_val[i] == 1]
    non_ru_ind = [i for i in range(len(lbls_train_val)) if i not in ru_indexes]
    n_splits = 10
    split_size = len(non_ru_ind) // n_splits
    f1_scores = []
    for i in range(n_splits):
        # seed = random.randint(0, 100)
        # print("seed = ", seed)
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        torch.cuda.empty_cache()
        imgs_0_ind = non_ru_ind[i * split_size: (i + 1) * split_size]
        imgs_0 = [imgs_train_val[k] for k in imgs_0_ind]
        lbls_0 = [0] * len(imgs_0)
        imgs_1 = [imgs_train_val[k] for k in ru_indexes]
        lbls_1 = [1] * len(imgs_1)
        imgs = imgs_0 + imgs_1
        lbls = lbls_0 + lbls_1

        X_train, X_val, y_train, y_val = train_test_split(imgs, lbls, test_size=0.25, random_state=42)
        print(sum(y_train))
        print(sum(y_val))

        train_set = ImagesDataset(X_train, y_train, transform=transform)
        val_set = ImagesDataset(X_val, y_val, transform=transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        model = CNN_1(img_c, img_h, img_w, n_maps)
        print("params count = ", sum([np.prod(p.size()) for p in model.parameters()]))
        optimizer = default_optimizer(model.parameters(), lr=lr)
        model_manager = ModelManager(f"model_{i}", make_valid_path(models_dir, is_dir=True))
        model, model_manager = train_model(model, model_manager, train_loader, val_loader, loss_function, optimizer,
                                           n_epochs, device)
        # print(model_manager.train_losses)
        plot_losses(f"Train_{i}", model_manager.train_losses)
        plot_losses(f"Validation_{i}", model_manager.val_losses)
        model_manager = ModelManager(f"model_{i}", make_valid_path(models_dir, is_dir=True))
        model, loss = model_manager.load_checkpoint(f"model_{i}.pkl", model)
        lbls_train, preds_train = test_model(model, train_loader, device)
        lbls_val, preds_val = test_model(model, val_loader, device)
        lbls_test, preds_test = test_model(model, test_loader, device)
        lbls_all_train_val, preds_all_train_val = test_model(model, all_train_val_loader, device)

        print(f"F1-score train{i}: ", f1_score(lbls_train, preds_train))
        print(f"F1-score val{i}: ", f1_score(lbls_val, preds_val))
        print(f"F1-score test{i}: ", f1_score(lbls_test, preds_test))
        print(f"F1-score all_train_val{i}: ", f1_score(lbls_all_train_val, preds_all_train_val))

        f1_scores.append(f1_score(lbls_test, preds_test))
    print(f1_scores)
