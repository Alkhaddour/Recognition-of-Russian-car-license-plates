import copy
import os

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import transform, img_c, img_h, img_w, n_maps, default_optimizer, lr, loss_function, n_epochs, \
    device, batch_size, models_dir, dataset_meta_2, dataset_meta_1, prediction_path
from models import CNN_1
from train_test import train_model, test_model
from utils.basic_utils import make_valid_path
from utils.data_util import ImagesDataset
from utils.display_util import plot_torch_image, plot_losses
from utils.models_util import ModelManager
import random

# torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    preds = {}
    # test_data_loader()
    meta_test = pd.read_csv(dataset_meta_2)
    imgs_test = meta_test['img_path'].tolist()
    lbls_test = meta_test['label'].tolist()

    test_set = ImagesDataset(imgs_test, lbls_test, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = CNN_1(img_c, img_h, img_w, n_maps)
    print("params count = ", sum([np.prod(p.size()) for p in model.parameters()]))

    for model_name in os.listdir('./output/models/'):
        if 'metric' in model_name:
            continue
        print("Testing model: ", model_name)
        model_manager = ModelManager(None, make_valid_path(models_dir, is_dir=True))
        model, loss = model_manager.load_checkpoint(model_name, model)
        optimizer = default_optimizer(model.parameters(), lr=lr)
        lbls_test, preds_test, ids = test_model(model, test_loader, device, with_samples_id=True)
        for idx, id in enumerate(ids):
            if id in preds:
                preds[id]['preds'] += preds_test[idx]
            else:
                preds[id] = {}
                preds[id]['preds'] = preds_test[idx]
                preds[id]['true'] = lbls_test[idx]

    preds_orig = copy.deepcopy(preds)
    threshold = 10
    preds = copy.deepcopy(preds_orig)
    for img, res in preds.items():
        if res['preds'] >= threshold:
            res['preds'] = 1
        else:
            res['preds'] = 0

    targets = []
    outputs = []
    imgs = []
    for img, res in preds.items():
        targets.append(res['true'])
        outputs.append(res['preds'])
        imgs.append(img)
    df = pd.DataFrame(list(zip(imgs, outputs)),
                      columns=['image', 'label'])
    df.to_csv(prediction_path, index=False)

    print(f"F1 at threshold {threshold}", f1_score(targets, outputs))