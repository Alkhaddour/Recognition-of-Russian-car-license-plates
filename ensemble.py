import copy
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import transform, img_c, img_h, img_w, n_maps, default_optimizer, lr, device, batch_size, \
    models_dir, dataset_meta_2, dataset_meta_1, prediction_path, val_meta, prediction_path2
from models import CNN_1
from train_test import test_model
from utils.basic_utils import make_valid_path
from utils.data_util import ImagesDataset
from utils.models_util import ModelManager
if __name__ == "__main__":
    preds = {}
    # test_data_loader()
    meta_test = pd.read_csv(val_meta)
    imgs_test = meta_test['img_path'].tolist()
    lbls_test = meta_test['label'].tolist()

    test_set = ImagesDataset(imgs_test, lbls_test, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = CNN_1(img_c, img_h, img_w, n_maps)
    print("params count = ", sum([np.prod(p.size()) for p in model.parameters()]))

    for model_name in os.listdir('./output/models/'):
        if 'metric' in model_name or 'rus_nan' in model_name:
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
                      columns=['img_path', 'label'])
    # sort and save results
    df['img_path'] = df['img_path'].apply(os.path.basename)
    df['tmp'] = df['img_path'].apply(lambda x: x[:-4])
    df['tmp'] = df['tmp'].astype(int)
    df = df.sort_values(['tmp'])
    del df['tmp']
    df.to_csv(prediction_path, index=False)
    print("--------------- One more filter -----------------")
    #  define one more filter model:
    model_manager = ModelManager(None, make_valid_path(models_dir, is_dir=True))
    model, loss = model_manager.load_checkpoint('model_rus_nan.pkl', model)
    optimizer = default_optimizer(model.parameters(), lr=lr)
    imgs_test = df.loc[df['label'] == 1]['img_path'].tolist()
    imgs_test = [os.path.join('./dataset/classification/val', x) for x in imgs_test]
    lbls_test = [-1] * len(imgs_test)
    test_set = ImagesDataset(imgs_test, lbls_test, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    lbls_test, preds_test, ids = test_model(model, test_loader, device, with_samples_id=True)
    df2 = pd.DataFrame(list(zip(ids, preds_test)),
                       columns=['img_path', 'label'])
    df2['img_path'] = df2['img_path'].apply(os.path.basename)
    df = pd.concat([df, df2])
    df = df.sort_values(['img_path', 'label'])
    df = df.groupby('img_path').first().reset_index()
    # sort and save results
    df['tmp'] = df['img_path'].apply(lambda x: x[:-4])
    df['tmp'] = df['tmp'].astype(int)
    df = df.sort_values(['tmp'])
    del df['tmp']
    df.to_csv(prediction_path2, index=False)  # got slightly better results
    # print(f"F1 at threshold {threshold}", f1_score(targets, outputs))

    # files_to_copy = df.loc[df['label'] == 1]['img_path'].tolist()
    # for filename in files_to_copy:
    #     shutil.copy(os.path.join('./dataset/classification/train_unlabelled', filename),
    #                 os.path.join('./dataset/classification/rus', filename))
