from datetime import datetime

import numpy as np
import sklearn
import torch
from tqdm import tqdm

from config import weights
from utils.models_util import get_best_threshold


def train_model(model,model_manager, train_loader, val_loader, loss_fn, optimizer, n_epochs, device):
    train_losses =[]
    val_losses = []
    model = model.to(device)
    for epoch in range(n_epochs):
        # Training
        model.train()
        for i, (idxs, images, labels) in enumerate(train_loader):
            class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight=weights,
                                                                            classes=np.unique(labels),
                                                                            y=labels.numpy())
            class_weights=torch.tensor(class_weights, dtype=torch.float).to(device)
            # Load a batch & transform to vectors
            images = images.to(device)
            labels = labels.to(device).long()
            # Forward
            outputs = model(images)

            train_loss = loss_fn(class_weights)(outputs.float(), labels)
            # Adjust the parameters using backprop
            train_loss.backward()
            # Compute gradients
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()
            # Inspect the losses
            model_manager.update_train_loss(train_loss, i, len(train_loader), epoch, n_epochs)

        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, (idxs, images, labels) in enumerate(val_loader):
                class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight=weights,
                                                                                classes=np.unique(labels),
                                                                                y=labels.numpy())
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                # Load a batch & transform to vectors
                images = images.to(device)
                labels = labels.to(device).long()
                # Forward
                outputs = model(images)
                # Loss
                val_loss = loss_fn(class_weights)(outputs.float(), labels)
                model_manager.update_val_loss(val_loss, i, len(val_loader), epoch, n_epochs)
                model_manager.update_model(model, val_loss)

    return model, model_manager


def test_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        print(f'[{datetime.now()}] -- Testing...')
        for _, images, lbls in tqdm(test_loader):
            images = images.to(device)
            prediction = model(images).detach().cpu().numpy()
            predictions += list(prediction)
            labels += list(lbls.detach().cpu().numpy())

    labels = [int(x) for x in labels]
    predictions = [np.argmax(x) for x in predictions]
    return labels, predictions

#