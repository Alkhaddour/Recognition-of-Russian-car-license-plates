import os
from datetime import datetime

import torch

from config import print_step


class ModelManager:
    def __init__(self, model_name, models_dir):
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('Inf')
        self.models_dir = models_dir
        self.model_name = model_name

    def save_checkpoint(self, chkpt_name, model, valid_loss):
        """
        Saves the model weights with current validation loss
        """
        torch.save({'model_state_dict': model.state_dict(), 'valid_loss': valid_loss},
                   os.path.join(self.models_dir, chkpt_name))

    def load_checkpoint(self, chkpt_path, model):
        """
        Load model weights from file
        """
        state_dict = torch.load(os.path.join(self.models_dir, chkpt_path))
        model.load_state_dict(state_dict['model_state_dict'])
        return model, state_dict['valid_loss']

    def save_metrics(self, metrics_file):
        state_dict = {'train_losses': self.train_losses,
                      'val_losses': self.val_losses}

        torch.save(state_dict, os.path.join(self.models_dir, metrics_file))

    def load_metrics(self, metrics_file):
        state_dict = torch.load(os.path.join(self.models_dir, metrics_file))
        return state_dict['train_losses'], state_dict['val_losses']

    def update_train_val_loss(self, model, train_loss, val_loss, epoch, num_epochs):
        """
        Prints the model training progress and save the model whenever we got a better version
        """
        model_file_name = self.model_name + '.pkl'
        metrics_file_name = self.model_name + '_metrics.pkl'
        train_loss = train_loss
        val_loss = val_loss
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        print(f'[{datetime.now()}] -- Epoch [{epoch}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        # checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(model_file_name, model, self.best_val_loss)
            self.save_metrics(metrics_file_name)
            return True
        else:
            return False

    def update_train_loss(self, train_loss, step, total_steps, epoch, num_epochs, print_freq=print_step):
        if (step + 1) % print_freq == 0:
            self.train_losses.append(train_loss.item())
            print(f'[{datetime.now()}] -- Training: Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{total_steps}], '
                  f'Train Loss: {train_loss:.4f}')

    def update_val_loss(self, val_loss, step, total_steps, epoch, num_epochs, print_freq=10):
        if (step + 1) % print_freq == 0:
            self.val_losses.append(val_loss.item())
            print(f'[{datetime.now()}] -- Validation: Epoch [{epoch + 1}/{num_epochs}], '
                  f'Step [{step + 1}/{total_steps}], Val Loss: {val_loss:.4f}')

    def update_model(self, model, val_loss):
        model_file_name = self.model_name + '.pkl'
        metrics_file_name = self.model_name + '_metrics.pkl'
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(model_file_name, model, self.best_val_loss)
            self.save_metrics(metrics_file_name)
            return True
        else:
            return False


# TODO: implement function: get_best_threshold
def get_best_threshold(preds, labels):
    return 0.5
