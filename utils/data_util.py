import cv2
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, image_paths, labels, pin=False, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.targets = self.labels
        self.transform = transform
        self.pin = pin
        if pin is True:
            self.images = self.__load_images__()

        assert len(self.image_paths) == len(self.labels), "Number of images is not equal to the number of labels"

    def __load_images__(self):
        # load all data to memory
        images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            img = img[:, :, :3]
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        return images

    def __get_image__(self, idx):
        if self.pin is True:
            return self.images[idx]
        else:
            img = cv2.imread(self.image_paths[idx])
            img = img[:, :, :3]
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return idx, self.__get_image__(idx), self.labels[idx]



