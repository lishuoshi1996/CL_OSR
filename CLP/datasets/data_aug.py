import os
from torchvision import transforms
from datasets.gaussian_blur import GaussianBlur
from PIL import Image

def contrastive_train_transform(img_size,s=1):

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    train_transforms = transforms.Compose([transforms.Resize(img_size),
                                           transforms.RandomCrop(img_size, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomApply([color_jitter], p=0.8),
                                           transforms.RandomGrayscale(p=0.2),
                                           GaussianBlur(kernel_size=int(0.1 * img_size)),
                                           transforms.ToTensor(),
                                         ])
    return train_transforms

